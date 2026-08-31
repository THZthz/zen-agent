import { afterEach, describe, expect, it } from 'vitest';
import { runChatCompletions } from './chat-completions.js';
import { testServerBaseUrl } from './test-server.js';

let server: import('node:http').Server | undefined;

afterEach(() => {
  server?.closeAllConnections?.();
  server?.close();
  server = undefined;
});

function startServer(
  handler: (
    req: import('node:http').IncomingMessage,
    res: import('node:http').ServerResponse,
  ) => void,
): Promise<number> {
  return new Promise((resolve) => {
    const srv = require('node:http').createServer(handler);
    server = srv;
    srv.listen(0, () => resolve((srv.address() as import('node:net').AddressInfo).port));
  });
}

describe('pi-ai message adapter', () => {
  it('preserves named users and converts supported media to the OpenAI wire format', async () => {
    let requestBody: Record<string, unknown> | undefined;
    const port = await startServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on('data', (chunk: Buffer) => chunks.push(chunk));
      req.on('end', () => {
        requestBody = JSON.parse(Buffer.concat(chunks).toString('utf8'));
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.end(
          `data: ${JSON.stringify({
            choices: [{ index: 0, delta: { content: 'ok' }, finish_reason: 'stop' }],
          })}\n\ndata: [DONE]\n\n`,
        );
      });
    });

    const result = await runChatCompletions({
      baseUrl: testServerBaseUrl(server, port),
      apiKey: 'test',
      provider: 'test',
      compat: { supportsDeveloperRole: false },
      label: 'Test',
      model: 'test-model',
      system: 'system',
      messages: [
        {
          role: 'user',
          name: 'Amias',
          content: [
            { type: 'text', text: 'look at these' },
            { type: 'image', mimeType: 'image/jpeg', data: 'QUJD' },
            { type: 'audio', mimeType: 'audio/wav', data: 'UklGRg==' },
            { type: 'audio', mimeType: 'audio/ogg', data: 'T0dn' },
          ],
        },
      ],
    });

    expect(result.text).toBe('ok');
    expect(requestBody?.messages).toEqual([
      { role: 'system', content: 'system' },
      {
        role: 'user',
        name: 'Amias',
        content: [
          { type: 'text', text: 'look at these' },
          { type: 'image_url', image_url: { url: 'data:image/jpeg;base64,QUJD' } },
          { type: 'input_audio', input_audio: { data: 'UklGRg==', format: 'wav' } },
          { type: 'text', text: '[audio attached (audio/ogg) omitted: unsupported format]' },
        ],
      },
    ]);
  });
});
