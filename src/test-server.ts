import type { Server } from 'node:http';

/**
 * Return a URL that reaches a test server through the same IP family it bound.
 * Node binds an omitted host to IPv6 when available, which is not always
 * dual-stack in CI/WSL environments.
 */
export function testServerBaseUrl(server: Server | undefined, port: number): string {
  const address = server?.address();
  if (!address || typeof address === 'string') {
    throw new Error('Test server is not listening on a TCP address');
  }
  const host = address.family === 'IPv6' ? '[::1]' : '127.0.0.1';
  return `http://${host}:${port}`;
}
