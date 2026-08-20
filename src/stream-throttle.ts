export type StreamKind = "thought" | "message";

/**
 * Batches streamed LLM deltas into small, throttled chunks.
 *
 * Zed renders each `agent_message_chunk`/`agent_thought_chunk` update
 * incrementally (`agent_ui/.../thread_view.rs`), and flooding it with
 * per-token updates across a slow JSON-RPC pipe makes the UI janky and drops
 * chunks. Throttling to ~24 chars per 16ms tick keeps the output visibly
 * streaming while staying well within Zed's per-update handling budget.
 */
export class StreamThrottle {
  private queue: Array<{ kind: StreamKind; text: string }> = [];
  private timer: NodeJS.Timeout | null = null;
  private running = false;

  constructor(
    private emit: (kind: StreamKind, text: string) => Promise<void>,
    private intervalMs = 16,
    private maxCharsPerTick = 24,
  ) {}

  push(kind: StreamKind, text: string): void {
    if (!text) {
      return;
    }
    this.queue.push({ kind, text });
    this.schedule();
  }

  async drain(): Promise<void> {
    while (this.queue.length > 0) {
      await new Promise((resolve) => setTimeout(resolve, this.intervalMs));
    }
  }

  private schedule(): void {
    if (this.running || this.timer) {
      return;
    }
    this.timer = setTimeout(() => {
      void this.tick();
    }, this.intervalMs);
  }

  private async tick(): Promise<void> {
    this.timer = null;
    if (this.queue.length === 0) {
      this.running = false;
      return;
    }

    this.running = true;
    let remaining = this.maxCharsPerTick;

    while (remaining > 0 && this.queue.length > 0) {
      const item = this.queue[0];
      if (item.text.length <= remaining) {
        this.queue.shift();
        await this.emit(item.kind, item.text);
        remaining -= item.text.length;
      } else {
        const chunk = item.text.slice(0, remaining);
        item.text = item.text.slice(remaining);
        await this.emit(item.kind, chunk);
        remaining = 0;
      }
    }

    this.running = false;
    if (this.queue.length > 0) {
      this.schedule();
    }
  }
}
