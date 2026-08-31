export type StreamKind = 'thought' | 'message';

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
  private discarded = false;
  /** First emit failure, surfaced to drain() so a broken pipe ends the turn. */
  private failure: { error: unknown } | null = null;

  constructor(
    private emit: (kind: StreamKind, text: string) => Promise<void>,
    private intervalMs = 16,
    private maxCharsPerTick = 24,
  ) {}

  push(kind: StreamKind, text: string): void {
    if (!text || this.discarded) {
      return;
    }
    this.queue.push({ kind, text });
    this.schedule();
  }

  /**
   * Resolves once every queued chunk has been emitted; rejects with the
   * emit error when the consumer is gone (e.g. the client connection
   * closed mid-stream), so callers stop instead of waiting on a queue that
   * can never drain.
   */
  async drain(): Promise<void> {
    for (;;) {
      if (this.discarded) {
        return;
      }
      // Checked unconditionally: on failure the queue is dropped, so a
      // queue-only condition could mask the error.
      if (this.failure) {
        throw this.failure.error;
      }
      if (this.queue.length === 0) {
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, this.intervalMs));
    }
  }

  /** Stop future emissions and drop queued output after an aborted/failed step. */
  discard(): void {
    this.discarded = true;
    this.queue = [];
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private schedule(): void {
    if (this.discarded || this.running || this.timer || this.failure) {
      return;
    }
    this.timer = setTimeout(() => {
      void this.tick();
    }, this.intervalMs);
  }

  private async tick(): Promise<void> {
    this.timer = null;
    if (this.discarded || this.queue.length === 0) {
      this.running = false;
      return;
    }

    this.running = true;
    try {
      let remaining = this.maxCharsPerTick;

      while (!this.discarded && remaining > 0 && this.queue.length > 0) {
        const item = this.queue[0]!;
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
    } catch (error) {
      // The emit callback failed (e.g. cx.notify on a closed connection).
      // Drop the queued output and record the failure instead of letting the
      // rejection escape `void this.tick()` — an unhandled rejection would
      // crash the process, and leaving `running` stuck true would hang
      // future schedule() calls and drain() forever.
      this.queue = [];
      this.failure ??= { error };
      this.running = false;
      return;
    }

    this.running = false;
    if (!this.discarded && this.queue.length > 0) {
      this.schedule();
    }
  }
}
