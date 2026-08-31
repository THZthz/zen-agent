import { describe, expect, it } from 'vitest';
import { piMaxRetries } from '../chat-completions.js';

describe('piMaxRetries', () => {
  it('maps total attempts (including the first) onto Pi retry count', () => {
    expect(piMaxRetries(4)).toBe(3);
    expect(piMaxRetries(1)).toBe(0);
  });

  it('defaults to 4 total attempts and never goes negative', () => {
    expect(piMaxRetries(undefined)).toBe(3);
    expect(piMaxRetries(0)).toBe(0);
  });
});
