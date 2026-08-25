import { describe, expect, it } from "vitest";
import { LruCache, TtlLruCache } from "./lru.js";

describe("LruCache", () => {
  it("evicts the least recently used entry beyond the limit", () => {
    const cache = new LruCache<string, number>(2);
    cache.set("a", 1);
    cache.set("b", 2);
    cache.get("a"); // promote a
    cache.set("c", 3); // evicts b
    expect(cache.size).toBe(2);
    expect(cache.get("a")).toBe(1);
    expect(cache.get("b")).toBeUndefined();
    expect(cache.get("c")).toBe(3);
  });

  it("stores nothing when the limit is zero or negative", () => {
    for (const limit of [0, -1]) {
      const cache = new LruCache<string, number>(limit);
      cache.set("a", 1);
      expect(cache.size).toBe(0);
      expect(cache.get("a")).toBeUndefined();
    }
  });
});

describe("TtlLruCache", () => {
  it("treats expired entries as misses", async () => {
    const cache = new TtlLruCache<string, number>(4, 10);
    cache.set("k", 7);
    expect(cache.get("k")).toBe(7);
    await new Promise((resolve) => setTimeout(resolve, 15));
    expect(cache.get("k")).toBeUndefined();
  });

  it("stores nothing when the limit is zero", () => {
    const cache = new TtlLruCache<string, number>(0, 100);
    cache.set("k", 7);
    expect(cache.get("k")).toBeUndefined();
  });
});
