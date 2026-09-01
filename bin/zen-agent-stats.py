#!/usr/bin/env python3
"""Read-only queries over Zen Agent's SQLite session statistics.

The agent keeps every session in one SQLite file (README, "SQLite
persistence"): the sessions table stores aggregate usage plus the JSON
arrays turn_stats and cache_diagnostics, llm_log one row per LLM
request/response. This script prints those numbers as plain text.

Usage:
  zen-agent-stats.py list              one line per session (default)
  zen-agent-stats.py show SESSION_ID    usage, per-turn and cache details
  zen-agent-stats.py totals             sums across all sessions

Database location: --db flag, else ZEN_AGENT_DB_FILE, else
$XDG_DATA_HOME/zen-agent/zen-agent.db. The file is only ever opened
read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

USAGE_KEYS = (
    "turns",
    "steps",
    "inputTokens",
    "outputTokens",
    "cacheReadTokens",
    "cacheMissTokens",
    "reasoningTokens",
    "cost",
    "llmMs",
    "thinkingMs",
    "answeringMs",
    "toolMs",
)


def default_db_path() -> Path:
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(data_home) / "zen-agent" / "zen-agent.db"


def resolve_db(flag_value: str | None) -> Path:
    if flag_value:
        return Path(flag_value)
    env_value = os.environ.get("ZEN_AGENT_DB_FILE", "").strip()
    if env_value:
        return Path(env_value)
    return default_db_path()


def open_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        sys.exit(f"error: database not found: {path} (use --db or ZEN_AGENT_DB_FILE)")
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def fmt_int(value) -> str:
    return "-" if value is None else str(value)


def fmt_cost(value) -> str:
    return "-" if value is None else f"{value:.6f}"


def fmt_hit_rate(value) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def fmt_ts(value) -> str:
    return "-" if not value else value.replace("T", " ")[:19]


def fmt_value(value) -> str:
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def print_kv(pairs) -> None:
    if not pairs:
        return
    width = max(len(key) for key, _ in pairs)
    for key, value in pairs:
        print(f"  {key.ljust(width)}  {value}")


def print_table(headers, rows, aligns) -> None:
    """Print pre-formatted string rows; aligns is one l/r per column."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    head = "  ".join(
        h.ljust(widths[i]) if aligns[i] == "l" else h.rjust(widths[i])
        for i, h in enumerate(headers)
    )
    print(head)
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(
            cell.ljust(widths[i]) if aligns[i] == "l" else cell.rjust(widths[i])
            for i, cell in enumerate(row)
        ))


def cmd_list(db: sqlite3.Connection) -> None:
    rows = db.execute(
        "SELECT session_id, title, cwd, updated_at, usage"
        " FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    if not rows:
        print("no sessions")
        return
    table = []
    for session_id, title, cwd, updated_at, usage_json in rows:
        usage = json.loads(usage_json)
        name = title or os.path.basename(cwd.rstrip("/")) or cwd
        table.append([
            session_id,
            name,
            fmt_int(usage.get("turns")),
            fmt_int(usage.get("inputTokens")),
            fmt_int(usage.get("outputTokens")),
            fmt_int(usage.get("cacheReadTokens")),
            fmt_int(usage.get("cacheMissTokens")),
            fmt_cost(usage.get("cost")),
            fmt_ts(updated_at),
        ])
    print_table(
        ["SESSION", "TITLE", "TURNS", "INPUT", "OUTPUT", "CACHE_RD", "CACHE_MISS", "COST", "UPDATED"],
        table,
        "llrrrrrrl",
    )


def cmd_show(db: sqlite3.Connection, session_id: str) -> None:
    row = db.execute(
        "SELECT session_id, title, cwd, created_at, updated_at, config, usage,"
        " turn_stats, cache_diagnostics FROM sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if row is None:
        sys.exit(f"error: session not found: {session_id}")
    session_id, title, cwd, created_at, updated_at, config_json, usage_json, \
        turn_stats_json, cache_json = row
    config = json.loads(config_json)
    usage = json.loads(usage_json)

    print(f"session   {session_id}")
    print(f"title     {title or '-'}")
    print(f"cwd       {cwd}")
    print(f"created   {created_at}")
    print(f"updated   {updated_at}")

    print("\nconfig")
    print_kv(list(config.items()))

    print("\nusage")
    print_kv([(key, fmt_value(value)) for key, value in usage.items()])

    print("\nturn stats")
    turns = json.loads(turn_stats_json)
    if not turns:
        print("  (none)")
    else:
        print_table(
            ["TURN", "STEPS", "INPUT", "OUTPUT", "CACHE_RD", "CACHE_MISS", "REASONING", "COST", "LLM_MS", "THINK_MS"],
            [[
                str(i + 1),
                fmt_int(t.get("steps")),
                fmt_int(t.get("inputTokens")),
                fmt_int(t.get("outputTokens")),
                fmt_int(t.get("cacheReadTokens")),
                fmt_int(t.get("cacheMissTokens")),
                fmt_int(t.get("reasoningTokens")),
                fmt_cost(t.get("cost")),
                fmt_int(t.get("llmMs")),
                fmt_int(t.get("thinkingMs")),
            ] for i, t in enumerate(turns)],
            "r" * 10,
        )

    print("\ncache diagnostics")
    diags = json.loads(cache_json)
    if not diags:
        print("  (none)")
    else:
        print_table(
            ["TURN", "MODEL", "INPUT", "CACHED", "MISS", "HIT_RATE", "SAVED", "REASON", "DETAIL"],
            [[
                fmt_int(d.get("turn")),
                str(d.get("model", "-")),
                fmt_int(d.get("inputTokens")),
                fmt_int(d.get("cachedTokens")),
                fmt_int(d.get("cacheMissTokens")),
                fmt_hit_rate(d.get("cacheHitRate")),
                fmt_cost(d.get("savedCost")),
                str(d.get("missReason", "-")),
                truncate(str(d.get("missReasonDetail", "")), 60),
            ] for d in diags],
            "rlrrrrrll",
        )

    print("\nllm log")
    entries = db.execute(
        "SELECT seq, created_at, entry FROM llm_log"
        " WHERE session_id = ? ORDER BY seq",
        (session_id,),
    ).fetchall()
    if not entries:
        print("  (none)")
    else:
        print_table(
            ["SEQ", "TIME", "TYPE", "INPUT", "OUTPUT", "CACHE_RD", "CACHE_MISS", "LLM_MS", "FINISH"],
            [[
                str(seq),
                fmt_ts(created_at),
                json.loads(entry).get("type", "-"),
                fmt_int((json.loads(entry).get("usage") or {}).get("inputTokens")),
                fmt_int((json.loads(entry).get("usage") or {}).get("outputTokens")),
                fmt_int((json.loads(entry).get("usage") or {}).get("cacheReadTokens")),
                fmt_int((json.loads(entry).get("usage") or {}).get("cacheMissTokens")),
                fmt_int((json.loads(entry).get("usage") or {}).get("llmMs")),
                json.loads(entry).get("finishReason", "-"),
            ] for seq, created_at, entry in entries],
            "rlrrrrrrl",
        )


def cmd_totals(db: sqlite3.Connection) -> None:
    rows = db.execute("SELECT usage FROM sessions").fetchall()
    if not rows:
        print("no sessions")
        return
    totals: dict[str, int | float] = {}
    for (usage_json,) in rows:
        for key, value in json.loads(usage_json).items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0) + value
    print_kv(
        [("sessions", len(rows))]
        + [(key, fmt_value(totals.get(key, 0))) for key in USAGE_KEYS]
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        prog="zen-agent-stats",
        description="Query Zen Agent session statistics (read-only).",
    )
    parser.add_argument(
        "--db",
        help="database file (default: ZEN_AGENT_DB_FILE, else ~/.local/share/zen-agent/zen-agent.db)",
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("list", help="one line per session (default)")
    show = sub.add_parser("show", help="usage, per-turn and cache details for one session")
    show.add_argument("session_id")
    sub.add_parser("totals", help="sums across all sessions")
    args = parser.parse_args(argv)
    args.command = args.command or "list"
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    db = open_db(resolve_db(args.db))
    try:
        if args.command == "show":
            cmd_show(db, args.session_id)
        elif args.command == "totals":
            cmd_totals(db)
        else:
            cmd_list(db)
    except sqlite3.DatabaseError as error:
        sys.exit(f"error: not a Zen Agent database: {error}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
