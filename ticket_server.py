"""
A tiny Jira/ClickUp-like MCP server that we write ourselves.

It talks over stdio, so the agent (the MCP *client*) launches it as a subprocess.
Tools it exposes:  create_ticket, list_tickets
Tickets are stored in a local SQLite file (tickets.db).

IMPORTANT: never print() in a stdio MCP server - stdout is the protocol channel.
"""
import json
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

DB_PATH = Path(__file__).parent / "tickets.db"
mcp = FastMCP("tickets")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            priority TEXT NOT NULL DEFAULT 'medium',
            status TEXT NOT NULL DEFAULT 'open',
            idempotency_key TEXT UNIQUE,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    return conn


def _row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["tags"] = json.loads(d["tags"])
    return d


@mcp.tool()
def create_ticket(
    title: str,
    description: str,
    tags: list[str] | None = None,
    priority: str = "medium",
    idempotency_key: str = "",
) -> str:
    """Create a ticket. priority must be low, medium or high.
    If idempotency_key was already used, the existing ticket is returned
    instead of creating a duplicate (safe to retry after a crash)."""
    if not title.strip():
        raise ValueError("title cannot be empty")
    if priority not in {"low", "medium", "high"}:
        raise ValueError("priority must be low, medium or high")

    with _conn() as conn:
        if idempotency_key:
            existing = conn.execute(
                "SELECT * FROM tickets WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing:
                return json.dumps({**_row(existing), "note": "already existed - not duplicated"})

        cur = conn.execute(
            "INSERT INTO tickets (title, description, tags, priority, idempotency_key) VALUES (?, ?, ?, ?, ?)",
            (title, description, json.dumps(tags or []), priority, idempotency_key or None),
        )
        row = conn.execute("SELECT * FROM tickets WHERE id = ?", (cur.lastrowid,)).fetchone()
    return json.dumps(_row(row))


@mcp.tool()
def list_tickets(limit: int = 10) -> str:
    """List the most recent tickets."""
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM tickets ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return json.dumps([_row(r) for r in rows])


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
