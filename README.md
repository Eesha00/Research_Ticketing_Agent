# Research & Ticketing Agent (LangGraph + MCP)

```
START → load_memory → plan → search → query_db → draft ──(has sources?)──► human_review → create_ticket → END
                               ▲                    │          no                │
                               └────────────────────┘                       reject → END
```

| File | What it does |
|---|---|
| `graph.py` | The LangGraph: state, nodes, conditional loop, HITL interrupt, memory read/write |
| `main.py` | CLI: connects to both MCP servers, sets up checkpointing + memory, runs/resumes the graph |
| `ticket_server.py` | **Our own MCP server** (mini Jira): `create_ticket`, `list_tickets` |
| `seed.sql` / `docker-compose.yml` | Fake company Postgres DB (`services`, `incidents`) |
| `requirements.txt` / `.env.example` | Dependencies and keys |

## Setup

1. **Install:** Python 3.10+, Node.js (for `npx`), Docker (or a local Postgres).
2. **Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   cp .env.example .env              # Windows: copy .env.example .env  → then add your keys
   ```
3. **Database**
   ```bash
   docker compose up -d
   ```
   No Docker? Install Postgres, then: `createdb company` and `psql company -f seed.sql`
   (and update `DATABASE_URL` in `.env`).
4. **Keys:** an LLM key (OpenAI by default) and a free Tavily key in `.env`.

## Run

```bash
python main.py run "Research why our auth service has high latency and file a ticket. Always tag my tickets as backend." --user eesha --thread demo1
```
The agent plans, searches, queries Postgres, drafts, then **pauses** and asks: approve / edit / reject.

## Demo each requirement (what to show your trainer)

| Requirement | How to demo |
|---|---|
| **Two MCP servers** | Postgres server (existing, via npx) + `ticket_server.py` (ours). Both are set up in `main.py → open_app`. |
| **Crash & resume** | `python main.py run "..." --thread crash1 --crash-after search` (process hard-kills), then `python main.py resume --thread crash1`. It continues at `query_db`; search is not repeated. |
| **HITL** | At the prompt, try **a**pprove, **e**dit (change title/tags/priority) and **r**eject on different runs. Reject ends with `status: rejected` and no ticket. |
| **Long-term memory** | Run 1 with "Always tag my tickets as backend" on `--user eesha`. Run 2 on a **new thread** without saying it: the draft still has the backend tag. See it with `python main.py memories --user eesha`. |
| **Short-term memory** | Thread state: `python main.py history --thread demo1` shows every saved step. |
| **Loop back to search** | If a draft has no URLs, you'll see `↩ draft has no sources, looping back to search` (max 3 attempts so it can't loop forever). |
| **Time-travel** | `python main.py history --thread demo1`, copy a `checkpoint=` id, then `python main.py replay --thread demo1 --checkpoint <id>`. |

## Things worth knowing

- **Why the thread ID matters:** same thread → continue that run; new thread → fresh run. It's also used as the ticket's `idempotency_key`, so resuming after a crash can never create a duplicate ticket.
- **Why `interrupt()` code reruns:** on resume, the review node restarts from its top, so nothing with side effects goes before `interrupt()`.
- **Short vs long-term memory:** short-term = checkpointed state per thread (`checkpoints.db`). Long-term = separate store per user (`memory.db`), keyed by `("memories", user_id)`.
- **When we write memory:** only when the user states a *standing* preference (LLM extracts it; same text = same key, so no duplicates). We read at the start of every run.
- **Error handling:** tool failures are returned to the model as text (`run_tool_loop`) and search/DB failures don't crash the graph.
- **Swapping to a Postgres checkpointer:** replace `AsyncSqliteSaver` with `AsyncPostgresSaver` (package `langgraph-checkpoint-postgres`). On Windows this needs extra event-loop care, which is why SQLite is the default here.

## Troubleshooting

- **`npx` not found / Postgres MCP fails:** install Node.js. The app still runs without DB access (it prints a warning).
- **The Postgres MCP server package is unavailable:** swap the `args` in `main.py` for any other Postgres MCP server.
- **Auth errors:** check `.env` keys; to use another provider, change `MODEL` (e.g. `anthropic:claude-sonnet-4-5`) and install `langchain-anthropic`.
- **Start clean:** delete `checkpoints.db`, `memory.db`, `tickets.db`.
