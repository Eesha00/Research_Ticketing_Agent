"""
CLI for the Research & Ticketing Agent.

  python main.py run "Research why auth latency is high and file a ticket" --user eesha --thread demo1
  python main.py run "..." --thread demo1 --crash-after search     # simulate a crash
  python main.py resume --thread demo1                             # continue from the last checkpoint
  python main.py history --thread demo1                            # list checkpoints (time-travel)
  python main.py replay --thread demo1 --checkpoint <id>           # re-run from an earlier checkpoint
  python main.py memories --user eesha                             # show long-term memory
"""
import argparse
import asyncio
import os
import shutil
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from graph import build_graph

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

try:
    from langgraph.store.sqlite.aio import AsyncSqliteStore  # persistent long-term store
except ImportError:  # older package version: fall back (memory then lasts only while the process runs)
    AsyncSqliteStore = None
    from langgraph.store.memory import InMemoryStore


# ------------------------------------------------------------ setup
async def open_app(stack: AsyncExitStack):
    llm = init_chat_model(os.getenv("MODEL", "openai:gpt-4o-mini"), temperature=0)

    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/company")
    client = MultiServerMCPClient({
        # EXISTING MCP server (Postgres), launched with npx
        "postgres": {
            "command": shutil.which("npx") or "npx",
            "args": ["-y", "@modelcontextprotocol/server-postgres", db_url],
            "transport": "stdio",
        },
        # OUR OWN MCP server (tickets)
        "tickets": {
            "command": sys.executable,
            "args": [str(HERE / "ticket_server.py")],
            "transport": "stdio",
        },
    })

    try:
        pg_tools = await client.get_tools(server_name="postgres")
    except Exception as e:
        print(f"⚠ Could not connect to the Postgres MCP server ({e}). Continuing without DB access.")
        pg_tools = []
    ticket_tools = {t.name: t for t in await client.get_tools(server_name="tickets")}

    checkpointer = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(str(HERE / "checkpoints.db")))
    if AsyncSqliteStore:
        store = await stack.enter_async_context(AsyncSqliteStore.from_conn_string(str(HERE / "memory.db")))
        await store.setup()
    else:
        print("⚠ AsyncSqliteStore not available - long-term memory will not survive restarts. Upgrade langgraph-checkpoint-sqlite.")
        store = InMemoryStore()

    search_tool = TavilySearch(max_results=4)
    graph = build_graph(llm, search_tool, pg_tools, ticket_tools["create_ticket"]).compile(
        checkpointer=checkpointer, store=store
    )
    return graph, store


# ------------------------------------------------------------ human review prompt
def ask_human(payload: dict) -> dict:
    d = payload["draft"]
    print("\n================ DRAFT TICKET ================")
    print(f"Title    : {d['title']}\nPriority : {d['priority']}\nTags     : {', '.join(d['tags'])}\n")
    print(d["description"])
    print("==============================================")
    choice = input("[a]pprove / [e]dit / [r]eject ? ").strip().lower()
    if choice.startswith("r"):
        return {"action": "reject"}
    if choice.startswith("e"):
        title = input(f"Title [{d['title']}]: ").strip() or d["title"]
        tags_in = input(f"Tags, comma separated [{', '.join(d['tags'])}]: ").strip()
        tags = [t.strip() for t in tags_in.split(",") if t.strip()] if tags_in else d["tags"]
        priority = input(f"Priority low/medium/high [{d['priority']}]: ").strip() or d["priority"]
        extra = input("Add a line to the description (Enter to skip): ").strip()
        description = d["description"] + (f"\n\n{extra}" if extra else "")
        return {"action": "edit", "draft": {**d, "title": title, "tags": tags, "priority": priority, "description": description}}
    return {"action": "approve"}


# ------------------------------------------------------------ run loop
async def drive(graph, inp, config, crash_after=None):
    """Stream the graph. When it pauses at an interrupt, ask the human and resume."""
    while True:
        interrupt_payload = None
        async for chunk in graph.astream(inp, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    interrupt_payload = update[0].value
                else:
                    print(f"✓ {node}")
                    if crash_after == node:
                        print(f"💥 Simulated crash after '{node}'. Run: python main.py resume --thread {config['configurable']['thread_id']}")
                        os._exit(1)  # hard kill, no cleanup - like a real crash
        if interrupt_payload is None:
            break
        inp = Command(resume=ask_human(interrupt_payload))

    values = (await graph.aget_state(config)).values
    print(f"\nStatus: {values.get('status')}")
    if values.get("ticket_result"):
        print(f"Ticket: {values['ticket_result']}")


async def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "resume", "history", "replay", "memories"):
        p = sub.add_parser(name)
        if name == "run":
            p.add_argument("request")
            p.add_argument("--crash-after", default=None, help="node name, e.g. search")
        if name != "memories":
            p.add_argument("--thread", default=None)
        p.add_argument("--user", default="default-user")
        if name == "replay":
            p.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    async with AsyncExitStack() as stack:
        graph, store = await open_app(stack)

        if args.cmd == "memories":
            items = await store.asearch(("memories", args.user), limit=50)
            print(f"Long-term memories for '{args.user}':")
            for i in items:
                print(" -", i.value["text"])
            return

        thread = args.thread or uuid.uuid4().hex[:8]
        config = {"configurable": {"thread_id": thread, "user_id": args.user}}

        if args.cmd == "run":
            print(f"Thread: {thread}   User: {args.user}")
            await drive(graph, {"request": args.request}, config, args.crash_after)

        elif args.cmd == "resume":
            state = await graph.aget_state(config)
            if not state.next:
                print("Nothing to resume: this thread is finished (or doesn't exist).")
                return
            print(f"Resuming thread {thread} - next step: {state.next}")
            await drive(graph, None, config)  # None = "continue from the saved checkpoint"

        elif args.cmd == "history":
            async for snap in graph.aget_state_history(config):
                cid = snap.config["configurable"]["checkpoint_id"]
                print(f"checkpoint={cid}  step={snap.metadata.get('step')}  next={snap.next}")

        elif args.cmd == "replay":
            config["configurable"]["checkpoint_id"] = args.checkpoint
            await drive(graph, None, config)


if __name__ == "__main__":
    asyncio.run(main())
