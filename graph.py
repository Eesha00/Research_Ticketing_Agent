"""
The LangGraph workflow:

  START -> load_memory -> plan -> search -> query_db -> draft --(has sources?)--> human_review -> create_ticket -> END
                                    ^                      |            no (and attempts left)      |
                                    +----------------------+                                  reject -> END
"""
import hashlib
import re
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

MAX_SEARCH_ATTEMPTS = 3


# ---------------------------------------------------------------- state
class State(TypedDict, total=False):
    request: str            # what the user asked
    preferences: list[str]  # long-term memories loaded for this user
    queries: list[str]      # web search queries
    db_question: str        # question for the internal database
    sources: list[dict]     # web results (title, url, content)
    search_attempts: int    # how many times we've searched (stops infinite loops)
    db_findings: str        # summary from the Postgres MCP tool
    draft: dict             # {title, description, tags, priority}
    status: str             # created / rejected / failed
    ticket_result: str


# ---------------------------------------------------------------- structured outputs
class Prefs(BaseModel):
    preferences: list[str] = Field(
        default_factory=list,
        description="Standing, long-term preferences the user stated (e.g. 'always tag tickets as backend'). Empty if none.",
    )


class Plan(BaseModel):
    search_queries: list[str] = Field(description="2-3 short web search queries")
    db_question: str = Field(description="One plain-English question for our internal database")


class Queries(BaseModel):
    queries: list[str]


class Draft(BaseModel):
    title: str
    description: str = Field(description="Ticket body. Must end with a 'Sources' section listing full URLs.")
    tags: list[str]
    priority: Literal["low", "medium", "high"]


# ---------------------------------------------------------------- helpers
def _text(content) -> str:
    """Tool/LLM output can be a string or a list of content blocks - normalise to text."""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        parts.append(p.get("text", "") if isinstance(p, dict) else str(p))
    return "".join(parts)


async def run_tool_loop(llm, tools, messages, max_steps: int = 5) -> str:
    """A tiny hand-written ReAct loop: think -> call tool -> read result -> repeat.
    Tool errors are returned to the model as text so it can recover."""
    bound = llm.bind_tools(tools)
    by_name = {t.name: t for t in tools}
    for _ in range(max_steps):
        ai = await bound.ainvoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return _text(ai.content)
        for call in ai.tool_calls:
            try:
                result = _text(await by_name[call["name"]].ainvoke(call["args"]))
            except Exception as e:  # error handling in tools
                result = f"Tool error: {e}. Fix your input or try another approach."
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return "Stopped after max steps without a final answer."


# ---------------------------------------------------------------- graph builder
def build_graph(llm, search_tool, pg_tools, create_ticket_tool) -> StateGraph:

    # 1) long-term memory: WRITE new preferences, then READ all of them
    async def load_memory(state: State, config: RunnableConfig, *, store: BaseStore):
        ns = ("memories", config["configurable"]["user_id"])

        found = await llm.with_structured_output(Prefs).ainvoke([
            SystemMessage(content="Extract only lasting preferences the user states about how tickets "
                                  "should be made (e.g. 'always...', 'never...', 'I prefer...'). "
                                  "Ignore one-off details of this task. Return an empty list if none."),
            HumanMessage(content=state["request"]),
        ])
        for pref in found.preferences:
            key = hashlib.md5(pref.lower().encode()).hexdigest()[:12]  # same text -> same key -> no duplicates
            await store.aput(ns, key, {"text": pref})

        items = await store.asearch(ns, limit=20)
        return {"preferences": [i.value["text"] for i in items], "search_attempts": 0, "sources": []}

    # 2) plan
    async def plan(state: State):
        p = await llm.with_structured_output(Plan).ainvoke([
            SystemMessage(content="You plan research tasks. Give 2-3 short web search queries and ONE plain-English "
                                  "question for our internal Postgres database (tables: services, incidents)."),
            HumanMessage(content=state["request"]),
        ])
        return {"queries": p.search_queries, "db_question": p.db_question}

    # 3) web search (loops back here if the draft has no sources)
    async def search(state: State):
        attempt = state.get("search_attempts", 0)
        queries = state["queries"]
        if attempt > 0:  # retry: ask for different, more specific queries
            r = await llm.with_structured_output(Queries).ainvoke([
                SystemMessage(content="Our earlier searches did not give citable sources. Write 3 different, more specific queries."),
                HumanMessage(content=f"Task: {state['request']}\nPrevious queries: {queries}"),
            ])
            queries = r.queries

        sources = list(state.get("sources", []))
        seen = {s["url"] for s in sources}
        for q in queries:
            try:
                res = await search_tool.ainvoke({"query": q})
                if isinstance(res, dict):
                    for item in res.get("results", []):
                        if item.get("url") and item["url"] not in seen:
                            seen.add(item["url"])
                            sources.append({"title": item.get("title", ""), "url": item["url"],
                                            "content": item.get("content", "")})
            except Exception as e:
                print(f"   (search failed for '{q}': {e})")
        return {"queries": queries, "sources": sources, "search_attempts": attempt + 1}

    # 4) internal database through the Postgres MCP server
    async def query_db(state: State):
        if not pg_tools:
            print("   DB: no Postgres tools connected")
            return {"db_findings": "(no database connected)"}
        print("   DB tools available:", [t.name for t in pg_tools])
        messages = [
            SystemMessage(content="You can query our internal Postgres database with the tools provided. "
                                  "The tables are 'services' and 'incidents'. "
                                  "Use read-only SELECT queries, for example: SELECT * FROM incidents. "
                                  "Then summarise what you found in a few bullets."),
            HumanMessage(content=state["db_question"]),
        ]
        try:
            findings = await run_tool_loop(llm, pg_tools, messages)
        except Exception as e:
            findings = f"(database lookup failed: {e})"
        print("   DB findings:", findings[:400])
        return {"db_findings": findings}

    # 5) draft the ticket
    async def draft(state: State):
        src = "\n".join(f"- {s['title']} | {s['url']}\n  {s['content'][:300]}" for s in state["sources"][:8]) or "(no web sources found)"
        prefs = "\n".join(f"- {p}" for p in state.get("preferences", [])) or "(none)"
        d = await llm.with_structured_output(Draft).ainvoke([
            SystemMessage(content="Write a clear engineering ticket. Put full source URLs in a 'Sources' section at the end. "
                                  "Apply the user's standing preferences (tags, priority, etc.). "
                                  "If the internal DB findings contain relevant incidents, mention them in the ticket as internal evidence."),
            HumanMessage(content=f"Request: {state['request']}\n\nWeb findings:\n{src}\n\n"
                                 f"Internal DB findings:\n{state['db_findings']}\n\nStanding preferences:\n{prefs}"),
        ])
        return {"draft": d.model_dump()}

    # conditional edge: no sources in the draft -> go back to search (up to MAX_SEARCH_ATTEMPTS)
    def route_after_draft(state: State) -> Literal["search", "human_review"]:
        has_sources = bool(re.search(r"https?://", state["draft"]["description"]))
        if not has_sources and state.get("search_attempts", 0) < MAX_SEARCH_ATTEMPTS:
            print("   ↩ draft has no sources, looping back to search")
            return "search"
        return "human_review"

    # 6) human-in-the-loop. interrupt() pauses the graph; the state is saved by the checkpointer.
    #    NOTE: when resumed, this node re-runs from the top, so keep code before interrupt() side-effect free.
    def human_review(state: State) -> Command[Literal["create_ticket"]]:
        decision = interrupt({"question": "Approve this ticket?", "draft": state["draft"]})
        action = decision.get("action")
        if action == "reject":
            return Command(goto=END, update={"status": "rejected"})
        if action == "edit":
            return Command(goto="create_ticket", update={"draft": decision["draft"]})
        return Command(goto="create_ticket")  # approve

    # 7) create the ticket through OUR MCP server
    async def create_ticket(state: State, config: RunnableConfig):
        d = state["draft"]
        try:
            result = await create_ticket_tool.ainvoke({
                "title": d["title"],
                "description": d["description"],
                "tags": d["tags"],
                "priority": d["priority"],
                "idempotency_key": config["configurable"]["thread_id"],  # resume-after-crash can't duplicate
            })
            return {"ticket_result": _text(result), "status": "created"}
        except Exception as e:
            return {"ticket_result": f"Ticket creation failed: {e}", "status": "failed"}

    g = StateGraph(State)
    g.add_node("load_memory", load_memory)
    g.add_node("plan", plan)
    g.add_node("search", search)
    g.add_node("query_db", query_db)
    g.add_node("draft", draft)
    g.add_node("human_review", human_review)
    g.add_node("create_ticket", create_ticket)

    g.add_edge(START, "load_memory")
    g.add_edge("load_memory", "plan")
    g.add_edge("plan", "search")
    g.add_edge("search", "query_db")
    g.add_edge("query_db", "draft")
    g.add_conditional_edges("draft", route_after_draft, {"search": "search", "human_review": "human_review"})
    g.add_edge("create_ticket", END)
    return g
