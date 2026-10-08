"""Tool calling (Phase 10): the model decides, our code executes.

The model never runs anything. It is given the tools' names, descriptions and argument schemas, and when it
thinks one would help it replies with a *request* ("call compare_movies with movie_a=Zodiac, movie_b=Prisoners")
instead of an answer. This loop:

1. sends the conversation to the model, bound to the tools;
2. if the reply contains tool requests, runs each one through `app/tools.py` (which validates the arguments
   first), and appends the result as a ToolMessage linked to the request by its call id;
3. sends everything back, so the model now sees the real numbers, and repeats;
4. stops when the model replies without a tool request: that reply is the answer.

At most MAX_TOOL_ROUNDS rounds; after that the model is called once more with tools switched off and must answer
with what it has. This caps both the wait and the cost if a model keeps asking for tools.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from app import llm

# Implements: specs/10.md (proposal 2)
MAX_TOOL_ROUNDS = 3


@dataclass
class ToolCallRecord:
    """One executed (or rejected) tool call, for the API response (Phase 10) and the UI (Phase 13)."""

    tool: str
    arguments: dict
    result: dict


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


# Implements: specs/10.md#AC-003, #AC-004
def execute(call: dict, tools_by_name: dict[str, BaseTool]) -> dict:
    """Run one requested call. Never raises: every problem becomes an error result the model can read."""
    tool = tools_by_name.get(call["name"])
    if tool is None:
        return _error("UNKNOWN_TOOL", f"There is no tool named {call['name']!r}. Available: {', '.join(tools_by_name)}.")
    try:
        # The LangChain wrapper validates the arguments against the tool's Pydantic schema and returns
        # INVALID_ARGUMENTS without running anything when they do not fit.
        return json.loads(tool.invoke(call.get("args") or {}))
    except Exception:  # an unexpected bug in a tool must not crash the chat
        return _error("TOOL_FAILED", f"The {call['name']} tool failed unexpectedly.")


# Implements: specs/10.md#AC-001, #AC-002
def run_with_tools(
    messages: list[BaseMessage], tools: list[BaseTool], on_event: Callable[[dict], None] | None = None
) -> tuple[str, list[ToolCallRecord]]:
    """Run the model with tools until it answers. Returns the answer text and every tool call made.

    Phase 25: with on_event, every model round is streamed and the loop reports what happens as events: each piece
    of answer text ("token"), each tool about to run ("status") and each tool result ("tool_call"). MiMo sometimes
    writes a sentence before calling a tool ("let me look that up"); it has already been shown, so it stays in the
    answer, as its own paragraph. Without on_event it works as before (only the final reply's text).
    """
    messages = list(messages)
    tools_by_name = {tool.name: tool for tool in tools}
    model = llm.tool_model(tools)
    records: list[ToolCallRecord] = []

    texts: list[str] = []  # everything the user saw: a model sometimes writes a sentence before a tool call

    def call_model(runnable) -> AIMessage:
        if on_event is None:
            return llm.invoke(runnable, messages)
        started_round = False

        def on_text(text: str) -> None:
            nonlocal started_round
            if not started_round and any(t.strip() for t in texts):
                on_event({"type": "token", "content": "\n\n"})  # new paragraph after an earlier round's text
            started_round = True
            on_event({"type": "token", "content": text})

        return llm.stream(runnable, messages, on_text)

    def shown() -> str:
        return "\n\n".join(t for t in texts if t.strip())

    for _ in range(MAX_TOOL_ROUNDS):
        reply = call_model(model)
        texts.append(reply.text)
        if not reply.tool_calls:
            return shown() if on_event else reply.text, records

        messages.append(reply)
        for call in reply.tool_calls:
            if on_event:
                on_event({"type": "status", "stage": "tool", "message": f"Calling {call['name']}"})
            result = execute(call, tools_by_name)
            record = ToolCallRecord(tool=call["name"], arguments=call.get("args") or {}, result=result)
            records.append(record)
            if on_event:
                on_event({"type": "tool_call", "data": asdict(record)})
            messages.append(ToolMessage(content=json.dumps(result, ensure_ascii=False), tool_call_id=call["id"]))
        if on_event:
            on_event({"type": "status", "stage": "answer", "message": "Writing the answer"})

    # Round limit reached: same tools in view (the history refers to them), but the model may not call any.
    final = call_model(llm.tool_model(tools, tool_choice="none"))
    texts.append(final.text)
    return shown() if on_event else final.text, records
