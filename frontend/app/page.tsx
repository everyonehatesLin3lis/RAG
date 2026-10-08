"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

import SourceList, { type Source } from "./SourceList";
import RagPanel, { type RagDebug } from "./RagPanel";
import ToolCallList, { type ToolCall } from "./ToolCallList";
import UsagePanel, { type Usage } from "./UsagePanel";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
  sources?: Source[];
  debug?: RagDebug | null;
  toolCalls?: ToolCall[];
  usage?: Usage | null;
  // Phase 25: set while the answer is still streaming in
  streaming?: boolean;
  status?: string; // progress ("Searching reviews…") or a note under the answer ("Stopped")
};

type ErrorResponse = { error: { code: string; message: string } };

// Phase 25: the events of POST /api/chat/stream (backend/app/streaming.py).
type StreamEvent =
  | { type: "status"; stage: string; message: string }
  | { type: "token"; content: string }
  | { type: "tool_call"; data: ToolCall }
  | { type: "sources"; data: Source[] }
  | {
      type: "metadata";
      data: { answer: string; conversation_id: string; tool_calls: ToolCall[]; debug: RagDebug | null; usage: Usage | null };
    }
  | { type: "done" }
  | { type: "error"; error: { code: string; message: string } };

// Server-Sent Events over fetch: the browser's EventSource only does GET, and we POST a JSON body. The body is read
// as it arrives; events are separated by a blank line, and each has one "data: <json>" line.
async function streamChat(
  message: string,
  conversationId: string,
  onEvent: (event: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: conversationId }),
      signal,
    });
  } catch (err) {
    if (signal.aborted) throw err;
    throw new Error("Cannot reach the server. Is the backend running?");
  }
  if (!res.ok || !res.body) {
    // Invalid input is rejected before the stream starts, with the usual JSON error.
    const body = (await res.json().catch(() => null)) as ErrorResponse | null;
    throw new Error(body?.error?.message ?? `Request failed (${res.status}).`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      for (const line of block.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as StreamEvent);
      }
    }
  }
}

// Phase 11: the conversation id is remembered in this browser, so a reload continues the same conversation.
// Storage can be unavailable (private windows, blocked site data): then each page load starts a new one.
const CONVERSATION_KEY = "movie-copilot-conversation-id";

function readStoredId(): string | null {
  try {
    return localStorage.getItem(CONVERSATION_KEY);
  } catch {
    return null;
  }
}

function storeId(id: string) {
  try {
    localStorage.setItem(CONVERSATION_KEY, id);
  } catch {
    // not fatal: the conversation still works until the page is reloaded
  }
}

type StoredConversation = { messages: { role: Role; content: string }[] };

async function loadConversation(id: string): Promise<Message[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/conversations/${id}`);
    if (!res.ok) return []; // 404: nothing saved yet for this id
    const body = (await res.json()) as StoredConversation;
    return body.messages.map((m) => ({ role: m.role, content: m.content }));
  } catch {
    return [];
  }
}

export default function Home() {
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // On first load: reuse the stored conversation id (or create one) and show its earlier messages.
  useEffect(() => {
    let id = readStoredId();
    if (!id) {
      id = crypto.randomUUID();
      storeId(id);
    }
    const conversation = id;
    loadConversation(conversation).then((earlier) => {
      setConversationId(conversation);
      setMessages(earlier);
    });
  }, []);

  function startNewChat() {
    const id = crypto.randomUUID();
    storeId(id);
    setConversationId(id);
    setMessages([]);
    setError(null);
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading || !conversationId) return;

    // Phase 25: an empty assistant message that the stream fills in.
    setMessages((prev) => [
      ...prev,
      { role: "user", content: text },
      { role: "assistant", content: "", streaming: true, status: "Sending…", toolCalls: [] },
    ]);
    setInput("");
    setError(null);
    setLoading(true);

    const updateAnswer = (change: (m: Message) => Message) =>
      setMessages((prev) => [...prev.slice(0, -1), change(prev[prev.length - 1])]);

    const controller = new AbortController();
    abortRef.current = controller;
    let finished = false;
    try {
      await streamChat(
        text,
        conversationId,
        (event) => {
          switch (event.type) {
            case "status":
              updateAnswer((m) => ({ ...m, status: event.message }));
              break;
            case "token":
              updateAnswer((m) => ({ ...m, content: m.content + event.content }));
              break;
            case "tool_call":
              updateAnswer((m) => ({ ...m, toolCalls: [...(m.toolCalls ?? []), event.data] }));
              break;
            case "sources":
              updateAnswer((m) => ({ ...m, sources: event.data }));
              break;
            case "metadata": {
              const d = event.data;
              updateAnswer((m) => ({ ...m, content: d.answer, toolCalls: d.tool_calls, debug: d.debug, usage: d.usage }));
              break;
            }
            case "done":
              finished = true;
              updateAnswer((m) => ({ ...m, streaming: false, status: undefined }));
              break;
            case "error":
              throw new Error(event.error.message);
          }
        },
        controller.signal,
      );
      if (!finished) throw new Error("The answer was cut off. Please try again.");
    } catch (err) {
      if (controller.signal.aborted) {
        // Stopped by the user: the server stops too and stores nothing, so say so under what was shown.
        updateAnswer((m) => ({ ...m, streaming: false, status: "Stopped. This answer was not saved." }));
      } else {
        // Failed: drop the unfinished answer (the server did not store it) and show the error.
        setMessages((prev) => (prev[prev.length - 1]?.streaming ? prev.slice(0, -1) : prev));
        setError(err instanceof Error ? err.message : "Something went wrong.");
      }
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 py-6">
      <header className="mb-4 flex items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Movie Research Copilot</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">Ask anything about movies.</p>
        </div>
        <button
          type="button"
          onClick={startNewChat}
          disabled={loading}
          className="shrink-0 rounded-lg border border-zinc-300 px-3 py-1.5 text-sm disabled:opacity-50 dark:border-zinc-700"
        >
          New chat
        </button>
      </header>

      <section className="flex flex-1 flex-col gap-3 overflow-y-auto pb-4" aria-live="polite">
        {messages.length === 0 && !loading && (
          <p className="mt-8 text-center text-zinc-500">Try: &ldquo;Recommend me a slow-burning thriller.&rdquo;</p>
        )}

        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="max-w-[85%] self-end whitespace-pre-wrap rounded-2xl bg-blue-600 px-4 py-2 text-white">
              {msg.content}
            </div>
          ) : (
            <div key={i} className="flex flex-col">
              {/* react-markdown builds React elements and does not render raw HTML, so model output cannot inject markup. */}
              <div className="markdown max-w-[85%] self-start rounded-2xl bg-zinc-100 px-4 py-2 dark:bg-zinc-800">
                {msg.content ? <ReactMarkdown>{msg.content}</ReactMarkdown> : null}
                {msg.status && (
                  <p className={`text-sm text-zinc-500 ${msg.streaming ? "animate-pulse" : ""}`}>{msg.status}</p>
                )}
              </div>
              <ToolCallList calls={msg.toolCalls ?? []} via={msg.debug?.tool_backend} />
              <SourceList sources={msg.sources ?? []} />
              <RagPanel debug={msg.debug} />
              <UsagePanel usage={msg.usage} />
            </div>
          ),
        )}

        {error && (
          <div
            role="alert"
            className="self-stretch rounded-lg border border-red-300 bg-red-50 px-4 py-2 text-red-700 dark:border-red-800 dark:bg-red-950 dark:text-red-300"
          >
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </section>

      <form onSubmit={handleSubmit} className="sticky bottom-0 flex gap-2 bg-background pt-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a movie…"
          maxLength={2000}
          disabled={loading}
          aria-label="Message"
          className="min-w-0 flex-1 rounded-lg border border-zinc-300 bg-transparent px-3 py-2 outline-none focus:border-blue-600 disabled:opacity-60 dark:border-zinc-700"
        />
        {loading ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded-lg border border-zinc-300 px-4 py-2 font-medium dark:border-zinc-700"
          >
            Stop
          </button>
        ) : (
          <button
            type="submit"
            disabled={!input.trim() || !conversationId}
            className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
          >
            Send
          </button>
        )}
      </form>
    </main>
  );
}
