"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type Role = "user" | "assistant";

type Message = {
  role: Role;
  content: string;
};

type ChatResponse = { answer: string };
type ErrorResponse = { error: { code: string; message: string } };

async function sendChat(message: string, conversationId: string): Promise<string> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, conversation_id: conversationId }),
    });
  } catch {
    throw new Error("Cannot reach the server. Is the backend running?");
  }

  const body = (await res.json().catch(() => null)) as ChatResponse | ErrorResponse | null;
  if (!res.ok || body === null || "error" in body) {
    const detail = body && "error" in body ? body.error.message : `Request failed (${res.status}).`;
    throw new Error(detail);
  }
  return body.answer;
}

export default function Home() {
  const [conversationId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setError(null);
    setLoading(true);

    try {
      const answer = await sendChat(text, conversationId);
      setMessages((prev) => [...prev, { role: "assistant", content: answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col px-4 py-6">
      <header className="mb-4">
        <h1 className="text-2xl font-semibold">Movie Research Copilot</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">Ask anything about movies.</p>
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
            // react-markdown builds React elements and does not render raw HTML, so model output cannot inject markup.
            <div key={i} className="markdown max-w-[85%] self-start rounded-2xl bg-zinc-100 px-4 py-2 dark:bg-zinc-800">
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            </div>
          ),
        )}

        {loading && (
          <div className="self-start rounded-2xl bg-zinc-100 px-4 py-2 text-zinc-500 dark:bg-zinc-800">
            Thinking…
          </div>
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
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="rounded-lg bg-blue-600 px-4 py-2 font-medium text-white disabled:opacity-50"
        >
          Send
        </button>
      </form>
    </main>
  );
}
