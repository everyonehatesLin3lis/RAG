"use client";

import { useEffect, useState } from "react";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

type BackendStatus = "checking" | "ok" | "unreachable";

export default function Home() {
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((body: { status: string }) => setStatus(body.status === "ok" ? "ok" : "unreachable"))
      .catch(() => setStatus("unreachable"));
  }, []);

  return (
    <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-4 px-4 py-16">
      <h1 className="text-3xl font-semibold">Movie Research Copilot</h1>
      <p className="text-zinc-600 dark:text-zinc-400">
        Ask questions about movies, answered from retrieved reviews with sources. Chat arrives in Phase 1.
      </p>
      <p>
        Backend:{" "}
        <span
          className={
            status === "ok" ? "text-green-600" : status === "unreachable" ? "text-red-600" : "text-zinc-500"
          }
        >
          {status}
        </span>
      </p>
    </main>
  );
}
