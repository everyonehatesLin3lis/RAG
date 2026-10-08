// Phase 26: one look for every collapsible debug panel under an answer (RAG process, tool calls, token usage).
// A closed panel shows its title and a one-line summary, so the essentials are readable without opening it.

import type { ReactNode } from "react";

export default function Panel({ title, meta, children }: { title: string; meta?: ReactNode; children: ReactNode }) {
  return (
    <details className="group border-t border-line first:border-t-0">
      <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-sm hover:bg-surface [&::-webkit-details-marker]:hidden">
        <span aria-hidden className="text-muted transition-transform group-open:rotate-90">
          ▸
        </span>
        <span className="shrink-0 whitespace-nowrap font-medium">{title}</span>
        {meta && <span className="ml-auto truncate pl-3 text-xs text-muted">{meta}</span>}
      </summary>
      <div className="px-3 pb-3 pt-1 text-sm">{children}</div>
    </details>
  );
}
