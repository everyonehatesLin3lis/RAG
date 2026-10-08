"use client";

// Phase 8, redesigned in Phase 26: every chunk the model was given, shown under the answer (the first 3, then all).

import { useState } from "react";

export type Source = {
  movie: string;
  year: number | null;
  review_id: string | null;
  chunk_id: string;
  source: string;
  critic: string | null;
  publication: string | null;
  url: string | null;
  excerpt: string;
};

// Only http(s) links are rendered; anything else in the data is shown as plain text.
function safeUrl(url: string | null): string | null {
  return url && /^https?:\/\//i.test(url) ? url : null;
}

function attribution(source: Source): string {
  if (source.source === "tmdb_imdb") return "Movie profile (TMDB / IMDb)";
  return [source.critic ?? "Unknown critic", source.publication].filter(Boolean).join(", ");
}

const FIRST = 3;

export default function SourceList({ sources }: { sources: Source[] }) {
  const [all, setAll] = useState(false);
  if (sources.length === 0) return null;
  const shown = all ? sources : sources.slice(0, FIRST);

  return (
    <section className="mt-3" aria-label="Sources">
      <h3 className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">Sources</h3>
      <ol className="grid gap-2 sm:grid-cols-3">
        {shown.map((source, i) => {
          const url = safeUrl(source.url);
          return (
            <li key={source.chunk_id} className="flex min-w-0 flex-col rounded-lg border border-line p-2.5 text-xs">
              <div className="flex items-baseline gap-1.5">
                <span className="text-muted">{i + 1}</span>
                <span className="truncate text-sm font-medium" title={source.movie}>
                  {source.movie}
                  {source.year ? ` (${source.year})` : ""}
                </span>
              </div>
              <div className="truncate text-muted" title={attribution(source)}>
                {attribution(source)}
              </div>
              <p className={`mt-1 text-foreground/80 ${all ? "" : "line-clamp-3"}`}>{source.excerpt}</p>
              <div className="mt-auto flex gap-2 pt-1.5 text-muted">
                {url && (
                  <a href={url} target="_blank" rel="noopener noreferrer" className="underline hover:text-foreground">
                    Original review
                  </a>
                )}
                <span className="ml-auto" title={source.review_id ? `review ${source.review_id}` : undefined}>
                  chunk {source.chunk_id}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
      {sources.length > FIRST && (
        <button
          type="button"
          onClick={() => setAll(!all)}
          className="mt-1.5 text-xs text-muted underline hover:text-foreground"
        >
          {all ? "Show fewer" : `Show all ${sources.length} sources`}
        </button>
      )}
    </section>
  );
}
