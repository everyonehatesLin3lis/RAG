// Phase 12: shows what retrieval did for one answer. Data comes from the `debug` object of /api/chat.

export type VectorResult = {
  rank: number;
  chunk_id: string;
  movie: string;
  year: number | null;
  doc_type: string | null;
  critic: string | null;
  distance: number;
  similarity: number;
  excerpt: string;
};

export type KeywordResult = {
  rank: number;
  chunk_id: string;
  movie: string;
  year: number | null;
  doc_type: string | null;
  critic: string | null;
  score: number;
  excerpt: string;
};

export type RagDebug = {
  original_query: string;
  translated_query: string;
  vector_results: VectorResult[];
  selected_chunks: string[];
  keywords: string[];
  filters: Record<string, unknown>;
  translation_origin: string;
  history_messages: number;
  timings_ms: Record<string, number>;
  keyword_results: KeywordResult[];
};

const ORIGIN_LABEL: Record<string, string> = {
  model: "rewritten by query translation",
  fallback: "translation failed, so the original question was searched",
  disabled: "query translation is switched off",
};

const TIMING_LABEL: Record<string, string> = {
  translation: "Query translation",
  embedding_and_search: "Embedding + vector search",
  keyword_search: "Keyword search",
  generation: "Answer (incl. tool calls)",
  total: "Total",
};

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(1)} s`;
}

function formatFilters(filters: Record<string, unknown>): string {
  return Object.entries(filters)
    .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(", ") : String(value)}`)
    .join(" · ");
}

export default function RagPanel({ debug }: { debug: RagDebug | null | undefined }) {
  if (!debug) return null;
  const selected = new Set(debug.selected_chunks);
  const vectorIds = new Set(debug.vector_results.map((r) => r.chunk_id));
  const filters = formatFilters(debug.filters ?? {});

  return (
    <details className="mt-2 max-w-[85%] self-start text-sm">
      <summary className="cursor-pointer select-none text-zinc-600 dark:text-zinc-400">RAG process</summary>
      <div className="mt-2 flex flex-col gap-3 rounded-lg border border-zinc-200 px-3 py-3 dark:border-zinc-700">
        <section>
          <h3 className="font-medium">1. Your question</h3>
          <p className="text-zinc-700 dark:text-zinc-300">{debug.original_query}</p>
          {debug.history_messages > 0 && (
            <p className="text-xs text-zinc-500">
              {debug.history_messages} earlier message{debug.history_messages === 1 ? "" : "s"} from this
              conversation were included.
            </p>
          )}
        </section>

        <section>
          <h3 className="font-medium">2. Search query</h3>
          <p className="text-zinc-700 dark:text-zinc-300">{debug.translated_query}</p>
          <p className="text-xs text-zinc-500">{ORIGIN_LABEL[debug.translation_origin] ?? debug.translation_origin}</p>
          {debug.keywords.length > 0 && (
            <p className="mt-1 flex flex-wrap gap-1">
              {debug.keywords.map((k) => (
                <span key={k} className="rounded bg-zinc-100 px-1.5 py-0.5 text-xs dark:bg-zinc-800">
                  {k}
                </span>
              ))}
            </p>
          )}
          {filters && <p className="mt-1 text-xs text-zinc-500">Filters (not applied yet): {filters}</p>}
        </section>

        <section>
          <h3 className="font-medium">3. Vector search: {debug.vector_results.length} closest chunks</h3>
          <p className="text-xs text-zinc-500">
            Similarity = 1 − cosine distance between the search query&apos;s embedding and each chunk&apos;s.
          </p>
          <ol className="mt-1 flex flex-col gap-1">
            {debug.vector_results.map((r) => (
              <li key={r.chunk_id} className="flex items-start gap-2">
                <span className="w-5 shrink-0 text-right text-xs text-zinc-500">{r.rank}</span>
                <span className="w-24 shrink-0" title={`distance ${r.distance}`}>
                  <span className="block h-1.5 rounded bg-zinc-200 dark:bg-zinc-700">
                    <span
                      className="block h-1.5 rounded bg-blue-600"
                      style={{ width: `${Math.max(0, Math.min(1, r.similarity)) * 100}%` }}
                    />
                  </span>
                  <span className="text-xs text-zinc-500">{r.similarity.toFixed(3)}</span>
                </span>
                <span className="min-w-0">
                  <span className="font-medium">
                    {r.movie}
                    {r.year ? ` (${r.year})` : ""}
                  </span>
                  <span className="text-zinc-500">
                    {" "}
                    · {r.doc_type === "profile" ? "movie profile" : r.critic ?? "review"} · chunk {r.chunk_id}
                    {selected.has(r.chunk_id) ? " · sent to the model" : ""}
                  </span>
                  <span className="block truncate text-xs text-zinc-600 dark:text-zinc-400">{r.excerpt}</span>
                </span>
              </li>
            ))}
          </ol>
        </section>

        <section>
          <h3 className="font-medium">4. Keyword search: {debug.keyword_results.length} full-text matches</h3>
          <p className="text-xs text-zinc-500">
            PostgreSQL full-text search for the keywords above (each as a phrase, word stems matched). Shown for
            comparison: the answer still uses the vector results until hybrid search combines both.
          </p>
          {debug.keyword_results.length === 0 ? (
            <p className="text-xs text-zinc-500">No keywords, or no chunk contains them.</p>
          ) : (
            <ol className="mt-1 flex flex-col gap-1">
              {debug.keyword_results.map((r) => (
                <li key={r.chunk_id} className="flex items-start gap-2">
                  <span className="w-5 shrink-0 text-right text-xs text-zinc-500">{r.rank}</span>
                  <span className="w-12 shrink-0 text-xs text-zinc-500" title="ts_rank_cd, normalised 0..1">
                    {r.score.toFixed(3)}
                  </span>
                  <span className="min-w-0">
                    <span className="font-medium">
                      {r.movie}
                      {r.year ? ` (${r.year})` : ""}
                    </span>
                    <span className="text-zinc-500">
                      {" "}
                      · {r.doc_type === "profile" ? "movie profile" : r.critic ?? "review"} · chunk {r.chunk_id}
                      {vectorIds.has(r.chunk_id) ? " · also found by vector search" : ""}
                    </span>
                    <span className="block truncate text-xs text-zinc-600 dark:text-zinc-400">{r.excerpt}</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </section>

        <section>
          <h3 className="font-medium">5. Time</h3>
          <ul className="text-xs text-zinc-600 dark:text-zinc-400">
            {Object.entries(debug.timings_ms).map(([step, ms]) => (
              <li key={step}>
                {TIMING_LABEL[step] ?? step}: {seconds(ms)}
              </li>
            ))}
          </ul>
        </section>
      </div>
    </details>
  );
}
