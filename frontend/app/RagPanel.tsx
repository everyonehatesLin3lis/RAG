import Panel from "./Panel";

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

export type FusedResult = {
  rank: number;
  chunk_id: string;
  movie: string;
  year: number | null;
  doc_type: string | null;
  found_by: string[];
  vector_rank: number | null;
  keyword_rank: number | null;
  fused_score: number;
  selected: boolean;
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
  strategy: string;
  tool_backend?: string; // Phase 24: "mcp" or "local"
  warnings?: string[]; // Phase 27: fallbacks used while answering
  per_film?: string[]; // Phase 29: films retrieved separately, each with an equal share of the chunks
  fused_results: FusedResult[];
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
  const t = debug.timings_ms;
  const retrievalMs = (t.translation ?? 0) + (t.embedding_and_search ?? 0) + (t.keyword_search ?? 0);
  const meta = `${debug.strategy} search · ${debug.selected_chunks.length} chunks to the model · ${seconds(retrievalMs)}`;

  return (
    <Panel title="RAG process" meta={meta}>
      <div className="flex flex-col gap-3">
        <section>
          <h3 className="font-medium">1. Your question</h3>
          <p className="text-foreground/85">{debug.original_query}</p>
          {debug.history_messages > 0 && (
            <p className="text-xs text-muted">
              {debug.history_messages} earlier message{debug.history_messages === 1 ? "" : "s"} from this
              conversation were included.
            </p>
          )}
        </section>

        <section>
          <h3 className="font-medium">2. Search query</h3>
          <p className="text-foreground/85">{debug.translated_query}</p>
          <p className="text-xs text-muted">{ORIGIN_LABEL[debug.translation_origin] ?? debug.translation_origin}</p>
          {debug.keywords.length > 0 && (
            <p className="mt-1 flex flex-wrap gap-1">
              {debug.keywords.map((k) => (
                <span key={k} className="rounded bg-surface px-1.5 py-0.5 text-xs">
                  {k}
                </span>
              ))}
            </p>
          )}
          {filters && <p className="mt-1 text-xs text-muted">Filters (not applied yet): {filters}</p>}
        </section>

        <section>
          <h3 className="font-medium">3. Vector search: {debug.vector_results.length} closest chunks</h3>
          <p className="text-xs text-muted">
            Similarity = 1 − cosine distance between the search query&apos;s embedding and each chunk&apos;s.
          </p>
          {(debug.per_film ?? []).length > 1 && (
            <p className="mt-1 text-xs text-muted">
              The question names {debug.per_film!.length} films, so each was searched separately and gets an equal
              share of the {debug.selected_chunks.length} chunks sent to the model: {debug.per_film!.join(", ")}.
            </p>
          )}
          <ol className="mt-1 flex flex-col gap-1">
            {debug.vector_results.map((r) => (
              <li key={r.chunk_id} className="flex items-start gap-2">
                <span className="w-5 shrink-0 text-right text-xs text-muted">{r.rank}</span>
                <span className="w-24 shrink-0" title={`distance ${r.distance}`}>
                  <span className="block h-1.5 rounded bg-line">
                    <span
                      className="block h-1.5 rounded bg-accent"
                      style={{ width: `${Math.max(0, Math.min(1, r.similarity)) * 100}%` }}
                    />
                  </span>
                  <span className="text-xs text-muted">{r.similarity.toFixed(3)}</span>
                </span>
                <span className="min-w-0">
                  <span className="font-medium">
                    {r.movie}
                    {r.year ? ` (${r.year})` : ""}
                  </span>
                  <span className="text-muted">
                    {" "}
                    · {r.doc_type === "profile" ? "movie profile" : r.critic ?? "review"} · chunk {r.chunk_id}
                    {selected.has(r.chunk_id) ? " · sent to the model" : ""}
                  </span>
                  <span className="block truncate text-xs text-muted">{r.excerpt}</span>
                </span>
              </li>
            ))}
          </ol>
        </section>

        <section>
          <h3 className="font-medium">4. Keyword search: {debug.keyword_results.length} full-text matches</h3>
          <p className="text-xs text-muted">
            PostgreSQL full-text search for the keywords above (each as a phrase, word stems matched).
            {debug.strategy === "hybrid"
              ? " Merged with the vector results in step 5."
              : " Not used: the retrieval strategy is vector-only."}
          </p>
          {debug.keyword_results.length === 0 ? (
            <p className="text-xs text-muted">No keywords, or no chunk contains them.</p>
          ) : (
            <ol className="mt-1 flex flex-col gap-1">
              {debug.keyword_results.map((r) => (
                <li key={r.chunk_id} className="flex items-start gap-2">
                  <span className="w-5 shrink-0 text-right text-xs text-muted">{r.rank}</span>
                  <span className="w-12 shrink-0 text-xs text-muted" title="ts_rank_cd, normalised 0..1">
                    {r.score.toFixed(3)}
                  </span>
                  <span className="min-w-0">
                    <span className="font-medium">
                      {r.movie}
                      {r.year ? ` (${r.year})` : ""}
                    </span>
                    <span className="text-muted">
                      {" "}
                      · {r.doc_type === "profile" ? "movie profile" : r.critic ?? "review"} · chunk {r.chunk_id}
                      {vectorIds.has(r.chunk_id) ? " · also found by vector search" : ""}
                    </span>
                    <span className="block truncate text-xs text-muted">{r.excerpt}</span>
                  </span>
                </li>
              ))}
            </ol>
          )}
        </section>

        {debug.strategy === "hybrid" && debug.fused_results.length > 0 && (
          <section>
            <h3 className="font-medium">5. Hybrid fusion: what the model received</h3>
            <p className="text-xs text-muted">
              Reciprocal Rank Fusion: each chunk scores 1 / (60 + rank) in every list it appears in, added up. Found high
              by both searches beats found by one. The top {debug.selected_chunks.length} were sent to the model.
            </p>
            <ol className="mt-1 flex flex-col gap-1">
              {debug.fused_results.map((r) => (
                <li key={r.chunk_id} className={`flex items-start gap-2 ${r.selected ? "" : "opacity-50"}`}>
                  <span className="w-5 shrink-0 text-right text-xs text-muted">{r.rank}</span>
                  <span className="w-14 shrink-0 text-xs text-muted">{r.fused_score.toFixed(4)}</span>
                  <span className="min-w-0">
                    <span className="font-medium">
                      {r.movie}
                      {r.year ? ` (${r.year})` : ""}
                    </span>
                    <span className="text-muted">
                      {" "}
                      · {r.vector_rank ? `vector #${r.vector_rank}` : ""}
                      {r.vector_rank && r.keyword_rank ? " + " : ""}
                      {r.keyword_rank ? `keyword #${r.keyword_rank}` : ""}
                      {r.selected ? " · sent to the model" : " · not sent"}
                    </span>
                  </span>
                </li>
              ))}
            </ol>
          </section>
        )}

        <section>
          <h3 className="font-medium">{debug.strategy === "hybrid" ? "6" : "5"}. Time</h3>
          <ul className="text-xs text-muted">
            {Object.entries(debug.timings_ms).map(([step, ms]) => (
              <li key={step}>
                {TIMING_LABEL[step] ?? step}: {seconds(ms)}
              </li>
            ))}
          </ul>
        </section>
      </div>
    </Panel>
  );
}
