// Phase 13: shows each tool call the model made (tool, arguments, result), from `tool_calls` in /api/chat.

export type ToolCall = {
  tool: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
};

type Movie = { title: string; year: number | null; imdb_rating: number | null; genres: string[]; review_count?: number };
type ToolError = {
  code: string;
  message: string;
  argument?: string;
  candidates?: { title: string; year: number | null }[];
  suggestions?: { title: string; year: number | null }[];
};

const label = (m: { title: string; year: number | null }) => (m.year ? `${m.title} (${m.year})` : m.title);

function ErrorResult({ error }: { error: ToolError }) {
  const options = error.candidates ?? error.suggestions ?? [];
  return (
    <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
      <div className="font-medium">
        {error.code}
        {error.argument ? ` (${error.argument})` : ""}
      </div>
      <div>{error.message}</div>
      {options.length > 0 && (
        <div className="mt-1 text-xs">
          {error.candidates ? "Candidates: " : "Did you mean: "}
          {options.map(label).join(", ")}
        </div>
      )}
    </div>
  );
}

function CompareResult({ result }: { result: { movies: Movie[]; higher_imdb_rating: string | null } }) {
  return (
    <div>
      <table className="w-full text-left text-xs">
        <thead className="text-zinc-500">
          <tr>
            <th className="pr-2 font-normal">Movie</th>
            <th className="pr-2 font-normal">IMDb</th>
            <th className="pr-2 font-normal">Genres</th>
            <th className="font-normal">Reviews</th>
          </tr>
        </thead>
        <tbody>
          {result.movies.map((m) => (
            <tr key={label(m)}>
              <td className="pr-2 font-medium">{label(m)}</td>
              <td className="pr-2">{m.imdb_rating ?? "–"}</td>
              <td className="pr-2">{m.genres.join(", ")}</td>
              <td>{m.review_count ?? "–"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {result.higher_imdb_rating && (
        <div className="mt-1 text-xs">
          Higher IMDb rating: <span className="font-medium">{result.higher_imdb_rating}</span>
        </div>
      )}
    </div>
  );
}

function FilterResult({ result }: { result: { total_matches: number; returned: number; movies: Movie[] } }) {
  return (
    <div>
      <div className="text-xs text-zinc-500">
        {result.total_matches} match{result.total_matches === 1 ? "" : "es"}, showing the best {result.returned} by IMDb rating
      </div>
      <ol className="mt-1 list-decimal pl-5 text-xs">
        {result.movies.map((m) => (
          <li key={label(m)}>
            <span className="font-medium">{label(m)}</span> · {m.imdb_rating ?? "–"} · {m.genres.join(", ")}
          </li>
        ))}
      </ol>
    </div>
  );
}

function SummaryResult({
  result,
}: {
  result: {
    movie: { title: string; year: number | null };
    average_critic_rating: number | null;
    number_of_reviews: number;
    rating_distribution: Record<string, number>;
    unrated_reviews: number;
    note: string;
  };
}) {
  const counts = Object.values(result.rating_distribution);
  const max = Math.max(1, ...counts);
  return (
    <div className="text-xs">
      <div>
        <span className="font-medium">{label(result.movie)}</span>: average critic rating{" "}
        <span className="font-medium">{result.average_critic_rating ?? "–"}</span>/10 from {result.number_of_reviews} reviews
      </div>
      <div className="mt-1 flex flex-col gap-0.5">
        {Object.entries(result.rating_distribution).map(([bucket, count]) => (
          <div key={bucket} className="flex items-center gap-2">
            <span className="w-10 text-zinc-500">{bucket}</span>
            <span className="h-1.5 rounded bg-blue-600" style={{ width: `${(count / max) * 8}rem` }} />
            <span className="text-zinc-500">{count}</span>
          </div>
        ))}
        {result.unrated_reviews > 0 && <div className="text-zinc-500">no score: {result.unrated_reviews}</div>}
      </div>
      <div className="mt-1 text-zinc-500">{result.note}</div>
    </div>
  );
}

type CompareData = Parameters<typeof CompareResult>[0]["result"];
type FilterData = Parameters<typeof FilterResult>[0]["result"];
type SummaryData = Parameters<typeof SummaryResult>[0]["result"];

// Picks a readable view by tool name and result shape; anything unexpected is shown as JSON.
function Result({ call }: { call: ToolCall }) {
  const r = call.result;
  if (r.error) return <ErrorResult error={r.error as ToolError} />;
  if (call.tool === "compare_movies" && Array.isArray(r.movies)) return <CompareResult result={r as unknown as CompareData} />;
  if (call.tool === "filter_movies" && Array.isArray(r.movies)) return <FilterResult result={r as unknown as FilterData} />;
  if (call.tool === "rating_summary" && r.rating_distribution) return <SummaryResult result={r as unknown as SummaryData} />;
  return <pre className="overflow-x-auto text-xs">{JSON.stringify(call.result, null, 2)}</pre>;
}

function formatArguments(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  return entries.length ? entries.map(([k, v]) => `${k}: ${JSON.stringify(v)}`).join(", ") : "(none)";
}

export default function ToolCallList({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) return null;
  const seen = new Set<string>();

  return (
    <details className="mt-2 max-w-[85%] self-start text-sm">
      <summary className="cursor-pointer select-none text-zinc-600 dark:text-zinc-400">
        Tool calls ({calls.length})
      </summary>
      <ol className="mt-2 flex flex-col gap-2">
        {calls.map((call, i) => {
          const key = `${call.tool}:${JSON.stringify(call.arguments)}`;
          const repeated = seen.has(key);
          seen.add(key);
          return (
            <li key={i} className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-700">
              <div className="text-xs text-zinc-500">Tool used{repeated ? " · same call repeated by the model" : ""}</div>
              <div className="font-mono font-medium">{call.tool}</div>
              <div className="mt-1 text-xs text-zinc-500">Arguments</div>
              <div className="font-mono text-xs">{formatArguments(call.arguments)}</div>
              <div className="mt-1 text-xs text-zinc-500">Result</div>
              <Result call={call} />
              <details className="mt-1 text-xs">
                <summary className="cursor-pointer select-none text-zinc-500">raw JSON (what the model received)</summary>
                <pre className="mt-1 overflow-x-auto rounded bg-zinc-100 p-2 dark:bg-zinc-900">
                  {JSON.stringify({ arguments: call.arguments, result: call.result }, null, 2)}
                </pre>
              </details>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
