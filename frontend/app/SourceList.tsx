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

export default function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;

  return (
    <details className="mt-2 max-w-[85%] self-start text-sm">
      <summary className="cursor-pointer select-none text-zinc-600 dark:text-zinc-400">
        Sources ({sources.length})
      </summary>
      <ol className="mt-2 flex flex-col gap-2">
        {sources.map((source) => {
          const url = safeUrl(source.url);
          return (
            <li key={source.chunk_id} className="rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-700">
              <div className="font-medium">
                {source.movie}
                {source.year ? ` (${source.year})` : ""}
                <span className="font-normal text-zinc-600 dark:text-zinc-400"> · {attribution(source)}</span>
              </div>
              <p className="mt-1 text-zinc-700 dark:text-zinc-300">{source.excerpt}</p>
              <div className="mt-1 flex gap-3 text-xs text-zinc-500">
                {url && (
                  <a href={url} target="_blank" rel="noopener noreferrer" className="underline">
                    Original review
                  </a>
                )}
                <span>
                  chunk {source.chunk_id}
                  {source.review_id ? ` · review ${source.review_id}` : ""}
                </span>
              </div>
            </li>
          );
        })}
      </ol>
    </details>
  );
}
