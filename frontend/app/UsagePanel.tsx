// Phase 14: tokens and cost for one answer, from `usage` in /api/chat.

export type ModelUsage = {
  model: string;
  calls: number;
  input_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  cached_input_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  cost_source: string;
};

export type Usage = {
  model: string;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  by_model: ModelUsage[];
};

const SOURCE_LABEL: Record<string, string> = {
  reported: "reported by OpenRouter",
  estimated: "estimated",
  "not reported": "not reported",
};

// Costs per answer are fractions of a cent, so show enough digits to be meaningful.
function dollars(value: number | null): string {
  if (value === null) return "–";
  if (value > 0 && value < 0.000001) return "< $0.000001";
  return `$${value < 0.01 ? value.toFixed(6) : value.toFixed(4)}`;
}

const n = (value: number) => value.toLocaleString("en-US");

export default function UsagePanel({ usage }: { usage: Usage | null | undefined }) {
  if (!usage || usage.by_model.length === 0) return null;

  return (
    <details className="mt-2 max-w-[85%] self-start text-sm">
      <summary className="cursor-pointer select-none text-zinc-600 dark:text-zinc-400">
        Tokens &amp; cost: {n(usage.total_tokens)} tokens · {dollars(usage.estimated_cost_usd)}
      </summary>
      <div className="mt-2 rounded-lg border border-zinc-200 px-3 py-2 dark:border-zinc-700">
        <table className="w-full text-left text-xs">
          <thead className="text-zinc-500">
            <tr>
              <th className="pr-2 font-normal">Model</th>
              <th className="pr-2 font-normal">Calls</th>
              <th className="pr-2 font-normal">Input</th>
              <th className="pr-2 font-normal">Output</th>
              <th className="font-normal">Cost</th>
            </tr>
          </thead>
          <tbody>
            {usage.by_model.map((m) => (
              <tr key={m.model} className="align-top">
                <td className="pr-2 font-mono">{m.model}</td>
                <td className="pr-2">{m.calls}</td>
                <td className="pr-2">
                  {n(m.input_tokens)}
                  {m.cached_input_tokens > 0 && <div className="text-zinc-500">{n(m.cached_input_tokens)} cached</div>}
                </td>
                <td className="pr-2">
                  {n(m.output_tokens)}
                  {m.reasoning_tokens > 0 && <div className="text-zinc-500">{n(m.reasoning_tokens)} reasoning</div>}
                </td>
                <td>
                  {dollars(m.cost_usd)}
                  <div className="text-zinc-500">{SOURCE_LABEL[m.cost_source] ?? m.cost_source}</div>
                </td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="font-medium">
              <td className="pr-2 pt-1">Total</td>
              <td />
              <td className="pr-2 pt-1">{n(usage.input_tokens)}</td>
              <td className="pr-2 pt-1">{n(usage.output_tokens)}</td>
              <td className="pt-1">{dollars(usage.estimated_cost_usd)}</td>
            </tr>
          </tfoot>
        </table>
        <p className="mt-2 text-xs text-zinc-500">
          Most of the input is context, not your question: system rules, recent history and the retrieved sources,
          sent again on every model round. Cached input is served from the provider&apos;s prompt cache at a discount.
        </p>
      </div>
    </details>
  );
}
