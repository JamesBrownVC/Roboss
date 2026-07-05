import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Boxes, Clock3, Film, LineChart as LineChartIcon, Percent, RefreshCw, ShieldCheck } from "lucide-react";
import PageHeader from "../components/PageHeader.jsx";
import { getStats } from "../lib/api.js";

// Neon palette (mirrors the tailwind `neon` tokens).
const NEON_GREEN = "#3cf28a";
const NEON_AMBER = "#ffb020";
const NEON_CYAN = "#2fe8ea";
const NEON_MAGENTA = "#f13df5";
const NEON_RED = "#ff3b6b";
const NEON_VIOLET = "#8b5cf6";
const STAGE_COLORS = [NEON_CYAN, NEON_AMBER, NEON_MAGENTA, NEON_VIOLET, NEON_GREEN];
const GREEN = NEON_GREEN;
const RED = NEON_RED;
const AXIS_TICK = "#a996c9";
const GRID_STROKE = "rgba(139, 92, 246, 0.16)";

const MOCK_CAMERAS = ["front_view", "rear_view", "side_view", "high_angle_inspection"];
const MOCK_PER_DAY = [4, 6, 5, 7, 6, 8];

// Deterministic sample data shown only until the first real run is recorded.
function buildMockRuns() {
  const dayMs = 86400000;
  const runs = [];
  let n = 0;
  for (let index = 0; index < MOCK_PER_DAY.length; index += 1) {
    const dayOffset = MOCK_PER_DAY.length - 1 - index;
    const dayStart = Date.now() - dayOffset * dayMs;
    for (let i = 0; i < MOCK_PER_DAY[index]; i += 1) {
      n += 1;
      const generationFailed = n % 13 === 0;
      const reviewFailed = !generationFailed && n % 6 === 0;
      const completed = !generationFailed && !reviewFailed;
      const generationSeconds = 42 + (n % 5) * 6;
      const reviewSeconds = 8 + (n % 4) * 2;
      const labelingSeconds = completed ? 12 + (n % 3) * 3 : null;
      const renderingSeconds = completed ? 5 + (n % 2) * 2 : null;
      const totalSeconds =
        generationSeconds + reviewSeconds + (labelingSeconds || 0) + (renderingSeconds || 0);
      runs.push({
        id: `sample-${n}`,
        createdAt: new Date(dayStart + i * 900000).toISOString(),
        status: completed ? "completed" : "failed",
        reviewStatus: generationFailed ? "pending" : reviewFailed ? "failed" : "passed",
        labelStatus: completed ? "completed" : generationFailed ? "failed" : "pending",
        cameraVariant: MOCK_CAMERAS[n % MOCK_CAMERAS.length],
        aspectRatio: n % 4 === 0 ? "9:16" : "16:9",
        zoneCount: completed ? 5 + (n % 7) : 0,
        totalSeconds,
        generationSeconds,
        reviewSeconds,
        labelingSeconds,
        renderingSeconds,
        mock: true,
      });
    }
  }
  return runs;
}

// Grafana-style dark tooltip.
const CHART_TOOLTIP_STYLE = {
  backgroundColor: "#0f0a1a",
  border: "1px solid rgba(204, 204, 220, 0.2)",
  borderRadius: 2,
  color: "#ccccdc",
  fontSize: 12,
  boxShadow: "0 8px 24px rgba(0, 0, 0, 0.5)",
};

const CHART_CURSOR = { stroke: "rgba(204, 204, 220, 0.35)", strokeWidth: 1 };

function renderLegend({ payload }) {
  return (
    <div className="mt-1 flex flex-wrap items-center justify-start gap-x-4 gap-y-1 pl-8 text-xs text-sage-300">
      {(payload || []).map((entry) => (
        <span key={entry.value} className="flex items-center gap-1.5">
          <span className="h-[3px] w-4 rounded-full" style={{ backgroundColor: entry.color }} />
          {entry.value}
        </span>
      ))}
    </div>
  );
}

function isSuccess(run) {
  return run.status === "completed";
}

function formatSeconds(value) {
  if (value == null || Number.isNaN(Number(value))) {
    return "-";
  }
  const seconds = Number(value);
  if (seconds < 90) {
    return `${Math.round(seconds)}s`;
  }
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function average(values) {
  const numbers = values.filter((value) => value != null && !Number.isNaN(Number(value)));
  if (!numbers.length) {
    return null;
  }
  return numbers.reduce((total, value) => total + Number(value), 0) / numbers.length;
}

function StatCard({ icon: Icon, label, value, hint }) {
  return (
    <div className="rounded-lg border border-surface-700 bg-surface-900 p-5">
      <div className="flex items-center gap-2 text-sage-400">
        <Icon size={16} aria-hidden="true" />
        <span className="text-xs font-medium uppercase tracking-wide">{label}</span>
      </div>
      <div className="mt-2 text-3xl font-semibold tracking-tight text-white">{value}</div>
      {hint ? <div className="mt-1 text-xs text-sage-500">{hint}</div> : null}
    </div>
  );
}

// Grafana-like panel: small centered title, tight body padding.
function ChartCard({ title, children }) {
  return (
    <div className="rounded-lg border border-surface-700 bg-surface-900 px-3 pb-3 pt-2">
      <h2 className="mb-2 text-center text-[13px] font-medium text-sage-200">{title}</h2>
      {children}
    </div>
  );
}

export default function Stats() {
  const [runs, setRuns] = useState(() => buildMockRuns());
  const [source, setSource] = useState("local");
  const [isDemo, setIsDemo] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  async function load() {
    setIsLoading(true);
    const result = await getStats();
    if (result.runs.length) {
      setRuns(result.runs);
      setIsDemo(false);
    } else {
      setRuns(buildMockRuns());
      setIsDemo(true);
    }
    setSource(result.source);
    setIsLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  const kpis = useMemo(() => {
    const total = runs.length;
    const succeeded = runs.filter(isSuccess).length;
    const datasets = runs.filter((run) => Number(run.zoneCount) > 0 || run.labelStatus === "completed").length;
    const successRate = total ? Math.round((succeeded / total) * 100) : null;
    const avgLatency = average(runs.map((run) => run.totalSeconds));
    const totalZones = runs.reduce((sum, run) => sum + (Number(run.zoneCount) || 0), 0);

    const validationReviewed = runs.filter(
      (run) => run.reviewStatus === "passed" || run.reviewStatus === "failed",
    ).length;
    const validationPassed = runs.filter((run) => run.reviewStatus === "passed").length;
    const validationFailed = runs.filter((run) => run.reviewStatus === "failed").length;
    const validationRate = validationReviewed
      ? Math.round((validationPassed / validationReviewed) * 100)
      : null;

    return {
      total,
      succeeded,
      datasets,
      successRate,
      avgLatency,
      totalZones,
      validationRate,
      validationFailed,
    };
  }, [runs]);

  const timeline = useMemo(() => {
    const byDay = new Map();
    for (const run of runs) {
      const day = String(run.createdAt || "").slice(0, 10) || "unknown";
      const entry = byDay.get(day) || { day, videos: 0, datasets: 0 };
      entry.videos += 1;
      if (Number(run.zoneCount) > 0 || run.labelStatus === "completed") {
        entry.datasets += 1;
      }
      byDay.set(day, entry);
    }
    return [...byDay.values()].sort((a, b) => a.day.localeCompare(b.day));
  }, [runs]);

  const latencyByStage = useMemo(() => {
    const stages = [
      { key: "generationSeconds", label: "Generation" },
      { key: "reviewSeconds", label: "Review" },
      { key: "labelingSeconds", label: "Labeling" },
      { key: "renderingSeconds", label: "Rendering" },
    ];
    const rows = stages
      .map(({ key, label }) => ({ stage: label, seconds: average(runs.map((run) => run[key])) }))
      .filter((row) => row.seconds != null)
      .map((row) => ({ ...row, seconds: Math.round(row.seconds * 10) / 10 }));
    if (rows.length) {
      return rows;
    }
    const avgTotal = average(runs.map((run) => run.totalSeconds));
    return avgTotal != null ? [{ stage: "End-to-end", seconds: Math.round(avgTotal * 10) / 10 }] : [];
  }, [runs]);

  const outcome = useMemo(() => {
    const succeeded = runs.filter(isSuccess).length;
    const failed = runs.length - succeeded;
    return [
      { name: "Succeeded", value: succeeded, color: GREEN },
      { name: "Failed", value: failed, color: RED },
    ].filter((slice) => slice.value > 0);
  }, [runs]);

  const recentRuns = useMemo(
    () =>
      [...runs]
        .sort((a, b) => String(b.createdAt || "").localeCompare(String(a.createdAt || "")))
        .slice(0, 8),
    [runs],
  );

  return (
    <>
      <PageHeader
        title="Analytics"
        subtitle={
          isDemo
            ? "Sample data for preview. Real metrics replace it after your first generation."
            : `Pipeline activity overview. Data source: ${source === "api" ? "backend" : "this browser"}.`
        }
      >
        <div className="flex items-center gap-3">
          {isDemo ? (
            <span className="inline-flex items-center gap-1.5 rounded-full border border-surface-600 bg-surface-850 px-3 py-1.5 text-xs font-medium text-sage-300">
              <span className="h-1.5 w-1.5 rounded-full bg-sage-400" />
              Sample data
            </span>
          ) : null}
          <button
            type="button"
            onClick={load}
            className="inline-flex h-9 items-center gap-2 rounded-md border border-surface-600 bg-surface-950 px-4 text-sm font-medium text-sage-200 transition hover:border-sage-500 hover:text-white"
          >
            <RefreshCw size={15} className={isLoading ? "animate-spin" : ""} aria-hidden="true" />
            Refresh
          </button>
        </div>
      </PageHeader>

      {runs.length === 0 ? (
        <div className="flex min-h-72 flex-col items-center justify-center gap-3 rounded-lg border border-surface-700 bg-surface-900 p-10 text-center">
          <LineChartIcon className="text-sage-400" size={28} aria-hidden="true" />
          <p className="text-lg font-medium text-white">No runs yet</p>
          <p className="max-w-sm text-sm text-sage-400">
            Generate your first videos from the Studio page and the dashboard will start
            filling up with latency, dataset and success metrics.
          </p>
        </div>
      ) : (
        <div className="flex flex-col gap-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            <StatCard icon={Film} label="Videos generated" value={kpis.total} hint={`${kpis.succeeded} succeeded`} />
            <StatCard
              icon={ShieldCheck}
              label="Validation pass rate"
              value={kpis.validationRate != null ? `${kpis.validationRate}%` : "-"}
              hint={`${kpis.validationFailed} not validated`}
            />
            <StatCard
              icon={Boxes}
              label="Datasets generated"
              value={kpis.datasets}
              hint={`${kpis.totalZones} zones detected`}
            />
            <StatCard
              icon={Percent}
              label="Success rate"
              value={kpis.successRate != null ? `${kpis.successRate}%` : "-"}
            />
            <StatCard
              icon={Clock3}
              label="Avg latency"
              value={formatSeconds(kpis.avgLatency)}
              hint="prompt to delivery"
            />
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <div className="xl:col-span-2">
              <ChartCard title="Generations over time">
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={timeline} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
                    <defs>
                      <linearGradient id="videosFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={NEON_MAGENTA} stopOpacity={0.25} />
                        <stop offset="100%" stopColor={NEON_MAGENTA} stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="datasetsFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={NEON_CYAN} stopOpacity={0.25} />
                        <stop offset="100%" stopColor={NEON_CYAN} stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="day" tick={{ fill: AXIS_TICK, fontSize: 11 }} tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} tick={{ fill: AXIS_TICK, fontSize: 11 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={CHART_TOOLTIP_STYLE} cursor={CHART_CURSOR} />
                    <Legend content={renderLegend} />
                    <Area
                      type="monotone"
                      dataKey="videos"
                      name="Videos"
                      stroke={NEON_MAGENTA}
                      strokeWidth={1.5}
                      fill="url(#videosFill)"
                      dot={{ r: 2, fill: NEON_MAGENTA, strokeWidth: 0 }}
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                    <Area
                      type="monotone"
                      dataKey="datasets"
                      name="Datasets"
                      stroke={NEON_CYAN}
                      strokeWidth={1.5}
                      fill="url(#datasetsFill)"
                      dot={{ r: 2, fill: NEON_CYAN, strokeWidth: 0 }}
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </ChartCard>
            </div>

            <ChartCard title="Outcome">
              <ResponsiveContainer width="100%" height={260}>
                <PieChart>
                  <Pie
                    data={outcome}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={64}
                    outerRadius={94}
                    paddingAngle={1}
                    stroke="#07040d"
                    strokeWidth={2}
                  >
                    {outcome.map((slice) => (
                      <Cell key={slice.name} fill={slice.color} />
                    ))}
                  </Pie>
                  <Tooltip contentStyle={CHART_TOOLTIP_STYLE} />
                </PieChart>
              </ResponsiveContainer>
              <div className="mt-1 flex flex-col gap-1 px-2 text-xs text-sage-300">
                {outcome.map((slice) => {
                  const total = outcome.reduce((sum, item) => sum + item.value, 0);
                  const pct = total ? Math.round((slice.value / total) * 100) : 0;
                  return (
                    <span key={slice.name} className="flex items-center gap-1.5">
                      <span className="h-[3px] w-4 rounded-full" style={{ backgroundColor: slice.color }} />
                      <span className="flex-1">{slice.name}</span>
                      <span className="font-mono text-sage-400">
                        {slice.value} ({pct}%)
                      </span>
                    </span>
                  );
                })}
              </div>
            </ChartCard>
          </div>

          <div className="grid gap-6 xl:grid-cols-3">
            <ChartCard title="Average latency per stage">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={latencyByStage}
                  layout="vertical"
                  margin={{ top: 8, right: 40, bottom: 0, left: 8 }}
                >
                  <CartesianGrid stroke={GRID_STROKE} strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    tick={{ fill: AXIS_TICK, fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                    unit="s"
                  />
                  <YAxis
                    type="category"
                    dataKey="stage"
                    width={72}
                    tick={{ fill: AXIS_TICK, fontSize: 11 }}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip contentStyle={CHART_TOOLTIP_STYLE} cursor={{ fill: "rgba(204,204,220,0.06)" }} />
                  <Bar dataKey="seconds" name="Seconds" radius={[0, 2, 2, 0]} maxBarSize={22}>
                    {latencyByStage.map((row, index) => (
                      <Cell key={row.stage} fill={STAGE_COLORS[index % STAGE_COLORS.length]} />
                    ))}
                    <LabelList
                      dataKey="seconds"
                      position="right"
                      formatter={(value) => `${value}s`}
                      style={{ fill: AXIS_TICK, fontSize: 11 }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <div className="xl:col-span-2">
              <ChartCard title="Recent runs">
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead>
                      <tr className="border-b border-surface-600 text-xs uppercase tracking-wide text-sage-500">
                        <th className="pb-2 pr-4 font-medium">Date</th>
                        <th className="pb-2 pr-4 font-medium">Camera</th>
                        <th className="pb-2 pr-4 font-medium">Status</th>
                        <th className="pb-2 pr-4 font-medium">Zones</th>
                        <th className="pb-2 font-medium">Latency</th>
                      </tr>
                    </thead>
                    <tbody>
                      {recentRuns.map((run) => (
                        <tr key={run.id} className="border-b border-surface-700 text-sage-300">
                          <td className="py-2.5 pr-4 text-xs">
                            {String(run.createdAt || "").replace("T", " ").slice(0, 16) || "-"}
                          </td>
                          <td className="py-2.5 pr-4 text-xs">{run.cameraVariant || "-"}</td>
                          <td className="py-2.5 pr-4">
                            <span
                              className={`rounded-full border px-2 py-0.5 text-xs font-medium ${
                                isSuccess(run)
                                  ? "border-[#3cf28a]/40 bg-[#3cf28a]/10 text-[#3cf28a]"
                                  : "border-[#ff3b6b]/40 bg-[#ff3b6b]/10 text-[#ff3b6b]"
                              }`}
                            >
                              {run.status || "unknown"}
                            </span>
                          </td>
                          <td className="py-2.5 pr-4 text-xs">{run.zoneCount ?? "-"}</td>
                          <td className="py-2.5 text-xs">{formatSeconds(run.totalSeconds)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </ChartCard>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
