/**
 * Drilldown metriky (HRV / RHR) – 1:1 s HRV.dc.html / RHR.dc.html.
 * Nad surovou denní řadou z `/api/coach/history/{metric}` se na klientu
 * spočítá vše: klouzavé průměry, odchylka od průměru, distribuce, rozpad
 * podle dne v týdnu. Stejný vzor jako `buildPolarization`.
 */

import type { MetricHistoryPoint } from "../api";
import { mon as monLabel, shortDate } from "../format";
import type { Theme } from "../theme";

export type MetricWhich = "hrv" | "rhr" | "spanek";

export interface MetricConfig {
  which: MetricWhich;
  apiKey: string;
  title: string;
  eyebrow: string;
  unit: string;
  lowerBetter: boolean;
  idealLo: number;
  idealHi: number;
  about: string[];
  buckets: { label: string; test: (v: number) => boolean; color: string }[];
}

export function metricConfig(which: MetricWhich): MetricConfig {
  if (which === "rhr") {
    return {
      which,
      apiKey: "rhr_day",
      title: "Klidová srdeční frekvence",
      eyebrow: "RHR",
      unit: "bpm",
      lowerBetter: true,
      idealLo: 45,
      idealHi: 58,
      about: [
        "Klidová srdeční frekvence je počet tepů za minutu v klidu.",
        "Nižší hodnoty obvykle znamenají lepší kardiovaskulární kondici a regeneraci.",
      ],
      buckets: [
        { label: "< 45 bpm", test: (v) => v < 45, color: "var(--ok)" },
        { label: "45–50 bpm", test: (v) => v >= 45 && v < 50, color: "var(--blue)" },
        { label: "50–55 bpm", test: (v) => v >= 50 && v < 55, color: "var(--warn)" },
        { label: "> 55 bpm", test: (v) => v >= 55, color: "var(--bad)" },
      ],
    };
  }
  return {
    which: "hrv",
    apiKey: "hrv_last_night",
    title: "Variabilita srdečního tepu",
    eyebrow: "HRV",
    unit: "ms",
    lowerBetter: false,
    idealLo: 60,
    idealHi: 80,
    about: [
      "HRV odráží schopnost těla zvládat stres a regenerovat.",
      "Vyšší hodnoty obvykle znamenají lepší připravenost a odolnost.",
    ],
    buckets: [
      { label: "< 50 ms", test: (v) => v < 50, color: "var(--bad)" },
      { label: "50–60 ms", test: (v) => v >= 50 && v < 60, color: "var(--warn)" },
      { label: "60–80 ms", test: (v) => v >= 60 && v < 80, color: "var(--blue)" },
      { label: "> 80 ms", test: (v) => v >= 80, color: "var(--ok)" },
    ],
  };
}

interface Row {
  d: string;
  v: number;
}

const W = 700;
const H = 220;
const DW = 700;
const DH = 140;
const WD_ORDER = ["Pondělí", "Úterý", "Středa", "Čtvrtek", "Pátek", "Sobota", "Neděle"];
const WD_NAME = ["Neděle", "Pondělí", "Úterý", "Středa", "Čtvrtek", "Pátek", "Sobota"];
const DONUT_C = 2 * Math.PI * 46;

export interface Tick {
  top: string;
  label: string;
}
export interface DevBar {
  x: string;
  y: string;
  w: string;
  h: string;
  color: string;
  op: string;
}
export interface DonutSeg {
  color: string;
  dash: string;
  offset: string;
}
export interface WeekdayRow {
  name: string;
  value: string;
  squares: { color: string }[];
}

export interface MetricView {
  hasData: boolean;
  // summary
  avg: string;
  deltaLabel: string;
  deltaColor: string;
  maxVal: string;
  maxDate: string;
  minVal: string;
  minDate: string;
  trendLabel: string;
  trendColor: string;
  trendRotate: string;
  // time chart
  linePath: string;
  areaPath: string;
  avg7Path: string;
  avg30Path: string;
  yTicks: Tick[];
  gridLines: { y: string }[];
  xTicks: string[];
  peak: { x: string; y: string; labelTop: string; labelLeft: string };
  trough: { x: string; y: string; labelTop: string; labelLeft: string };
  scrub: { x: string; y: string; left: string; label: string; opacity: string };
  count: number;
  // deviation
  devBars: DevBar[];
  devYTicks: Tick[];
  devGridLines: { y: string }[];
  devZeroY: string;
  todayDevLabel: string;
  todayDevColor: string;
  devScrubLabel: string;
  // distribution
  donut: DonutSeg[];
  donutLegend: { color: string; label: string; pct: number }[];
  idealText: string;
  // weekday
  weekdays: WeekdayRow[];
}

function movAvg(vals: number[], win: number): number[] {
  return vals.map((_, i) => {
    const w = vals.slice(Math.max(0, i - win + 1), i + 1);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });
}

function niceTicks(l: number, h: number, n: number): number[] {
  const step = Math.ceil((h - l) / (n - 1) / 10) * 10 || 10;
  const start = Math.floor(l / step) * step;
  const out: number[] = [];
  for (let v = start; v <= h + 0.001; v += step) out.push(v);
  return out;
}

export function buildMetricHistory(
  points: MetricHistoryPoint[],
  cfg: MetricConfig,
  theme: Theme,
  rangeDays: number,
  devWindow: 7 | 30,
  selIdx: number | null,
  scrubbing: boolean,
): MetricView {
  const all: Row[] = points
    .filter((p): p is { date: string; value: number } => p.value != null && p.date != null)
    .map((p) => ({ d: p.date.slice(0, 10), v: p.value }));
  const rows = rangeDays > 0 ? all.slice(-rangeDays) : all;

  const empty: MetricView = {
    hasData: false,
    avg: "–", deltaLabel: "–", deltaColor: theme.mut, maxVal: "–", maxDate: "–",
    minVal: "–", minDate: "–", trendLabel: "–", trendColor: theme.mut, trendRotate: "0deg",
    linePath: "", areaPath: "", avg7Path: "", avg30Path: "", yTicks: [], gridLines: [], xTicks: [],
    peak: { x: "-10", y: "-10", labelTop: "-100px", labelLeft: "-100px" },
    trough: { x: "-10", y: "-10", labelTop: "-100px", labelLeft: "-100px" },
    scrub: { x: "-10", y: "-10", left: "-100px", label: "", opacity: "0" },
    count: 0,
    devBars: [], devYTicks: [], devGridLines: [], devZeroY: "70",
    todayDevLabel: "–", todayDevColor: theme.mut, devScrubLabel: "",
    donut: [], donutLegend: [], idealText: `Ideální rozsah: ${cfg.idealLo}–${cfg.idealHi} ${cfg.unit}`,
    weekdays: [],
  };
  if (rows.length === 0) return empty;

  const dfmt = cfg.unit === "ms" ? 0 : 1;
  const num = (v: number) => v.toFixed(dfmt).replace(".", ",");
  const vals = rows.map((r) => r.v);
  const lo = Math.min(...vals) - 3;
  const hi = Math.max(...vals) + 3;
  const x = (i: number) => (rows.length === 1 ? W / 2 : (i / (rows.length - 1)) * W);
  const y = (v: number) => 14 + (1 - (v - lo) / (hi - lo)) * (H - 28);
  const linePts = (arr: number[]) => arr.map((v, i) => [x(i), y(v)] as [number, number]);
  const path = (arr: number[]) =>
    linePts(arr).map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");

  const linePath = path(vals);
  const areaPath = `${linePath} L${W} ${H} L0 ${H} Z`;
  const avg7 = movAvg(vals, 7);
  const avg30 = movAvg(vals, 30);

  const avg = vals.reduce((a, b) => a + b, 0) / vals.length;
  let maxIdx = 0;
  let minIdx = 0;
  vals.forEach((v, i) => {
    if (v > vals[maxIdx]) maxIdx = i;
    if (v < vals[minIdx]) minIdx = i;
  });
  const withYear = rows[0].d.slice(0, 4) !== rows[rows.length - 1].d.slice(0, 4);

  // delta vs. předchozí okno stejné délky (mimo aktuální řadu)
  const prevWindow = all.slice(Math.max(0, all.length - rows.length * 2), all.length - rows.length);
  const prevAvg = prevWindow.length ? prevWindow.reduce((a, r) => a + r.v, 0) / prevWindow.length : avg;
  const delta = avg - prevAvg;
  const deltaImproving = cfg.lowerBetter ? delta < 0 : delta > 0;
  const deltaFlat = Math.abs(delta) < (cfg.unit === "ms" ? 1 : 0.5);
  const deltaLabel = prevWindow.length
    ? `${delta >= 0 ? "+" : "−"}${Math.abs(delta).toFixed(1).replace(".", ",")} ${cfg.unit}`
    : "bez srovnání";
  const deltaColor = deltaFlat ? theme.mut : deltaImproving ? theme.ok : theme.bad;

  // trend: sklon posledních 7 dní
  const tail = vals.slice(-7);
  const prevTail = vals.slice(-14, -7);
  const tAvg = tail.length ? tail.reduce((a, b) => a + b, 0) / tail.length : avg;
  const pAvg = prevTail.length ? prevTail.reduce((a, b) => a + b, 0) / prevTail.length : tAvg;
  const tDiff = tAvg - pAvg;
  const tFlat = Math.abs(tDiff) < (cfg.unit === "ms" ? 1.5 : 0.8);
  const tImproving = cfg.lowerBetter ? tDiff < 0 : tDiff > 0;
  const trendLabel = tFlat ? "Stabilní" : tImproving ? "Zlepšuje se" : "Zhoršuje se";
  const trendColor = tFlat ? theme.mut : tImproving ? theme.ok : theme.bad;
  // šipka: nahoru pro rostoucí hodnotu, dolů pro klesající (bez ohledu na „lepší")
  const trendRotate = tFlat ? "0deg" : tDiff > 0 ? "-45deg" : "45deg";

  const yTicksVals = niceTicks(lo, hi, 5).reverse();
  const yTicks = yTicksVals.map((v) => ({ top: `${((y(v) / H) * 100).toFixed(1)}%`, label: String(Math.round(v)) }));
  const gridLines = yTicksVals.slice(1, -1).map((v) => ({ y: y(v).toFixed(1) }));

  const tickCount = Math.min(6, rows.length);
  const multiYear = rows[0].d.slice(0, 4) !== rows[rows.length - 1].d.slice(0, 4);
  const xTicks = Array.from({ length: tickCount }, (_, i) => {
    const idx = Math.round((i / (tickCount - 1 || 1)) * (rows.length - 1));
    return rows.length > 90 ? monLabel(rows[idx].d, multiYear) : shortDate(rows[idx].d, false);
  });

  // ── odchylka od průměru ──
  const maDev = devWindow === 30 ? avg30 : avg7;
  const devVals = vals.map((v, i) => v - maDev[i]);
  const maxAbsDev = Math.max(4, Math.ceil(Math.max(...devVals.map((v) => Math.abs(v))) / 2) * 2);
  const devScale = (DH / 2 - 8) / maxAbsDev;
  const devY0 = DH / 2;
  const barW = Math.max(2, (DW / rows.length) * 0.55);
  const GREY_T = cfg.unit === "ms" ? 1 : 0.7;
  const devBars: DevBar[] = devVals.map((v, i) => {
    const h = Math.abs(v) * devScale;
    const up = v >= 0;
    return {
      x: (x(i) - barW / 2).toFixed(1),
      y: (up ? devY0 - h : devY0).toFixed(1),
      w: barW.toFixed(1),
      h: Math.max(1.5, h).toFixed(1),
      color: v > GREY_T ? "var(--ok)" : v < -GREY_T ? "var(--bad)" : "var(--grey)",
      op: selIdx == null || selIdx === i ? "1" : "0.4",
    };
  });
  const devTicksVals = [maxAbsDev, maxAbsDev / 2, 0, -maxAbsDev / 2, -maxAbsDev];
  const devYTicks = devTicksVals.map((v) => ({
    top: `${(((devY0 - v * devScale) / DH) * 100).toFixed(1)}%`,
    label: `${v > 0 ? "+" : ""}${Math.round(v)}`,
  }));
  const devGridLines = devTicksVals.filter((v) => v !== 0).map((v) => ({ y: (devY0 - v * devScale).toFixed(1) }));
  const todayDev = Math.round(devVals[devVals.length - 1]);
  const todayDevColor = todayDev > GREY_T ? theme.ok : todayDev < -GREY_T ? theme.bad : theme.mut;
  const todayDevLabel = `${todayDev >= 0 ? "+" : ""}${todayDev} ${cfg.unit}`;

  // ── scrub ──
  const sIdx = selIdx != null && selIdx < rows.length ? selIdx : null;
  const scrubX = sIdx == null ? -10 : x(sIdx);
  const scrub = {
    x: scrubX.toFixed(1),
    y: sIdx == null ? "-10" : y(rows[sIdx].v).toFixed(1),
    left: `${((sIdx == null ? 0 : sIdx / Math.max(1, rows.length - 1)) * 100).toFixed(2)}%`,
    label: sIdx == null ? "" : `${shortDate(rows[sIdx].d, withYear)} · ${num(rows[sIdx].v)} ${cfg.unit}`,
    opacity: scrubbing && sIdx != null ? "1" : "0",
  };
  const devScrubLabel = scrub.label;

  // ── distribuce ──
  const counts = cfg.buckets.map((b) => vals.filter(b.test).length);
  const total = vals.length || 1;
  const pcts = counts.map((c) => Math.round((c / total) * 100));
  let cum = 0;
  const donut: DonutSeg[] = cfg.buckets.map((b, i) => {
    const dash = (pcts[i] / 100) * DONUT_C;
    const seg = { color: b.color, dash: `${dash.toFixed(1)} ${(DONUT_C - dash).toFixed(1)}`, offset: (-cum).toFixed(1) };
    cum += dash;
    return seg;
  });
  const donutLegend = cfg.buckets.map((b, i) => ({ color: b.color, label: b.label, pct: pcts[i] }));

  // ── rozpad podle dne v týdnu ──
  const wdSums: Record<string, number> = {};
  const wdCounts: Record<string, number> = {};
  rows.forEach((r) => {
    const name = WD_NAME[new Date(`${r.d}T00:00:00Z`).getUTCDay()];
    wdSums[name] = (wdSums[name] ?? 0) + r.v;
    wdCounts[name] = (wdCounts[name] ?? 0) + 1;
  });
  const wdAvgs = WD_ORDER.map((name) => ({ name, avg: wdCounts[name] ? wdSums[name] / wdCounts[name] : 0 }));
  const wdMax = Math.max(...wdAvgs.map((w) => w.avg));
  const wdMin = Math.min(...wdAvgs.map((w) => w.avg));
  const weekdays: WeekdayRow[] = wdAvgs.map((w) => {
    const filled = wdMax === wdMin ? 5 : Math.round(1 + ((w.avg - wdMin) / (wdMax - wdMin)) * 7);
    return {
      name: w.name,
      value: w.avg ? String(Math.round(w.avg)) : "–",
      squares: Array.from({ length: 8 }, (_, i) => ({
        color: i < filled ? (w.avg === wdMax ? "var(--ok)" : "var(--bad)") : "var(--track)",
      })),
    };
  });

  return {
    hasData: true,
    avg: num(avg),
    deltaLabel,
    deltaColor,
    maxVal: num(vals[maxIdx]),
    maxDate: shortDate(rows[maxIdx].d, withYear),
    minVal: num(vals[minIdx]),
    minDate: shortDate(rows[minIdx].d, withYear),
    trendLabel,
    trendColor,
    trendRotate,
    linePath,
    areaPath,
    avg7Path: path(avg7),
    avg30Path: path(avg30),
    yTicks,
    gridLines,
    xTicks,
    peak: {
      x: x(maxIdx).toFixed(1),
      y: y(vals[maxIdx]).toFixed(1),
      labelTop: `${((y(vals[maxIdx]) / H) * 100).toFixed(1)}%`,
      labelLeft: `${((maxIdx / Math.max(1, rows.length - 1)) * 100).toFixed(2)}%`,
    },
    trough: {
      x: x(minIdx).toFixed(1),
      y: y(vals[minIdx]).toFixed(1),
      labelTop: `${((y(vals[minIdx]) / H) * 100).toFixed(1)}%`,
      labelLeft: `${((minIdx / Math.max(1, rows.length - 1)) * 100).toFixed(2)}%`,
    },
    scrub,
    count: rows.length,
    devBars,
    devYTicks,
    devGridLines,
    devZeroY: devY0.toFixed(1),
    todayDevLabel,
    todayDevColor,
    devScrubLabel,
    donut,
    donutLegend,
    idealText: `${cfg.idealLo}–${cfg.idealHi} ${cfg.unit}`,
    weekdays,
  };
}
