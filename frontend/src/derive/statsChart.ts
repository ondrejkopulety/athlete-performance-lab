/**
 * Stránka Stats: agregace jízd do grafu po týdnech (nebo dnech, když je
 * jízd málo) s přepínačem týdny/kumulativně a srovnáním s dřívějším rokem.
 *
 * Čistě odvozovací modul – žádný stav, jen `ActivityRow[]` → data pro graf.
 * Interaktivní stav (posuvník, výběr do lupy) drží Stats.tsx.
 */

import type { ActivityRow } from "../api";
import { tick } from "../format";

export type MetricKey =
  | "km"
  | "ascent"
  | "kcal"
  | "time"
  | "speed"
  | "fat"
  | "carb"
  | "up"
  | "down"
  | "flat";

export type NumericField = "km" | "asc" | "kcal" | "dur" | "up" | "down" | "flat" | "fat" | "carb";

export interface MetricDef {
  key: MetricKey;
  field: NumericField | null; // null = speed, počítá se z km/dur
  color: string;
  label: string;
  unit: string;
  fmt: (v: number) => string;
}

const fmtInt = (v: number) => Math.round(v).toLocaleString("cs-CZ");
const fmtMinShort = (m: number): string => {
  const h = Math.floor(m / 60);
  const r = Math.round(m % 60);
  return h ? `${h} h ${String(r).padStart(2, "0")} min` : `${r} min`;
};
const fmtDec1 = (v: number) => v.toFixed(1).replace(".", ",");

export const METRIC_DEFS: Record<MetricKey, MetricDef> = {
  km: { key: "km", field: "km", color: "var(--fg2)", label: "Vzdálenost", unit: "km", fmt: (v) => v.toFixed(0) },
  ascent: { key: "ascent", field: "asc", color: "var(--warn)", label: "Převýšení", unit: "m", fmt: fmtInt },
  kcal: { key: "kcal", field: "kcal", color: "var(--orange)", label: "Kalorie", unit: "kcal", fmt: fmtInt },
  time: { key: "time", field: "dur", color: "var(--fg2)", label: "Čas na kole", unit: "min", fmt: fmtMinShort },
  speed: { key: "speed", field: null, color: "var(--blue)", label: "Prům. rychlost", unit: "km/h", fmt: fmtDec1 },
  fat: { key: "fat", field: "fat", color: "var(--fatc)", label: "Tuky", unit: "g", fmt: fmtInt },
  carb: { key: "carb", field: "carb", color: "var(--carbc)", label: "Cukry", unit: "g", fmt: fmtInt },
  up: { key: "up", field: "up", color: "var(--warn)", label: "Do kopce", unit: "min", fmt: fmtMinShort },
  down: { key: "down", field: "down", color: "var(--ok)", label: "Z kopce", unit: "min", fmt: fmtMinShort },
  flat: { key: "flat", field: "flat", color: "var(--blue)", label: "Po rovině", unit: "min", fmt: fmtMinShort },
};

export function weekStart(d: string): string {
  const dt = new Date(`${d}T00:00:00Z`);
  const day = dt.getUTCDay() || 7;
  dt.setUTCDate(dt.getUTCDate() - day + 1);
  return dt.toISOString().slice(0, 10);
}

export function weekRangeLabel(k: string): string {
  const s = new Date(`${k}T00:00:00Z`);
  const e = new Date(s);
  e.setUTCDate(e.getUTCDate() + 6);
  return `${tick(k)}–${tick(e.toISOString().slice(0, 10))}`;
}

interface Bucket {
  km: number;
  dur: number;
  val: number;
}

function group(rows: ActivityRow[], metric: MetricDef, daily: boolean): Map<string, Bucket> {
  const groups = new Map<string, Bucket>();
  for (const r of rows) {
    const k = daily ? r.d : weekStart(r.d);
    let b = groups.get(k);
    if (!b) {
      b = { km: 0, dur: 0, val: 0 };
      groups.set(k, b);
    }
    b.km += r.km ?? 0;
    b.dur += r.dur ?? 0;
    if (metric.field) b.val += (r[metric.field] as number | null) ?? 0;
  }
  return groups;
}

function seriesValues(
  keys: string[],
  groups: Map<string, Bucket>,
  metric: MetricDef,
  cumulative: boolean,
): number[] {
  if (metric.key === "speed") {
    let cumKm = 0;
    let cumDur = 0;
    return keys.map((k) => {
      const g = groups.get(k) as Bucket;
      if (cumulative) {
        cumKm += g.km;
        cumDur += g.dur;
        return cumDur ? cumKm / (cumDur / 60) : 0;
      }
      return g.dur ? g.km / (g.dur / 60) : 0;
    });
  }
  let run = 0;
  return keys.map((k) => {
    const v = (groups.get(k) as Bucket).val;
    if (cumulative) {
      run += v;
      return run;
    }
    return v;
  });
}

export interface ChartPoint {
  x: number;
  y: number;
}

export interface StatsChartData {
  metric: MetricKey;
  color: string;
  label: string;
  unit: string;
  daily: boolean;
  keys: string[];
  isBar: boolean;
  isLine: boolean;
  bars: { h: string; tick: string }[];
  points: ChartPoint[];
  linePath: string;
  areaPath: string;
  fmtVals: string[];
  max: string;
  rawMax: number;
}

/** Základní stavba grafu jedné metriky nad danou množinou jízd. */
export function buildStatsChart(
  metricKey: MetricKey,
  rows: ActivityRow[],
  mode: "weekly" | "cumulative",
): StatsChartData | null {
  if (!rows.length) return null;
  const metric = METRIC_DEFS[metricKey];

  let daily = false;
  let groups = group(rows, metric, false);
  let keys = [...groups.keys()].sort();
  if (keys.length < 15) {
    daily = true;
    groups = group(rows, metric, true);
    keys = [...groups.keys()].sort();
  }

  const cumulative = mode === "cumulative";
  const vals = seriesValues(keys, groups, metric, cumulative);
  const max = Math.max(...vals, 1);
  const tickEvery = Math.max(1, Math.ceil(keys.length / 9));
  const bars = keys.map((k, i) => ({
    h: `${((vals[i] / max) * 100).toFixed(1)}%`,
    tick: i % tickEvery === 0 ? tick(k) : "",
  }));

  const n = vals.length;
  const points = vals.map((v, i) => ({
    x: n > 1 ? (i / (n - 1)) * 700 : 350,
    y: 120 - (v / max) * 120,
  }));
  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(" ");
  const areaPath = points.length
    ? `${linePath} L${points[points.length - 1].x.toFixed(1)} 120 L${points[0].x.toFixed(1)} 120 Z`
    : "";
  const fmtVals = vals.map((v) => metric.fmt(v) + (metric.unit === "min" ? "" : ` ${metric.unit}`));

  return {
    metric: metricKey,
    color: metric.color,
    label: metric.label,
    unit: metric.unit,
    daily,
    keys,
    isBar: !cumulative,
    isLine: cumulative,
    bars,
    points,
    linePath,
    areaPath,
    fmtVals,
    max: metric.fmt(max),
    rawMax: max,
  };
}

/** Hodnoty srovnávacího roku na stejné mřížce (dny/týdny) jako hlavní graf. */
export function buildCompareSeries(
  metricKey: MetricKey,
  rows: ActivityRow[],
  mode: "weekly" | "cumulative",
  daily: boolean,
): number[] {
  const metric = METRIC_DEFS[metricKey];
  const groups = group(rows, metric, daily);
  const keys = [...groups.keys()].sort();
  return seriesValues(keys, groups, metric, mode === "cumulative");
}

export function totalOf(metricKey: MetricKey, rows: ActivityRow[]): number {
  const metric = METRIC_DEFS[metricKey];
  if (metricKey === "speed") {
    const dur = rows.reduce((a, r) => a + (r.dur ?? 0), 0);
    const km = rows.reduce((a, r) => a + (r.km ?? 0), 0);
    return dur ? km / (dur / 60) : 0;
  }
  return rows.reduce((a, r) => a + ((r[metric.field as NumericField] ?? 0) as number), 0);
}

export function fmtMetricTotal(metricKey: MetricKey, raw: number): string {
  const metric = METRIC_DEFS[metricKey];
  return metric.fmt(raw) + (metric.unit === "min" ? "" : ` ${metric.unit}`);
}
