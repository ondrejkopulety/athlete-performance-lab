/**
 * Karta „Tepová regenerace · 60 s" – pokles tepu první minutu po zátěži.
 * Čára je surová hodnota, tučná čára klouzavý průměr z 5 jízd.
 */

import type { ActivityRow } from "../api";
import { shortDate } from "../format";
import type { Theme } from "../theme";

const W = 320;
const H = 72;
const P = 8;
/** Hodnota, od které se pokles považuje za dobrý (zvýrazněný pás v grafu). */
const GOOD = 50;

export interface Hrr {
  hasData: boolean;
  count: number;
  last: string;
  color: string;
  unit: string;
  trend: string;
  trendColor: string;
  line: string;
  trendLine: string;
  area: string;
  goodY: string;
  goodH: string;
  hi: number | string;
  lo: number | string;
  selX: string;
  selPct: string;
  selLabel: string;
  crossOpacity: string;
  lastX: string;
  lastY: string;
  from: string;
  to: string;
}

export function buildHrr(
  activities: ActivityRow[],
  theme: Theme,
  selected: number | null,
  scrubbing: boolean,
): Hrr {
  const src = activities.filter((a) => a.hrr != null);

  if (src.length < 2) {
    return {
      hasData: false,
      count: src.length,
      last: src.length ? String(Math.round(src[0].hrr as number)) : "–",
      color: theme.mut,
      unit: "tepů",
      trend: "málo dat v období",
      trendColor: theme.faint,
      line: "",
      trendLine: "",
      area: "",
      goodY: "0",
      goodH: "0",
      hi: "",
      lo: "",
      selX: "-10",
      selPct: "50%",
      selLabel: "",
      crossOpacity: "0",
      lastX: "-10",
      lastY: "-10",
      from: "",
      to: "",
    };
  }

  const vals = src.map((a) => a.hrr as number);
  const lo = Math.min(...vals, 40) - 4;
  const hi = Math.max(...vals, 60) + 4;
  const x = (i: number) => (src.length === 1 ? W / 2 : (i / (src.length - 1)) * W);
  const y = (v: number) => P + (1 - (v - lo) / (hi - lo)) * (H - P * 2);
  const path = (arr: number[]) =>
    arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

  const k = Math.min(5, src.length);
  const smooth = vals.map((_, i) => {
    const w = vals.slice(Math.max(0, i - k + 1), i + 1);
    return w.reduce((a, b) => a + b, 0) / w.length;
  });

  const headline = vals.reduce((a, b) => a + b, 0) / vals.length;
  const sIdx = selected != null && selected < vals.length ? selected : null;
  const shown = sIdx == null ? headline : vals[sIdx];
  const diff = smooth[smooth.length - 1] - smooth[Math.max(0, Math.floor(smooth.length / 4))];

  const spanYears = src[0].d.slice(0, 4) !== src[src.length - 1].d.slice(0, 4);
  const fmt = (d: string) => shortDate(d, spanYears);
  const markIdx = sIdx == null ? vals.length - 1 : sIdx;

  return {
    hasData: true,
    count: src.length,
    last: String(Math.round(shown)),
    color: shown >= 50 ? theme.ok : shown >= 40 ? theme.warn : theme.bad,
    unit: sIdx == null ? "tepů · průměr" : `tepů · ${fmt(src[sIdx].d)}`,
    trend: `${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(1).replace(".", ",")} za období`,
    trendColor: diff >= 0 ? theme.ok : theme.bad,
    line: path(vals),
    trendLine: path(smooth),
    area: `${path(smooth)} L${W} ${H} L0 ${H} Z`,
    goodY: y(hi).toFixed(1),
    goodH: Math.max(0, y(GOOD) - y(hi)).toFixed(1),
    hi: Math.round(hi),
    lo: Math.round(lo),
    selX: x(markIdx).toFixed(1),
    selPct: `${Math.min(88, Math.max(12, (markIdx / Math.max(1, vals.length - 1)) * 100)).toFixed(2)}%`,
    selLabel: fmt(src[markIdx].d),
    crossOpacity: scrubbing ? "1" : "0",
    lastX: x(markIdx).toFixed(1),
    lastY: y(vals[markIdx]).toFixed(1),
    from: fmt(src[0].d),
    to: fmt(src[src.length - 1].d),
  };
}
