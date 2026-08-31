/**
 * Výškový profil pro záložku „Stoupání" v detailu jízdy – z `records.altitude`
 * proti kumulativní vzdálenosti. Kategorizace kopců („Vyhodnocené kopce")
 * předpočítaná není, takže tu je jen samotný profil + souhrn stoupání.
 */

import type { ActivityDetail, RecordPoint } from "../api";
import { dec } from "../format";

export interface ElevationProfile {
  line: string;
  area: string;
  minEl: number;
  maxEl: number;
  distKm: number;
  distAxis: string[];
  elAxis: { label: string; top: string }[];
  empty: boolean;
  ascent: string;
  descent: string;
  avgGrad: string;
}

const W = 700;
const H = 200;
const PAD = 8;

export function buildElevation(
  records: RecordPoint[],
  activity: ActivityDetail | null,
): ElevationProfile {
  const pts = records
    .filter((r) => r.altitude != null && r.distance != null)
    .map((r) => ({ d: r.distance as number, el: r.altitude as number }));

  if (pts.length < 2) {
    return {
      line: "",
      area: "",
      minEl: 0,
      maxEl: 0,
      distKm: 0,
      distAxis: [],
      elAxis: [],
      empty: true,
      ascent: "–",
      descent: "–",
      avgGrad: "–",
    };
  }

  const els = pts.map((p) => p.el);
  const minEl = Math.min(...els);
  const maxEl = Math.max(...els);
  const lo = minEl - 8;
  const hi = maxEl + 8;
  const maxD = pts[pts.length - 1].d || 1;

  const x = (d: number) => (d / maxD) * W;
  const y = (el: number) => PAD + (1 - (el - lo) / (hi - lo)) * (H - PAD * 2);

  const line = pts
    .map((p, i) => `${i ? "L" : "M"}${x(p.d).toFixed(1)} ${y(p.el).toFixed(1)}`)
    .join(" ");
  const area = `${line} L${W} ${H} L0 ${H} Z`;

  const distKm = maxD / 1000;

  return {
    line,
    area,
    minEl: Math.round(minEl),
    maxEl: Math.round(maxEl),
    distKm,
    distAxis: [0, 0.25, 0.5, 0.75, 1].map((f) => `${(distKm * f).toFixed(0)} km`),
    elAxis: [0.2, 0.5, 0.8].map((f) => ({
      label: `${Math.round(lo + (1 - f) * (hi - lo))} m`,
      top: `${(f * 100).toFixed(0)}%`,
    })),
    empty: false,
    ascent: activity?.ascent_m == null ? "–" : `${Math.round(activity.ascent_m)} m`,
    descent: activity?.descent_m == null ? "–" : `${Math.round(activity.descent_m)} m`,
    avgGrad:
      activity?.distance_km && activity?.ascent_m != null && activity.distance_km > 0
        ? `${dec((activity.ascent_m / (activity.distance_km * 1000)) * 100, 1)} %`
        : "–",
  };
}
