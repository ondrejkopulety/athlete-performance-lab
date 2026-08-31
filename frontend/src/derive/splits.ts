/**
 * „Průměrný tep po 5 km" – odvozeno na klientu z vteřinových `records`
 * (mají kumulativní `distance` v metrech a `heart_rate`). Tabulka
 * `activity_hr_by_distance_bucket` v DB zatím není, tohle ji nahrazuje bez
 * zásahu do backendu.
 */

import type { RecordPoint } from "../api";
import { dec } from "../format";
import type { Theme } from "../theme";

export interface Split {
  range: string;
  hr: string;
  hrNum: number | null;
  speed: string;
  w: string;
  color: string;
}

const BUCKET_M = 5000;

export function buildSplits(records: RecordPoint[], theme: Theme): Split[] {
  const pts = records
    .filter((r) => r.distance != null && r.heart_rate != null)
    .map((r) => ({ dist: r.distance as number, hr: r.heart_rate as number, spd: r.speed ?? null }));
  if (pts.length < 2) return [];

  const maxDist = pts[pts.length - 1].dist;
  const nBuckets = Math.max(1, Math.ceil(maxDist / BUCKET_M));

  const acc = Array.from({ length: nBuckets }, () => ({ hrSum: 0, spdSum: 0, spdN: 0, n: 0 }));
  for (const p of pts) {
    const b = Math.min(nBuckets - 1, Math.floor(p.dist / BUCKET_M));
    acc[b].hrSum += p.hr;
    acc[b].n += 1;
    if (p.spd != null) {
      acc[b].spdSum += p.spd;
      acc[b].spdN += 1;
    }
  }

  const hrAvgs = acc.map((a) => (a.n ? a.hrSum / a.n : null));
  const known = hrAvgs.filter((v): v is number => v != null);
  const maxHr = known.length ? Math.max(...known) : 1;

  const color = (hr: number | null) =>
    hr == null
      ? theme.mut
      : hr >= 170
        ? theme.bad
        : hr >= 155
          ? theme.orange
          : hr >= 140
            ? theme.warn
            : theme.blue;

  return acc.map((a, i) => {
    const hr = hrAvgs[i];
    const spd = a.spdN ? (a.spdSum / a.spdN) * 3.6 : null;
    const from = (i * BUCKET_M) / 1000;
    const to = Math.min(maxDist / 1000, ((i + 1) * BUCKET_M) / 1000);
    return {
      range: `${from.toFixed(0)}–${to.toFixed(0)} km`,
      hr: hr == null ? "–" : String(Math.round(hr)),
      hrNum: hr == null ? null : Math.round(hr),
      speed: spd == null ? "–" : `${dec(spd, 1)} km/h`,
      w: `${hr == null ? 0 : (hr / maxHr) * 100}%`,
      color: color(hr),
    };
  });
}
