/**
 * Souvislé bloky nad prahem zóny pro detail jízdy – 1:1 s bloky v
 * Aktivita.dc.html, ale spočítané na klientu z (downsamplovaných) `records`.
 * Backend per-aktivita blokový endpoint neexistuje; tady stačí `hr_zone` +
 * timestamp a přemostění krátkých propadů (tolerance 0 / 15 s).
 */

import type { RecordPoint } from "../api";
import { fmtMin } from "../format";

const ZONE_RANK: Record<string, number> = { Z1: 1, Z2: 2, Z3: 3, Z4: 4, Z5: 5 };

const BUCKETS: { label: string; lo: number; hi: number }[] = [
  { label: "< 0,5 min", lo: 0, hi: 0.5 },
  { label: "0,5–1 min", lo: 0.5, hi: 1 },
  { label: "1–5 min", lo: 1, hi: 5 },
  { label: "5–10 min", lo: 5, hi: 10 },
  { label: "10+ min", lo: 10, hi: Infinity },
];

export interface BlockBucket {
  label: string;
  count: string;
  min: string;
  w: string;
}

export interface ActivityBlockCard {
  key: string;
  title: string;
  color: string;
  thresholdCaption: string;
  isNormal: boolean;
  isEmpty: boolean;
  isNoData: boolean;
  longest: string;
  buckets: BlockBucket[];
  countFooter: string;
  inBlocksFooter: string;
  emptyMsg: string;
}

const CONFIGS = [
  { key: "z4", zone: "Z4", label: "Souvislé bloky · nad Z4", color: "var(--z4)" },
  { key: "z2", zone: "Z2", label: "Souvislé bloky · Z2", color: "var(--z2)" },
  { key: "z1", zone: "Z1", label: "Souvislé bloky · Z1", color: "var(--z1)" },
] as const;

const LONG_MIN = 3;

/** Vrátí délky souvislých úseků (minuty) nad `minRank` s přemostěním `tolS`. */
function segments(records: RecordPoint[], minRank: number, tolS: number): number[] {
  const pts = records
    .filter((r) => r.hr_zone != null && r.ts != null)
    .map((r) => ({ t: new Date(r.ts).getTime() / 1000, rank: ZONE_RANK[r.hr_zone as string] ?? 0 }));
  if (pts.length < 2) return [];

  const out: number[] = [];
  let start: number | null = null;
  let lastAbove = 0;
  let belowSince: number | null = null;

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const above = p.rank >= minRank;
    if (above) {
      if (start == null) start = p.t;
      lastAbove = p.t;
      belowSince = null;
    } else {
      if (start != null) {
        if (belowSince == null) belowSince = p.t;
        if (p.t - (belowSince as number) > tolS) {
          out.push((lastAbove - start) / 60);
          start = null;
          belowSince = null;
        }
      }
    }
  }
  if (start != null) out.push((lastAbove - start) / 60);
  return out.filter((m) => m > 0);
}

function fmtBlk(min: number): string {
  const s = Math.round(min * 60);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function buildActivityBlocks(
  records: RecordPoint[],
  tolerances: Record<string, number>,
  hasRecords: boolean,
): ActivityBlockCard[] {
  return CONFIGS.map((cfg) => {
    const tolS = tolerances[cfg.key] ?? 0;
    const segs = hasRecords ? segments(records, ZONE_RANK[cfg.zone], tolS) : [];
    const total = segs.reduce((a, b) => a + b, 0);
    const inLong = segs.filter((m) => m >= LONG_MIN).reduce((a, b) => a + b, 0);
    const longest = segs.length ? Math.max(...segs) : 0;

    const maxBucketMin = Math.max(
      1e-6,
      ...BUCKETS.map((b) => segs.filter((m) => m >= b.lo && m < b.hi).reduce((a, m) => a + m, 0)),
    );
    const buckets: BlockBucket[] = BUCKETS.map((b) => {
      const inB = segs.filter((m) => m >= b.lo && m < b.hi);
      const sum = inB.reduce((a, m) => a + m, 0);
      return {
        label: b.label,
        count: `${inB.length}×`,
        min: sum === 0 ? "0 min" : sum < 1 ? "<1 min" : `${Math.round(sum)} min`,
        w: sum === 0 ? "0%" : `${Math.max(4, (sum / maxBucketMin) * 100).toFixed(1)}%`,
      };
    });

    return {
      key: cfg.key,
      title: cfg.label,
      color: cfg.color,
      thresholdCaption: `nad ${cfg.zone}`,
      isNoData: !hasRecords,
      isEmpty: hasRecords && segs.length === 0,
      isNormal: hasRecords && segs.length > 0,
      longest: fmtBlk(longest),
      buckets,
      countFooter: `${segs.length} úseků · ${fmtMin(total)} v ${cfg.zone} a výš celkem`,
      inBlocksFooter: `${fmtMin(inLong)} v souvislých blocích (≥ ${LONG_MIN} min)`,
      emptyMsg: `Bez souvislého úseku v ${cfg.zone} v této jízdě.`,
    };
  });
}
