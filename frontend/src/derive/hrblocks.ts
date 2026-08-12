/**
 * Panel „Souvislé bloky" – jak dlouho vydrží tep nad prahem v kuse.
 *
 * Práh přichází ze serveru už vyřešený (LTHR → nejbližší práh uložené
 * mřížky). Tady se jen formátuje.
 *
 * Histogram ukazuje počet úseků **i** součet minut. Obojí schválně: 199 úseků
 * pod 30 sekund vypadá jinak než 18 minut, které dohromady dají, a bez toho
 * druhého čísla se z fragmentace nedá poznat, jestli šlo o promarněnou
 * hodinu nebo o šum na hranici prahu.
 */

import type { HrBlocksPayload } from "../api";
import { fmtMin, shortDate, signed } from "../format";
import type { Theme } from "../theme";

export interface HistRow {
  bucket: string;
  count: string;
  time: string;
  w: string;
}

export interface HrBlocksView {
  hasData: boolean;
  headline: string;
  headlineNote: string;
  thresholdLabel: string;
  trend: string | null;
  trendColor: string;
  hist: HistRow[];
  totals: string;
  intentional: string;
  ridesNote: string;
  emptyNote: string | null;
}

/** Sekundy jako „18:00" (min:s) nebo „2:03 h" u delších úseků. */
export function fmtBlock(seconds: number): string {
  if (seconds >= 3600) {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    return `${h}:${String(m).padStart(2, "0")} h`;
  }
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function buildHrBlocks(
  payload: HrBlocksPayload | null,
  theme: Theme,
  mounted: boolean,
): HrBlocksView {
  const empty: HrBlocksView = {
    hasData: false,
    headline: "–",
    headlineNote: "",
    thresholdLabel: "",
    trend: null,
    trendColor: theme.mut,
    hist: [],
    totals: "",
    intentional: "",
    ridesNote: "",
    emptyNote: null,
  };
  if (!payload) return empty;

  const ridesNote =
    payload.rides_excluded === 0
      ? `${payload.rides_total} jízd v období · žádná vyloučená`
      : `${payload.rides_excluded} z ${payload.rides_total} jízd vyloučeno pro neúplná data`;

  const thresholdLabel =
    `nad ${payload.zone} · ${payload.threshold_bpm} tepů` +
    (payload.threshold_bpm === payload.requested_bpm
      ? ""
      : ` (z LTHR ${payload.lthr_bpm} → ${payload.requested_bpm}, nejbližší práh mřížky)`);

  // null = v období není ani jedna jízda. Nula by znamenala "jel jsem, ale
  // nad práh se nedostal", a to je jiné tvrzení – proto pomlčka.
  if (payload.longest_block_s == null) {
    return {
      ...empty,
      thresholdLabel,
      ridesNote,
      emptyNote:
        payload.rides_excluded > 0 && payload.rides_excluded === payload.rides_total
          ? `Všech ${payload.rides_total} jízd v období má neúplná data. Vypni filtr, ` +
            "čísla ale budou podhodnocená – blok ukončí pauza, ne fyziologie."
          : "V období není žádná jízda se spočítanými bloky.",
    };
  }

  const maxSeconds = Math.max(...payload.hist.map((h) => h.seconds), 1);
  const longest = payload.longest_block;
  const totalMin = payload.totals.total_time_s / 60;
  const longMin = payload.totals.time_in_long_blocks_s / 60;

  return {
    hasData: true,
    headline: fmtBlock(payload.longest_block_s),
    headlineNote:
      longest?.date != null
        ? `${shortDate(longest.date, false)} · ${longest.label ?? ""}`
        : "",
    thresholdLabel,
    trend:
      payload.trend_pct == null
        ? payload.previous_longest_block_s == null
          ? "předchozí období bez dat"
          : null
        : `${signed(payload.trend_pct, 0)} % · předtím ${fmtBlock(
            payload.previous_longest_block_s ?? 0,
          )}`,
    trendColor:
      payload.trend_pct == null
        ? theme.mut
        : payload.trend_pct >= 0
          ? theme.ok
          : theme.bad,
    hist: payload.hist.map((h) => ({
      bucket: h.bucket,
      count: `${h.count}×`,
      time: fmtMin(h.seconds / 60),
      // Šířka pruhu podle času, ne podle počtu: jinak by 199 vteřinových
      // úseků přebilo devět pětiminutových bloků, které jsou tréninkově
      // podstatnější.
      w: `${(mounted ? (h.seconds / maxSeconds) * 100 : 0).toFixed(1)}%`,
    })),
    totals: `${payload.totals.segment_count} úseků · ${fmtMin(totalMin)} nad prahem`,
    intentional: `${fmtMin(longMin)} v souvislých blocích`,
    ridesNote,
    emptyNote: null,
  };
}
