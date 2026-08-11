/**
 * Přepínač období. V designu byl rok 2026 zadrátovaný napevno; tady se
 * odvozuje z posledního dne v datech, aby dashboard fungoval i příští rok.
 */

import type { Day } from "../api";

export type Range = number | "ytd" | "all" | string; // string = "2024"

export interface RangeOption {
  label: string;
  value: Range;
}

export interface Window {
  rows: Day[];
  startIdx: number;
  label: string;
}

const YEAR_RE = /^\d{4}$/;

export function rangeOptions(days: Day[]): RangeOption[] {
  const currentYear = currentYearOf(days);
  const years = [...new Set(days.map((r) => r.d.slice(0, 4)))]
    .filter((y) => y !== currentYear)
    .sort()
    .reverse();

  return [
    { label: "7D", value: 7 },
    { label: "14D", value: 14 },
    { label: "1M", value: 30 },
    { label: "3M", value: 90 },
    { label: "6M", value: 180 },
    { label: "Letos", value: "ytd" },
    ...years.map((y) => ({ label: y, value: y })),
    { label: "Vše", value: "all" },
  ];
}

/** Poslední den v datech je "dnešek" – kalendář v pipeline vždy končí dneškem. */
export function currentYearOf(days: Day[]): string {
  return days.length ? days[days.length - 1].d.slice(0, 4) : String(new Date().getFullYear());
}

export function selectWindow(days: Day[], range: Range): Window {
  const year = typeof range === "string" && YEAR_RE.test(range) ? range : null;
  const currentYear = currentYearOf(days);

  let startIdx: number;
  if (range === "all") startIdx = 0;
  else if (range === "ytd") startIdx = indexFrom(days, `${currentYear}-01-01`);
  else if (year) startIdx = indexFrom(days, `${year}-01-01`);
  else startIdx = Math.max(0, days.length - (range as number));

  const endIdx = year ? days.findIndex((r) => r.d > `${year}-12-31`) : -1;
  const rows = endIdx > 0 ? days.slice(startIdx, endIdx) : days.slice(startIdx);

  return { rows, startIdx, label: rangeLabel(range, year) };
}

function indexFrom(days: Day[], iso: string): number {
  const i = days.findIndex((r) => r.d >= iso);
  return i === -1 ? days.length : i;
}

function rangeLabel(range: Range, year: string | null): string {
  if (range === "all") return "celá historie";
  if (range === "ytd") return "od 1. ledna";
  if (year) return `rok ${year}`;
  switch (range) {
    case 7:
      return "posledních 7 dní";
    case 14:
      return "posledních 14 dní";
    case 30:
      return "posledních 30 dní";
    case 90:
      return "poslední 3 měsíce";
    default:
      return "posledních 6 měsíců";
  }
}
