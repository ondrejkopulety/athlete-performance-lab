/**
 * Panel „Tepová křivka" – maximální průměrný tep za okno 5 s až 60 min.
 *
 * Osa x je logaritmická: mezi 5 a 30 sekundami se toho děje stejně jako mezi
 * 20 a 60 minutami, a na lineární ose by první polovina křivky byla slepená
 * u nuly.
 *
 * **Chybějící okno není nula.** Bod s `hr === null` se nekreslí a čára se na
 * něm přeruší – jízda, která nemá hodinové okno, neznamená hodinu na nule.
 * Kdyby se null nahradilo nulou, křivka by na konci spadla na osu a tvrdila
 * něco, co se nikdy nenaměřilo.
 */

import type { CurvePeriod, CurvePoint, HrCurvePayload } from "../api";
import { shortDate } from "../format";
import type { Theme } from "../theme";

// Souřadnice grafu. SVG se roztahuje přes celou šířku karty
// (preserveAspectRatio="none"), takže se do něj kreslí **jen čáry** – text i
// body jsou HTML umístěné v procentech. Písmo ani kolečka se tím
// nedeformují, což je stejný postup jako u ostatních grafů dashboardu.
const W = 520;
const H = 190;
const PAD_T = 12;
const PAD_B = 22;

export interface CurveRow {
  duration: string;
  value: string;
  source: string;
}

export interface CurveDot {
  left: string;
  top: string;
  hr: number;
}

export interface CurveSeries {
  /** Souvislé úseky čáry – přerušení tam, kde okno v období nevyšlo. */
  paths: string[];
  dots: CurveDot[];
}

export interface HrCurveView {
  hasData: boolean;
  width: number;
  height: number;
  period: CurveSeries;
  reference: CurveSeries | null;
  referenceLabel: string;
  xTicks: { left: string; label: string }[];
  yTicks: { top: string; label: string }[];
  rows: CurveRow[];
  ridesNote: string;
  /** Prázdno po filtru – ať se nekreslí prázdný graf bez vysvětlení. */
  emptyNote: string | null;
  lastEffort: { text: string; stale: boolean } | null;
}

/** Délka okna jako text: 5 s, 2 min, 1 h. */
export function fmtWindow(seconds: number): string {
  if (seconds < 60) return `${seconds} s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  const h = seconds / 3600;
  return h % 1 === 0 ? `${h} h` : `${h.toFixed(1).replace(".", ",")} h`;
}

export function buildHrCurve(
  payload: HrCurvePayload | null,
  theme: Theme,
): HrCurveView {
  const empty: HrCurveView = {
    hasData: false,
    width: W,
    height: H,
    period: { paths: [], dots: [] },
    reference: null,
    referenceLabel: "",
    xTicks: [],
    yTicks: [],
    rows: [],
    ridesNote: "",
    emptyNote: null,
    lastEffort: null,
  };
  if (!payload) return empty;

  const { period, reference } = payload;
  const values = [...period.points, ...(reference?.points ?? [])]
    .map((p) => p.hr)
    .filter((v): v is number => v != null);

  const ridesNote = excludedNote(period);

  if (values.length === 0) {
    return {
      ...empty,
      ridesNote,
      emptyNote:
        period.rides_excluded > 0 && period.rides_excluded === period.rides_total
          ? `Všech ${period.rides_total} jízd v období má neúplná data. Vypni filtr, ` +
            "čísla ale budou podhodnocená."
          : "V období není žádná jízda s tepovou křivkou.",
    };
  }

  const durations = payload.durations_s;
  const minD = Math.min(...durations);
  const maxD = Math.max(...durations);
  const lo = Math.floor((Math.min(...values) - 4) / 5) * 5;
  const hi = Math.ceil((Math.max(...values) + 4) / 5) * 5;

  // Osa x logaritmicky: mezi 5 a 30 sekundami se toho děje stejně jako mezi
  // 20 a 60 minutami.
  const fx = (d: number) =>
    (Math.log(d) - Math.log(minD)) / (Math.log(maxD) - Math.log(minD));
  const fy = (v: number) => 1 - (v - lo) / (hi - lo);

  const x = (d: number) => fx(d) * W;
  const y = (v: number) => PAD_T + fy(v) * (H - PAD_T - PAD_B);

  const series = (points: CurvePoint[]): CurveSeries => {
    const paths: string[] = [];
    const dots: CurveDot[] = [];
    let run: string[] = [];

    for (const p of [...points].sort((a, b) => a.d - b.d)) {
      if (p.hr == null) {
        // Díra v křivce ukončí úsek čáry; další bod začne nový "M".
        if (run.length > 1) paths.push(run.join(" "));
        run = [];
        continue;
      }
      run.push(`${run.length ? "L" : "M"}${x(p.d).toFixed(1)} ${y(p.hr).toFixed(1)}`);
      dots.push({
        left: `${(fx(p.d) * 100).toFixed(2)}%`,
        top: `${((y(p.hr) / H) * 100).toFixed(2)}%`,
        hr: p.hr,
      });
    }
    if (run.length > 1) paths.push(run.join(" "));
    return { paths, dots };
  };

  const spanYears =
    (period.from_date ?? "").slice(0, 4) !== (period.to_date ?? "").slice(0, 4);

  return {
    hasData: true,
    width: W,
    height: H,
    period: series(period.points),
    reference: reference ? series(reference.points) : null,
    referenceLabel: reference?.label ?? "",
    xTicks: durations
      .filter((d) => [5, 30, 120, 300, 1200, 3600].includes(d))
      .map((d) => ({ left: `${(fx(d) * 100).toFixed(2)}%`, label: fmtWindow(d) })),
    yTicks: [hi, Math.round((lo + hi) / 2), lo].map((v) => ({
      top: `${((y(v) / H) * 100).toFixed(2)}%`,
      label: String(v),
    })),
    rows: period.points.map((p) => ({
      duration: fmtWindow(p.d),
      // Pomlčka, ne nula: okno v období prostě není.
      value: p.hr == null ? "–" : String(Math.round(p.hr)),
      source:
        p.hr == null || !p.date
          ? ""
          : `${shortDate(p.date, spanYears)} · ${p.label ?? ""}`,
    })),
    ridesNote,
    emptyNote: null,
    lastEffort: lastEffortText(period, theme),
  };
}

function excludedNote(period: CurvePeriod): string {
  if (period.rides_excluded === 0) {
    return `${period.rides_total} jízd v období · žádná vyloučená`;
  }
  return `${period.rides_excluded} z ${period.rides_total} jízd vyloučeno pro neúplná data`;
}

function lastEffortText(
  period: CurvePeriod,
  _theme: Theme,
): { text: string; stale: boolean } | null {
  const e = period.last_max_effort;
  if (!e) {
    return {
      text: `žádné ${fmtWindow(1200)} maximum v období`,
      stale: true,
    };
  }
  const days = e.days_ago;
  const when =
    days === 0 ? "dnes" : days === 1 ? "včera" : `před ${days} dny`;
  return { text: `${when} · ${e.label ?? ""}`.trim(), stale: e.stale };
}
