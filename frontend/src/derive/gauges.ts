/**
 * Čtyři půlkruhové ukazatele pod hlavním kolečkem.
 *
 * Geometrie (poloměr, 270° oblouk, dash pole) je opsaná z designu.
 * Co se změnilo: hodnoty i pásma nejsou natvrdo. Design měl HRV 53 s ideálem
 * 62–78 a tep 51 s ideálem 42–48 – z čísel je vidět, že spodní mez pásma je
 * vlastní baseline sportovce (týdenní průměr HRV, 14denní bazál tepu), takže
 * se tady počítají z něj a posouvají se s formou.
 */

import type { Biometric, Today } from "../api";
import { dec, fmtShort } from "../format";
import type { Theme } from "../theme";

export interface Gauge {
  name: string;
  value: string;
  delta: string;
  ideal: string;
  color: string;
  nameColor: string;
  deltaColor: string;
  bandColor: string;
  trackDash: string;
  bandDash: string;
  valDash: string;
  bx1: string;
  by1: string;
  bx2: string;
  by2: string;
}

const R = 26;
const C = 2 * Math.PI * R;
const ARC = 0.75 * C;

/** Horní mez HRV pásma = baseline + 26 %; přesně tenhle poměr měl design. */
const HRV_BAND_TOP = 1.26;
/** Spodní mez tepového pásma = bazál − 6 tepů. */
const RHR_BAND_SPAN = 6;
/** Obecné doporučení 7:30–9:00; Garminova „potřeba spánku" se používá zvlášť
 *  v popisku, protože bývá výrazně velkorysejší. */
const SLEEP_BAND: [number, number] = [450, 540];

interface Spec {
  name: string;
  value: number | null;
  min: number;
  max: number;
  band: [number, number] | null;
  okColor: string;
  display: string;
  delta: string;
  deltaColor: string;
  ideal: string;
  /** Pod pásmem jen varování, ne poplach (nízký strain není chyba). */
  softLow?: boolean;
}

function build(spec: Spec, theme: Theme, mounted: boolean): Gauge {
  const { value, min, max } = spec;
  const band = spec.band ?? [min, max];
  const clamp = (v: number) => Math.max(0, Math.min(1, (v - min) / (max - min)));
  const f = value === null ? 0 : clamp(value);
  const b0 = clamp(band[0]);
  const b1 = clamp(band[1]);

  const inBand = value !== null && value >= band[0] && value <= band[1];
  const color =
    value === null
      ? theme.grey
      : inBand
        ? spec.okColor
        : spec.softLow && value < band[0]
          ? theme.warn
          : theme.bad;

  const angle = (a: number) => ((135 + a * 270) * Math.PI) / 180;
  const at = (a: number, r: number): [number, number] => [
    34 + Math.cos(angle(a)) * r,
    34 + Math.sin(angle(a)) * r,
  ];
  const [bx1, by1] = at(b0, 21.5);
  const [bx2, by2] = at(b0, 30.5);

  return {
    name: spec.name,
    value: spec.display,
    delta: spec.delta,
    ideal: spec.ideal,
    color,
    nameColor: value === null || inBand ? theme.mut : color,
    deltaColor: inBand ? spec.deltaColor : color,
    bandColor: `${spec.okColor}3d`,
    trackDash: `${ARC.toFixed(1)} ${C.toFixed(1)}`,
    bandDash: `0 ${(b0 * ARC).toFixed(1)} ${((b1 - b0) * ARC).toFixed(1)} ${C.toFixed(1)}`,
    valDash: `${(mounted ? f * ARC : 0).toFixed(1)} ${C.toFixed(1)}`,
    bx1: bx1.toFixed(1),
    by1: by1.toFixed(1),
    bx2: bx2.toFixed(1),
    by2: by2.toFixed(1),
  };
}

/** Hodnota z dneška, jinak poslední naměřená – s datem do popisku. */
function pick(
  todayValue: number | null | undefined,
  todayRef: number | null | undefined,
  fallback: Biometric | null,
  todayDate: string | undefined,
): { value: number | null; reference: number | null; stale: string | null } {
  if (todayValue != null) {
    return { value: todayValue, reference: todayRef ?? null, stale: null };
  }
  if (fallback?.value != null) {
    const p = fallback.date.split("-");
    return {
      value: fallback.value,
      reference: fallback.reference,
      stale: fallback.date === todayDate ? null : `${Number(p[2])}. ${Number(p[1])}.`,
    };
  }
  return { value: null, reference: null, stale: null };
}

export function buildGauges(
  theme: Theme,
  today: Today | null,
  lastKnown: Record<"hrv" | "rhr" | "sleep", Biometric | null>,
  mounted: boolean,
): Gauge[] {
  const hrv = pick(today?.hrv_last_night, today?.hrv_weekly_avg, lastKnown.hrv, today?.date);
  const rhr = pick(today?.rhr_day, today?.rhr_baseline_14d, lastKnown.rhr, today?.date);
  const sleep = pick(
    today?.sleep_duration_min,
    today?.sleep_need_min,
    lastKnown.sleep,
    today?.date,
  );
  const strain = today?.whoop_strain ?? null;

  const stale = (s: string | null) => (s ? ` · ${s}` : "");

  const hrvBand: [number, number] | null =
    hrv.reference != null ? [hrv.reference, hrv.reference * HRV_BAND_TOP] : null;
  const hrvDelta =
    hrv.value != null && hrv.reference
      ? `${signedPct(hrv.value / hrv.reference - 1)} vs ${Math.round(hrv.reference)}${stale(hrv.stale)}`
      : hrv.value != null
        ? `bez týdenního průměru${stale(hrv.stale)}`
        : "chybí měření";

  const rhrBand: [number, number] | null =
    rhr.reference != null ? [rhr.reference - RHR_BAND_SPAN, rhr.reference] : null;
  const rhrDelta =
    rhr.value != null && rhr.reference != null
      ? `${signedInt(rhr.value - rhr.reference)} vs bazál ${Math.round(rhr.reference)}${stale(rhr.stale)}`
      : rhr.value != null
        ? `bez bazálu${stale(rhr.stale)}`
        : "chybí měření";

  const sleepDelta =
    sleep.value != null && sleep.reference != null
      ? sleep.value >= sleep.reference
        ? `potřeba splněna${stale(sleep.stale)}`
        : `−${fmtShort(sleep.reference - sleep.value)} do potřeby${stale(sleep.stale)}`
      : sleep.value != null
        ? `bez potřeby spánku${stale(sleep.stale)}`
        : "chybí měření";

  return [
    build(
      {
        name: "HRV",
        value: hrv.value,
        min: 25,
        max: 90,
        band: hrvBand,
        okColor: theme.ok,
        display: hrv.value == null ? "–" : String(Math.round(hrv.value)),
        delta: hrvDelta,
        deltaColor: theme.bad,
        ideal: hrvBand ? `${Math.round(hrvBand[0])}–${Math.round(hrvBand[1])} ms` : "–",
      },
      theme,
      mounted,
    ),
    build(
      {
        name: "Klid. tep",
        value: rhr.value,
        min: 38,
        max: 70,
        band: rhrBand,
        okColor: theme.blue,
        display: rhr.value == null ? "–" : String(Math.round(rhr.value)),
        delta: rhrDelta,
        deltaColor: theme.bad,
        ideal: rhrBand ? `${Math.round(rhrBand[0])}–${Math.round(rhrBand[1])} tep` : "–",
      },
      theme,
      mounted,
    ),
    build(
      {
        name: "Spánek",
        value: sleep.value,
        min: 240,
        max: 600,
        band: SLEEP_BAND,
        okColor: theme.blue2,
        display: sleep.value == null ? "–" : fmtShort(sleep.value),
        delta: sleepDelta,
        deltaColor: theme.warn,
        ideal: `${fmtShort(SLEEP_BAND[0])}–${fmtShort(SLEEP_BAND[1])}`,
      },
      theme,
      mounted,
    ),
    build(
      {
        name: "Strain",
        value: strain,
        min: 0,
        max: 21,
        band: [8, 14],
        okColor: theme.orange,
        display: strain == null ? "–" : dec(strain),
        delta: "za 24 h",
        deltaColor: theme.faint,
        ideal: "8–14",
        softLow: true,
      },
      theme,
      mounted,
    ),
  ];
}

function signedPct(ratio: number): string {
  const pct = Math.round(ratio * 100);
  return `${pct > 0 ? "+" : pct < 0 ? "−" : ""}${Math.abs(pct)} %`;
}

function signedInt(diff: number): string {
  const v = Math.round(diff);
  return `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v)}`;
}
