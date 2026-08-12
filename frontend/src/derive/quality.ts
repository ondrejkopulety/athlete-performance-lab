/**
 * Sekce „Kvalita tréninku": polarizace, junk miles, čas v zónách.
 *
 * Polarizace se počítá z minut v zónách za zvolené období – ne z denní
 * metriky `polarization_low_pct`, která je klouzavá přes 14 dní a neseděla by
 * na období vybrané přepínačem.
 */

import type { ActivityRow } from "../api";
import { fmtMin, rideCount } from "../format";
import { ZONE_SHORT, type Theme } from "../theme";

export interface Polarization {
  low: number;
  junk: number;
  high: number;
  lowW: string;
  junkW: string;
  highW: string;
  polColor: string;
  junkColor: string;
  junkW30: string;
  hasData: boolean;
  /** Nezáměrná Z3 v procentech času ve všech zónách; null = bloky chybí. */
  unintended: number | null;
  unintendedW30: string;
  unintendedColor: string;
  unintendedTime: string;
  unintendedShare: string;
}

export interface ZoneRow {
  name: string;
  color: string;
  w: string;
  time: string;
  pct: string;
}

export interface ZoneTime {
  total: string;
  rides: string;
  rows: ZoneRow[];
}

export function inRange(
  activities: ActivityRow[],
  from: string,
  to: string,
): ActivityRow[] {
  return activities.filter((a) => a.d >= from && a.d <= to);
}

function zoneSums(activities: ActivityRow[]): number[] {
  const sum = [0, 0, 0, 0, 0];
  activities.forEach((a) => a.z.forEach((v, i) => (sum[i] += v || 0)));
  return sum;
}

export function buildPolarization(
  activities: ActivityRow[],
  theme: Theme,
  mounted: boolean,
): Polarization {
  const sum = zoneSums(activities);
  const total = sum.reduce((a, b) => a + b, 0);
  const pct = (v: number) => (total ? (v / total) * 100 : 0);

  const low = pct(sum[0] + sum[1]);
  const junk = pct(sum[2]);
  const high = pct(sum[3] + sum[4]);

  // Nezáměrná Z3: čas v Z3 v úsecích kratších než tři minuty. Server ho
  // počítá z předpočítaných bloků (čas nad prahem mínus čas v dlouhých
  // blocích), tady se jen sčítá přes období.
  //
  // Aktivity bez spočítaných bloků se nezapočítají ani do jmenovatele –
  // jinak by metrika klesala tím, že se něco nespočítalo.
  const withBlocks = activities.filter((a) => a.z3u != null);
  const unintendedMin = withBlocks.reduce((a, r) => a + (r.z3u as number) / 60, 0);
  const blockTotal = withBlocks.reduce(
    (a, r) => a + r.z.reduce((x, y) => x + (y || 0), 0),
    0,
  );
  const unintended = blockTotal > 0 ? (unintendedMin / blockTotal) * 100 : null;
  const z3Min = withBlocks.reduce((a, r) => a + (r.z[2] || 0), 0);

  return {
    low,
    junk,
    high,
    lowW: mounted ? `${low.toFixed(1)}%` : "0%",
    junkW: mounted ? `${junk.toFixed(1)}%` : "0%",
    highW: mounted ? `${high.toFixed(1)}%` : "0%",
    polColor: low >= 75 ? theme.ok : low >= 65 ? theme.warn : theme.bad,
    junkColor: junk <= 15 ? theme.ok : junk <= 22 ? theme.warn : theme.bad,
    junkW30: mounted ? `${Math.min(100, (junk / 30) * 100).toFixed(1)}%` : "0%",
    hasData: total > 0,
    unintended,
    unintendedW30:
      mounted && unintended != null
        ? `${Math.min(100, (unintended / 30) * 100).toFixed(1)}%`
        : "0%",
    unintendedColor:
      unintended == null
        ? theme.mut
        : unintended <= 8
          ? theme.ok
          : unintended <= 15
            ? theme.warn
            : theme.bad,
    unintendedTime: unintended == null ? "–" : fmtMin(unintendedMin),
    unintendedShare:
      unintended == null || z3Min <= 0
        ? "bloky zatím nespočítané"
        : `z ${fmtMin(z3Min)} v Z3 celkem`,
  };
}

export function buildZoneTime(
  activities: ActivityRow[],
  theme: Theme,
  mounted: boolean,
): ZoneTime {
  const sum = zoneSums(activities);
  const total = sum.reduce((a, b) => a + b, 0);
  const max = Math.max(...sum, 1);

  return {
    total: fmtMin(total),
    rides: rideCount(activities.length),
    rows: sum.map((v, i) => ({
      name: ZONE_SHORT[i],
      color: theme.zones[i],
      w: `${(mounted ? (v / max) * 100 : 0).toFixed(1)}%`,
      time: fmtMin(v),
      pct: total ? ((v / total) * 100).toFixed(1).replace(".", ",") : "0",
    })),
  };
}
