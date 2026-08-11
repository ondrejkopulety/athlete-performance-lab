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
