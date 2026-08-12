/** Karty posledních jízd včetně rozpadu na zóny. */

import type { Ride } from "../api";
import { czDate, dec, fmtMin, fmtShort, rideCount } from "../format";
import { ZONE_NAMES, type Theme } from "../theme";

export interface RideZone {
  w: string;
  color: string;
  name: string;
  mins: string;
}

export interface RideCard {
  id: string;
  date: string;
  distance: string;
  duration: string;
  avgHr: string;
  ascent: string;
  trimp: number;
  maxHr: string;
  kcal: string;
  tag: string;
  tagBg: string;
  tagColor: string;
  zones: RideZone[];
  /** Odznak se ukáže jen při neúplných datech – „100 % OK" štítek je šum. */
  partial: boolean;
  coverageNote: string | null;
}

export function buildRides(rides: Ride[], theme: Theme, light: boolean): RideCard[] {
  return rides.map((r) => {
    const total = r.z.reduce((a, b) => a + b, 0) || 1;
    const hard = (r.z[3] + r.z[4]) / total;

    return {
      id: r.id,
      date: czDate(r.d, { weekday: "short", day: "numeric", month: "short" }),
      distance: dec(r.km ?? 0),
      duration: fmtShort(r.dur ?? 0),
      avgHr: r.avg == null ? "–" : `${Math.round(r.avg)} tep`,
      ascent: `${Math.round(r.asc ?? 0)} m`,
      trimp: Math.round(r.trimp ?? 0),
      maxHr: r.max == null ? "–" : String(Math.round(r.max)),
      kcal: r.kcal == null ? "–" : String(Math.round(r.kcal)),
      tag: hard > 0.15 ? "Práh" : hard > 0.05 ? "Tempo" : "Vytrvalost",
      tagBg:
        (hard > 0.15 ? theme.orange2 : hard > 0.05 ? theme.ok : theme.blue) +
        (light ? "26" : "18"),
      tagColor: hard > 0.15 ? theme.orange : hard > 0.05 ? theme.ok : theme.blue,
      zones: r.z.map((m, i) => ({
        w: `${((m / total) * 100).toFixed(1)}%`,
        color: theme.zones[i],
        name: ZONE_NAMES[i],
        mins: fmtMin(m),
      })),
      // `cov === null` = pokrytí ještě spočítané není, což není totéž co
      // špatné – karta v tom případě odznak nedostane.
      partial: r.cov ? !r.cov.ok : false,
      coverageNote: r.cov?.note ?? null,
    };
  });
}

export function ridesSummary(rides: Ride[]): string {
  const km = rides.reduce((a, r) => a + (r.km ?? 0), 0);
  const min = rides.reduce((a, r) => a + (r.dur ?? 0), 0);
  return `${rideCount(rides.length)} · ${km.toFixed(0)} km · ${fmtMin(min)}`;
}
