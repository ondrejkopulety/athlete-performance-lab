/**
 * Karta „Stoupání · VAM".
 *
 * VAM = převýšení za hodinu stoupání. Počítá se z celého období naráz
 * (součet metrů / součet minut do kopce), ne jako průměr jednotlivých VAM –
 * krátká prudká jízda by jinak přetlačila celotýdenní objem.
 */

import type { ActivityRow } from "../api";
import { dec, fmtShort } from "../format";
import type { Theme } from "../theme";

export interface Climb {
  hasData: boolean;
  vam: number;
  vamMax: number;
  vamColor: string;
  vamTrend: string;
  vamTrendColor: string;
  bars: { h: string; color: string }[];
  uphill: string;
  grad: string;
}

/** Jízdy, které vůbec mají stoupání – rovina by dělila nulou. */
function climbable(activities: ActivityRow[]): ActivityRow[] {
  return activities.filter((a) => (a.up ?? 0) > 0 && (a.asc ?? 0) > 0);
}

export function buildClimb(
  activities: ActivityRow[],
  allActivities: ActivityRow[],
  theme: Theme,
  mounted: boolean,
): Climb {
  const inRange = climbable(activities);
  // Když v období není žádné stoupání, ukáže se posledních 8 jízd z historie –
  // stejné chování jako v designu, aby karta nezůstala prázdná.
  const src = inRange.length ? inRange : climbable(allActivities).slice(-8);

  if (!src.length) {
    return {
      hasData: false,
      vam: 0,
      vamMax: 0,
      vamColor: theme.mut,
      vamTrend: "málo dat",
      vamTrendColor: theme.faint,
      bars: [],
      uphill: "0 min",
      grad: "0",
    };
  }

  const vamOf = (r: ActivityRow) => ((r.asc as number) / (r.up as number)) * 60;
  const upTotal = src.reduce((a, r) => a + (r.up as number), 0);
  const ascTotal = src.reduce((a, r) => a + (r.asc as number), 0);
  const avgVam = (ascTotal / upTotal) * 60;
  const grad = src.reduce((a, r) => a + (r.grad ?? 0) * (r.up as number), 0) / (upTotal || 1);

  const half = Math.max(1, Math.floor(src.length / 2));
  const early = src.slice(0, half);
  const late = src.slice(-half);
  const vamOfGroup = (g: ActivityRow[]) =>
    (g.reduce((a, r) => a + (r.asc as number), 0) / g.reduce((a, r) => a + (r.up as number), 0)) *
    60;
  const diff = vamOfGroup(late) - vamOfGroup(early);

  const bars = src.slice(-16);
  const max = Math.max(...bars.map(vamOf), 1);

  return {
    hasData: true,
    vam: Math.round(avgVam),
    vamMax: Math.round(max),
    vamColor: avgVam >= 800 ? theme.ok : avgVam >= 600 ? theme.warn : theme.orange,
    vamTrend: `${diff >= 0 ? "+" : "−"}${Math.abs(Math.round(diff))} m/h`,
    vamTrendColor: diff >= 0 ? theme.ok : theme.bad,
    bars: bars.map((r) => ({
      h: `${mounted ? (vamOf(r) / max) * 100 : 0}%`,
      color: vamOf(r) >= 800 ? theme.ok : vamOf(r) >= 600 ? theme.warn : theme.grey,
    })),
    uphill: fmtShort(upTotal) + (upTotal >= 60 ? " h" : ""),
    grad: dec(grad),
  };
}
