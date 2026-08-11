/**
 * Graf bilance zátěže (CTL/ATL) a sloupce denní zátěže.
 * Geometrie, prahy barev i logika slučování do týdnů/měsíců jsou z designu.
 */

import type { Day } from "../api";
import { dec, mon, signed, tick } from "../format";
import type { Theme } from "../theme";

const W = 700;
const H = 196;
const PAD = 14;
/** Výška SVG viewBoxu; popisky osy Y se do ní přepočítávají na procenta. */
const VIEW_H = 220;

export interface Bar {
  tick: string;
  tickColor: string;
  barH: string;
  barColor: string;
  barOpacity: string;
  dayIndex: number;
}

export interface Pmc {
  empty: boolean;
  ctlLine: string;
  atlLine: string;
  ctlArea: string;
  atlArea: string;
  gridLines: { y: string }[];
  yTicks: { top: string; label: number }[];
  selX: string;
  selCtlY: string;
  selAtlY: string;
  selPct: string;
  sel: { ctl: string; atl: string; tsb: string; acwr: string; label: string };
  tsbColor: string;
  bars: Bar[];
  barsLabel: string;
  barMax: number;
}

export function buildPmc(
  rows: Day[],
  startIdx: number,
  selIdxAbs: number,
  days: Day[],
  theme: Theme,
): Pmc {
  if (rows.length === 0) {
    return emptyPmc(theme);
  }

  const selRow = days[selIdxAbs] ?? rows[rows.length - 1];
  const values = rows.flatMap((r) => [r.ctl, r.atl]);
  const lo = Math.min(...values) - 8;
  const hi = Math.max(...values) + 8;

  const x = (i: number) => (rows.length === 1 ? W / 2 : (i / (rows.length - 1)) * W);
  const y = (v: number) => PAD + (1 - (v - lo) / (hi - lo)) * (H - PAD * 2);
  const line = (key: "ctl" | "atl") =>
    rows.map((r, i) => `${i ? "L" : "M"}${x(i).toFixed(1)} ${y(r[key]).toFixed(1)}`).join(" ");
  const area = (key: "ctl" | "atl") => `${line(key)} L${W} ${H} L0 ${H} Z`;

  const selIdx = selIdxAbs - startIdx;

  // Sloupce: do 40 dní po dnech, do ~400 po týdnech, dál po čtyřech týdnech.
  const bucketDays = rows.length > 400 ? 28 : rows.length > 40 ? 7 : 1;
  const buckets: { trimp: number; first: Day; last: Day; has: boolean }[] = [];
  for (let i = 0; i < rows.length; i += bucketDays) {
    const chunk = rows.slice(i, i + bucketDays);
    buckets.push({
      trimp: chunk.reduce((a, r) => a + r.trimp, 0),
      first: chunk[0],
      last: chunk[chunk.length - 1],
      has: chunk.some((r) => r.d === selRow.d),
    });
  }
  const maxT = Math.max(...buckets.map((b) => b.trimp), 1);
  const multiYear = rows[0].d.slice(0, 4) !== rows[rows.length - 1].d.slice(0, 4);
  const tickEvery = Math.max(1, Math.ceil(buckets.length / 8));

  const bars: Bar[] = buckets.map((b, i) => ({
    tick: i % tickEvery === 0 ? (bucketDays > 1 ? mon(b.first.d, multiYear) : tick(b.first.d)) : "",
    tickColor: b.has ? theme.fg2 : theme.mut,
    barH: `${(b.trimp / maxT) * 100}%`,
    barColor:
      b.trimp === 0
        ? theme.zero
        : b.trimp / bucketDays >= 220
          ? theme.orange2
          : b.trimp / bucketDays >= 90
            ? theme.orange
            : theme.ok,
    barOpacity: b.has ? "1" : "0.5",
    dayIndex: days.findIndex((z) => z.d === b.last.d),
  }));

  return {
    empty: false,
    ctlLine: line("ctl"),
    atlLine: line("atl"),
    ctlArea: area("ctl"),
    atlArea: area("atl"),
    gridLines: [0.25, 0.5, 0.75].map((f) => ({ y: (PAD + f * (H - PAD * 2)).toFixed(1) })),
    yTicks: [0.25, 0.5, 0.75].map((fr) => ({
      top: `${(((PAD + fr * (H - PAD * 2)) / VIEW_H) * 100).toFixed(1)}%`,
      label: Math.round(lo + (1 - fr) * (hi - lo)),
    })),
    selX: x(selIdx).toFixed(1),
    selCtlY: y(selRow.ctl).toFixed(1),
    selAtlY: y(selRow.atl).toFixed(1),
    selPct: `${(rows.length < 2
      ? 50
      : Math.min(94, Math.max(6, (selIdx / (rows.length - 1)) * 100))
    ).toFixed(2)}%`,
    sel: {
      ctl: dec(selRow.ctl),
      atl: dec(selRow.atl),
      tsb: signed(selRow.tsb),
      acwr: selRow.acwr == null ? "–" : dec(selRow.acwr, 2),
      label: new Date(`${selRow.d}T12:00:00`)
        .toLocaleDateString("cs-CZ", { day: "numeric", month: "short" })
        .toUpperCase(),
    },
    tsbColor: selRow.tsb > 5 ? theme.ok : selRow.tsb < -20 ? theme.bad : theme.fg2,
    bars,
    barsLabel:
      bucketDays > 7
        ? "Měsíční zátěž · TRIMP"
        : bucketDays > 1
          ? "Týdenní zátěž · TRIMP"
          : "Denní zátěž · TRIMP",
    barMax: Math.round(maxT),
  };
}

function emptyPmc(theme: Theme): Pmc {
  return {
    empty: true,
    ctlLine: "",
    atlLine: "",
    ctlArea: "",
    atlArea: "",
    gridLines: [],
    yTicks: [],
    selX: "-10",
    selCtlY: "-10",
    selAtlY: "-10",
    selPct: "50%",
    sel: { ctl: "–", atl: "–", tsb: "–", acwr: "–", label: "" },
    tsbColor: theme.fg2,
    bars: [],
    barsLabel: "Denní zátěž · TRIMP",
    barMax: 0,
  };
}
