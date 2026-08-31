/**
 * Graf bilance zátěže (CTL/ATL) a sloupce denní zátěže – 1:1 s Readiness
 * Dashboard.dc.html. Křivky jsou vyhlazené (Catmull-Rom → kubické Bézier),
 * podporuje srovnání s předchozím rokem a výběr úseku (brush).
 */

import type { Day } from "../api";
import { dec, mon, signed, tick } from "../format";
import type { Theme } from "../theme";

const W = 700;
const H = 196;
const PAD = 14;
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
  ctlLineCompare: string;
  gridLines: { y: string }[];
  yTicks: { top: string; label: number }[];
  selX: string;
  selCtlY: string;
  selAtlY: string;
  selPct: string;
  sel: { ctl: string; atl: string; tsb: string; acwr: string; ctlPrev: string; label: string };
  tsbColor: string;
  showCompareLegend: boolean;
  compareLegendYear: string;
  brushX: string;
  brushW: string;
  brushOn: string;
  bars: Bar[];
  barsLabel: string;
  barMax: number;
}

/** Vyhlazená čára (Catmull-Rom → kubické Bézier) jako v designu. */
function smoothLine(pts: [number, number][]): string {
  if (pts.length < 3) {
    return pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
  }
  let d = `M${pts[0][0].toFixed(1)} ${pts[0][1].toFixed(1)}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i === 0 ? i : i - 1];
    const p1 = pts[i];
    const p2 = pts[i + 1];
    const p3 = pts[i + 2 < pts.length ? i + 2 : i + 1];
    const c1x = p1[0] + (p2[0] - p0[0]) / 6;
    const c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6;
    const c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${p2[0].toFixed(1)} ${p2[1].toFixed(1)}`;
  }
  return d;
}

export function buildPmc(
  rows: Day[],
  startIdx: number,
  selIdxAbs: number,
  days: Day[],
  theme: Theme,
  compareRows: (Day | null)[] | null = null,
  compareYearLabel = "",
  brush: [number, number] | null = null,
): Pmc {
  if (rows.length === 0) {
    return emptyPmc(theme);
  }

  const selRow = days[selIdxAbs] ?? rows[rows.length - 1];
  const compareValid = compareRows && compareRows.every((r) => r != null) ? (compareRows as Day[]) : null;
  const values = rows
    .flatMap((r) => [r.ctl, r.atl])
    .concat(compareValid ? compareValid.map((r) => r.ctl) : []);
  const lo = Math.min(...values) - 8;
  const hi = Math.max(...values) + 8;

  const x = (i: number) => (rows.length === 1 ? W / 2 : (i / (rows.length - 1)) * W);
  const y = (v: number) => PAD + (1 - (v - lo) / (hi - lo)) * (H - PAD * 2);
  const line = (data: Day[], key: "ctl" | "atl") =>
    smoothLine(data.map((r, i) => [x(i), y(r[key])]));
  const area = (data: Day[], key: "ctl" | "atl") => `${line(data, key)} L${W} ${H} L0 ${H} Z`;

  const selIdx = selIdxAbs - startIdx;

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

  const brushLo = brush ? Math.max(0, Math.min(rows.length - 1, brush[0])) : 0;
  const brushHi = brush ? Math.max(0, Math.min(rows.length - 1, brush[1])) : 0;

  return {
    empty: false,
    ctlLine: line(rows, "ctl"),
    atlLine: line(rows, "atl"),
    ctlArea: area(rows, "ctl"),
    ctlLineCompare: compareValid ? line(compareValid, "ctl") : "",
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
      ctlPrev:
        compareValid && compareValid[selIdx] ? dec(compareValid[selIdx].ctl) : "–",
      label: new Date(`${selRow.d}T12:00:00`)
        .toLocaleDateString("cs-CZ", { day: "numeric", month: "short" })
        .toUpperCase(),
    },
    tsbColor: selRow.tsb > 5 ? theme.ok : selRow.tsb < -20 ? theme.bad : theme.fg2,
    showCompareLegend: !!compareValid,
    compareLegendYear: compareYearLabel,
    brushX: x(brushLo).toFixed(1),
    brushW: Math.max(1, x(brushHi) - x(brushLo)).toFixed(1),
    brushOn: brush ? "1" : "0",
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
    ctlLineCompare: "",
    gridLines: [],
    yTicks: [],
    selX: "-10",
    selCtlY: "-10",
    selAtlY: "-10",
    selPct: "50%",
    sel: { ctl: "–", atl: "–", tsb: "–", acwr: "–", ctlPrev: "–", label: "" },
    tsbColor: theme.fg2,
    showCompareLegend: false,
    compareLegendYear: "",
    brushX: "-10",
    brushW: "0",
    brushOn: "0",
    bars: [],
    barsLabel: "Denní zátěž · TRIMP",
    barMax: 0,
  };
}
