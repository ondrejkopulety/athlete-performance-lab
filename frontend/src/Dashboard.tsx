import { useMemo, useRef, useState } from "react";

import { toDays, type DailyRow, type DashboardPayload, type Day } from "./api";
import { DailyStatus } from "./components/DailyStatus";
import { Header } from "./components/Header";
import { LoadBalance, type CompareOption } from "./components/LoadBalance";
import { Shell } from "./components/Shell";
import { StateScreen } from "./components/StateScreen";
import { buildGauges } from "./derive/gauges";
import { buildPmc } from "./derive/pmc";
import { rangeOptions, selectWindow, type Range } from "./derive/ranges";
import { czDate, tick } from "./format";
import { THEMES, type ThemeName } from "./theme";

type Zoom = { from: string; to: string };

/**
 * Přehled – 1:1 s Readiness Dashboard.dc.html: pozdrav + kolečko
 * připravenosti se čtyřmi ukazateli a denním stepperem + „Bilance zátěže"
 * (CTL/ATL, srovnání s předchozím rokem, výběr úseku).
 *
 * „Čistý pohled" – všechno z propů, žádný fetch uvnitř (SSR v render-check).
 */
export function Dashboard({
  payload,
  dailySeries,
  theme,
  mounted,
  hidden,
  onOpenMetric,
}: {
  payload: DashboardPayload;
  dailySeries: DailyRow[];
  theme: ThemeName;
  mounted: boolean;
  hidden?: boolean;
  onOpenMetric: (path: string) => void;
}) {
  const T = THEMES[theme];
  const days = useMemo(() => toDays(payload.days), [payload]);

  const [range, setRange] = useState<Range>(30);
  const [zoom, setZoom] = useState<Zoom | null>(null);
  const [sel, setSel] = useState<number>(days.length - 1);
  const [scrubbing, setScrubbing] = useState(false);
  const [dayIdx, setDayIdx] = useState<number>(-1);
  const [compareYear, setCompareYear] = useState<number | null>(null);
  const [brushMode, setBrushMode] = useState(false);
  const [brushRange, setBrushRange] = useState<[number, number] | null>(null);
  const prevRangeRef = useRef<Range>(30);

  const ranges = useMemo(() => rangeOptions(days), [days]);

  // Okno: buď „zoom" z brushe (od–do datum), nebo standardní období.
  const window = useMemo(() => {
    if (zoom) {
      let s = days.findIndex((r) => r.d >= zoom.from);
      let e = days.findIndex((r) => r.d > zoom.to);
      if (s < 0) s = 0;
      if (e < 0) e = days.length;
      return { rows: days.slice(s, e), startIdx: s, label: `${tick(zoom.from)}–${tick(zoom.to)}` };
    }
    return selectWindow(days, range);
  }, [days, range, zoom]);
  const { rows, startIdx } = window;

  const selIdxAbs = Math.min(
    Math.max(sel, startIdx),
    startIdx + Math.max(0, rows.length - 1),
  );

  const winRef = useRef({ start: startIdx, len: rows.length });
  winRef.current = { start: startIdx, len: rows.length };

  // Srovnání s předchozím rokem: nabídni jen roky, které mají celé posunuté okno.
  const isYear = typeof range === "string" && /^\d{4}$/.test(range);
  const compareEligible = !zoom && !isYear && range !== "all" && rows.length > 0;
  const shiftRows = (off: number): (Day | null)[] =>
    rows.map((r) => {
      const dt = new Date(`${r.d}T00:00:00Z`);
      dt.setUTCFullYear(dt.getUTCFullYear() - off);
      return days.find((z) => z.d === dt.toISOString().slice(0, 10)) ?? null;
    });
  const compareOptions: CompareOption[] = useMemo(() => {
    if (!compareEligible) return [];
    const endYear = Number(rows[rows.length - 1].d.slice(0, 4));
    const minYear = Number(days[0].d.slice(0, 4));
    const opts: CompareOption[] = [];
    for (let off = 1; off <= endYear - minYear; off++) {
      const shifted = shiftRows(off);
      if (shifted.every((r) => r != null)) opts.push({ off, label: String(endYear - off) });
    }
    return opts;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [compareEligible, rows, days]);

  const compareValid = compareYear != null && compareOptions.some((o) => o.off === compareYear);
  const compareRows = compareValid ? shiftRows(compareYear as number) : null;
  const compareYearLabel = compareValid
    ? String(compareOptions.find((o) => o.off === compareYear)?.label ?? "")
    : "";

  const pmc = useMemo(
    () =>
      buildPmc(
        rows,
        startIdx,
        selIdxAbs,
        days,
        T,
        compareRows,
        compareYearLabel,
        brushRange,
      ),
    [rows, startIdx, selIdxAbs, days, T, compareRows, compareYearLabel, brushRange],
  );

  // Vybraný den pro kolečko + ukazatele.
  const lastRowIdx = dailySeries.length - 1;
  const effIdx = dayIdx < 0 || dayIdx > lastRowIdx ? lastRowIdx : dayIdx;
  const isLastDay = effIdx === lastRowIdx || lastRowIdx < 0;
  const selDay: DailyRow | null = dailySeries[effIdx] ?? null;

  const readiness =
    (isLastDay ? payload.today?.readiness_score : selDay?.readiness_score) ??
    selDay?.readiness_score ??
    null;
  const readyColor =
    readiness == null ? T.mut : readiness >= 66 ? T.ok : readiness >= 40 ? T.warn : T.bad;

  const gaugeSource = isLastDay ? (payload.today ?? selDay) : selDay;
  const gauges = useMemo(
    () => buildGauges(T, gaugeSource, payload.last_known, mounted),
    [T, gaugeSource, payload.last_known, mounted],
  );

  const advice = isLastDay
    ? (payload.today?.coach_advice ?? null)
    : (selDay?.coach_advice ?? null);

  const dayLabel = selDay
    ? czDate(selDay.date, { day: "numeric", month: "long" }).toUpperCase()
    : "—";

  if (days.length === 0) {
    return (
      <StateScreen
        title="V databázi nejsou žádné denní metriky"
        detail="Spusť pipeline: python scripts/main.py"
      />
    );
  }

  const pickRange = (value: Range) => {
    setZoom(null);
    setBrushRange(null);
    setRange(value);
    setSel(days.length - 1);
  };

  return (
    <Shell hidden={hidden}>
      <Header today={payload.today} readiness={readiness} statusDot={readyColor} />

      <section
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))",
          gap: 20,
          alignItems: "stretch",
        }}
      >
        <DailyStatus
          readiness={readiness}
          readyColor={readyColor}
          mounted={mounted}
          gauges={gauges}
          advice={advice}
          dayLabel={dayLabel}
          canPrev={effIdx > 0}
          canNext={!isLastDay}
          onPrev={() => setDayIdx(Math.max(0, effIdx - 1))}
          onNext={() => setDayIdx(effIdx + 1 >= lastRowIdx ? -1 : effIdx + 1)}
          onOpenMetric={onOpenMetric}
        />

        <LoadBalance
          pmc={pmc}
          ranges={ranges}
          range={range}
          rangeLabel={window.label}
          theme={T}
          onPickRange={pickRange}
          onSelectDay={setSel}
          onScrub={(fraction) => {
            const w = winRef.current;
            if (w.len < 2) return;
            setSel(w.start + Math.round(fraction * (w.len - 1)));
          }}
          scrubbing={scrubbing}
          onScrubbingChange={(active) => {
            setScrubbing(active);
            if (!active) {
              const w = winRef.current;
              setSel(w.len ? w.start + w.len - 1 : days.length - 1);
            }
          }}
          compareOptions={compareOptions}
          compareYear={compareYear}
          onPickCompare={setCompareYear}
          brushMode={brushMode}
          onToggleBrushMode={() => {
            setBrushMode((m) => !m);
            setBrushRange(null);
          }}
          onBrushChange={setBrushRange}
          onCommitZoom={(fromRel, toRel) => {
            const from = rows[fromRel]?.d;
            const to = rows[toRel]?.d;
            if (!from || !to) return;
            prevRangeRef.current = range;
            setZoom({ from, to });
            setBrushRange(null);
            setBrushMode(false);
            setSel(startIdx + toRel);
          }}
          windowLen={rows.length}
        />
      </section>
    </Shell>
  );
}
