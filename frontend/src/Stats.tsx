import { useEffect, useMemo, useRef, useState } from "react";

import type { ActivityRow, DashboardPayload } from "./api";
import { toDays } from "./api";
import { Dropdown, type DropdownItem } from "./components/Dropdown";
import { StatsChart, type CompareUI } from "./components/StatsChart";
import { mono } from "./components/ui";
import { buildZoneTime, inRange } from "./derive/quality";
import { rangeOptions, selectWindow, type Range } from "./derive/ranges";
import {
  buildCompareSeries,
  buildStatsChart,
  fmtMetricTotal,
  totalOf,
  type MetricKey,
} from "./derive/statsChart";
import { fmtMin, tick } from "./format";
import type { ThemeName } from "./theme";
import { THEMES } from "./theme";

type StatsRange = Range | { from: string; to: string };

function rideWord(n: number): string {
  if (n === 1) return "jízda";
  if (n >= 2 && n <= 4) return "jízdy";
  return "jízd";
}

function shiftYear(iso: string, off: number): string {
  const dt = new Date(`${iso}T00:00:00Z`);
  dt.setUTCFullYear(dt.getUTCFullYear() - off);
  return dt.toISOString().slice(0, 10);
}

export function Stats({
  payload,
  theme,
}: {
  payload: DashboardPayload;
  theme: ThemeName;
}) {
  const T = THEMES[theme];
  const days = useMemo(() => toDays(payload.days), [payload]);
  const activities = payload.activities;

  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setMounted(true), 30);
    return () => clearTimeout(t);
  }, []);

  const [range, setRange] = useState<StatsRange>(30);
  const [rangeMenuOpen, setRangeMenuOpen] = useState(false);
  const [metric, setMetric] = useState<MetricKey | null>(null);
  const [chartMode, setChartMode] = useState<"weekly" | "cumulative">("weekly");
  const [scrubIdx, setScrubIdx] = useState<number | null>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const [brushMode, setBrushMode] = useState(false);
  const [brushEnd, setBrushEnd] = useState<number | null>(null);
  const [compareYear, setCompareYear] = useState<number | null>(null);
  const [compareMenuOpen, setCompareMenuOpen] = useState(false);
  const brushAnchorRef = useRef<number | null>(null);
  const prevRangeRef = useRef<StatsRange>(30);

  const ranges = useMemo(() => rangeOptions(days), [days]);
  const isZoomed = typeof range === "object";
  const isYear = !isZoomed && typeof range === "string" && /^\d{4}$/.test(range);

  const { from, to, rangeLabel } = useMemo(() => {
    if (isZoomed) {
      const r = range as { from: string; to: string };
      return { from: r.from, to: r.to, rangeLabel: `${tick(r.from)}–${tick(r.to)}` };
    }
    const w = selectWindow(days, range as Range);
    const f = w.rows.length ? w.rows[0].d : days[0]?.d ?? "";
    const t = w.rows.length ? w.rows[w.rows.length - 1].d : days[days.length - 1]?.d ?? "";
    return { from: f, to: t, rangeLabel: w.label };
  }, [days, range, isZoomed]);

  const rangeShort = isZoomed
    ? `${tick((range as { from: string }).from)}–${tick((range as { to: string }).to)}`
    : (ranges.find((r) => r.value === range) ?? ranges[0])?.label ?? "";

  const src = useMemo(() => (from && to ? inRange(activities, from, to) : []), [activities, from, to]);

  const pick = (k: MetricKey) => () => {
    setMetric((cur) => (cur === k ? null : k));
    setChartMode("weekly");
  };
  const border = (k: MetricKey) => (metric === k ? "var(--fg2)" : "var(--line)");

  const chart = metric ? buildStatsChart(metric, src, chartMode) : null;

  // ── Srovnání s dřívějším rokem ──────────────────────────────────────────
  const compareEligible = !isZoomed && !isYear && range !== "all";
  const endYear = to ? Number(to.slice(0, 4)) : new Date().getUTCFullYear();
  const minYear = activities.length
    ? Math.min(...activities.map((a) => Number(a.d.slice(0, 4))))
    : endYear;
  const compareOptions = useMemo(() => {
    if (!compareEligible || !from || !to) return [];
    const opts: { off: number; label: string; rows: ActivityRow[] }[] = [];
    for (let off = 1; off <= Math.max(0, endYear - minYear); off++) {
      const f = shiftYear(from, off);
      const t = shiftYear(to, off);
      const rows = activities.filter((a) => a.d >= f && a.d <= t);
      if (rows.length) opts.push({ off, label: String(endYear - off), rows });
    }
    return opts;
  }, [compareEligible, from, to, endYear, minYear, activities]);

  const effectiveCompareYear = compareOptions.some((o) => o.off === compareYear) ? compareYear : null;
  const compareSel = effectiveCompareYear != null ? compareOptions.find((o) => o.off === effectiveCompareYear) : null;

  let compareUI: CompareUI | null = null;
  if (chart && metric) {
    const items: DropdownItem[] = [
      { key: "off", label: "Vypnuto", active: effectiveCompareYear == null, onPick: () => { setCompareYear(null); setCompareMenuOpen(false); } },
      ...compareOptions.map((o) => ({
        key: String(o.off),
        label: o.label,
        active: effectiveCompareYear === o.off,
        onPick: () => { setCompareYear(o.off); setCompareMenuOpen(false); },
      })),
    ];
    let path = "";
    let total = "";
    let diffLabel = "";
    let diffColor = "var(--faint)";
    if (compareSel) {
      const cvals = buildCompareSeries(metric, compareSel.rows, chartMode, chart.daily);
      if (cvals.length) {
        const n = cvals.length;
        const rawMax = chart.rawMax || 1;
        const pts2 = cvals.map((v, i) => ({ x: n > 1 ? (i / (n - 1)) * 700 : 350, y: 120 - Math.min(1, v / rawMax) * 120 }));
        path = pts2.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" ");
        const compareTotalRaw = totalOf(metric, compareSel.rows);
        const currentTotalRaw = totalOf(metric, src);
        total = fmtMetricTotal(metric, compareTotalRaw);
        if (compareTotalRaw > 0) {
          const diffPct = ((currentTotalRaw - compareTotalRaw) / compareTotalRaw) * 100;
          const sign = diffPct > 0.05 ? "+" : diffPct < -0.05 ? "−" : "";
          diffLabel = `${sign}${Math.abs(diffPct).toFixed(0)} %`;
          diffColor = diffPct > 0.05 ? "var(--ok)" : diffPct < -0.05 ? "var(--bad)" : "var(--faint)";
        }
      }
    }
    compareUI = {
      items,
      open: compareMenuOpen,
      onToggle: () => setCompareMenuOpen((v) => !v),
      onClose: () => setCompareMenuOpen(false),
      buttonLabel: compareSel ? compareSel.label : "Srovnat",
      active: !!compareSel,
      path,
      total,
      diffLabel,
      diffColor,
    };
  }

  const onScrubStart = (idx: number) => {
    if (brushMode) {
      brushAnchorRef.current = idx;
      setScrubIdx(idx);
      setBrushEnd(null);
    } else {
      setScrubbing(true);
      setScrubIdx(idx);
    }
  };
  const onScrubMove = (idx: number, isMouse: boolean) => {
    if (brushMode) {
      if (brushAnchorRef.current == null) return;
      setScrubIdx(idx);
      setBrushEnd(idx);
      return;
    }
    if (!(scrubbing || isMouse)) return;
    setScrubIdx(idx);
  };
  const onScrubEnd = () => {
    if (brushMode) {
      const anchor = brushAnchorRef.current;
      const end = brushEnd;
      if (anchor != null && end != null && Math.abs(end - anchor) >= 1 && chart) {
        const keys = chart.keys;
        const a = Math.min(anchor, end);
        const b = Math.max(anchor, end);
        const toDt = new Date(`${keys[b]}T00:00:00Z`);
        if (!chart.daily) toDt.setUTCDate(toDt.getUTCDate() + 6);
        prevRangeRef.current = range;
        setRange({ from: keys[a], to: toDt.toISOString().slice(0, 10) });
        setRangeMenuOpen(false);
      }
      setScrubIdx(null);
      setBrushEnd(null);
      setBrushMode(false);
      brushAnchorRef.current = null;
      return;
    }
    setScrubbing(false);
    setScrubIdx(null);
  };
  const onResetZoom = () => {
    if (!isZoomed) return;
    setRange(prevRangeRef.current ?? 30);
    setScrubIdx(null);
    setBrushEnd(null);
    setBrushMode(false);
  };
  const onToggleBrush = () => {
    setBrushMode((v) => !v);
    setScrubIdx(null);
    setBrushEnd(null);
  };

  // ── Souhrny za období ────────────────────────────────────────────────────
  const totKm = totalOf("km", src);
  const totAsc = totalOf("ascent", src);
  const totKcal = totalOf("kcal", src);
  const totDur = totalOf("time", src);
  const totFat = totalOf("fat", src);
  const totCarb = totalOf("carb", src);
  const totUpT = totalOf("up", src);
  const totDownT = totalOf("down", src);
  const totFlatT = totalOf("flat", src);
  const avgSpeed = totDur ? totKm / (totDur / 60) : 0;
  const fatKcal = totFat * 9;
  const carbKcal = totCarb * 4;
  const macroTot = fatKcal + carbKcal || 1;
  const zoneTime = useMemo(() => buildZoneTime(src, T, mounted), [src, T, mounted]);

  const chartAfter = (k: MetricKey) => !!chart && metric === k;

  const rangeItems: DropdownItem[] = ranges.map((r) => ({
    key: String(r.value),
    label: r.label,
    active: !isZoomed && range === r.value,
    onPick: () => { setRange(r.value); setRangeMenuOpen(false); },
  }));

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg)",
        color: "var(--fg)",
        fontFamily: "var(--font-sans),'DM Sans',system-ui,sans-serif",
        padding: "28px 20px 96px",
        transition: "background .35s ease,color .35s ease",
      }}
    >
      <div style={{ maxWidth: 1120, margin: "0 auto", display: "flex", flexDirection: "column", gap: 20 }}>
        <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap", padding: "2px 2px 6px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>
              STATS
            </span>
            <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>Statistiky</h1>
          </div>
          <Dropdown
            open={rangeMenuOpen}
            onToggle={() => setRangeMenuOpen((v) => !v)}
            onClose={() => setRangeMenuOpen(false)}
            buttonLabel={<span style={{ minWidth: 38, textAlign: "left" }}>{rangeShort}</span>}
            buttonExtra={
              <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="var(--mut)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M6 9l6 6 6-6" />
              </svg>
            }
            items={rangeItems}
          />
        </header>

        <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 12, flexWrap: "wrap" }}>
            <span style={mono(11, { color: "var(--faint)" })}>
              {rangeLabel} · {src.length} {rideWord(src.length)}
            </span>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2,1fr)", gap: 12 }}>
            <StatCard label="Vzdálenost" big={totKm.toFixed(0)} unit="km" active={border("km")} onClick={pick("km")} subLabel="za jízdu" subValue={`${src.length ? (totKm / src.length).toFixed(1).replace(".", ",") : "0"} km`} />
            {chartAfter("km") && chart && metric && (
              <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
            )}

            <StatCard label="Převýšení" big={Math.round(totAsc).toLocaleString("cs-CZ")} unit="m" active={border("ascent")} onClick={pick("ascent")} subLabel="na 100 km" subValue={`${totKm ? Math.round(totAsc / totKm * 100).toLocaleString("cs-CZ") : "0"} m`} />
            {chartAfter("ascent") && chart && metric && (
              <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
            )}

            <StatCard label="Kalorie" big={Math.round(totKcal).toLocaleString("cs-CZ")} bigColor="var(--orange)" unit="kcal" active={border("kcal")} onClick={pick("kcal")} subLabel="za hodinu" subValue={totDur ? `${Math.round(totKcal / (totDur / 60)).toLocaleString("cs-CZ")} kcal` : "0 kcal"} />
            {chartAfter("kcal") && chart && metric && (
              <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
            )}

            <div onClick={pick("time")} style={{ cursor: "pointer", border: `1px solid ${border("time")}`, borderRadius: 18, background: "var(--card)", padding: 20, display: "flex", flexDirection: "column", gap: 6, minWidth: 0, transition: "border-color .2s" }}>
              <span style={mono(9, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>Čas na kole</span>
              <div style={{ display: "flex", alignItems: "baseline", gap: 6, minWidth: 0 }}>
                <span style={mono(22, { lineHeight: 1.15, letterSpacing: "-.02em", overflowWrap: "break-word" })}>{fmtMin(totDur)}</span>
              </div>
              <div style={{ display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)", minWidth: 0 }}>
                <span style={mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", whiteSpace: "nowrap" })}>za jízdu</span>
                <span style={mono(12, { color: "var(--fg2)", marginLeft: "auto", whiteSpace: "nowrap" })}>{src.length ? fmtMin(totDur / src.length) : "0 min"}</span>
              </div>
            </div>
            {chartAfter("time") && chart && metric && (
              <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
            )}

            <StatCard label="Prům. rychlost" big={avgSpeed.toFixed(1).replace(".", ",")} bigColor="var(--blue)" unit="km/h" active={border("speed")} onClick={pick("speed")} subLabel={rangeLabel} subValue="" />
            {chartAfter("speed") && chart && metric && (
              <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
            )}
          </div>

          <div style={{ height: 1, background: "var(--line)" }} />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>Živiny ze spálených kalorií</span>
              <span style={mono(11, { color: "var(--faint)" })}>{Math.round(totKcal).toLocaleString("cs-CZ")} kcal celkem</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 20 }}>
              <NutrientCard label="Tuky" dot="var(--fatc)" value={Math.round(totFat).toLocaleString("cs-CZ")} active={border("fat")} onClick={pick("fat")} />
              {chartAfter("fat") && chart && metric && (
                <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
              )}
              <NutrientCard label="Cukry" dot="var(--carbc)" value={Math.round(totCarb).toLocaleString("cs-CZ")} active={border("carb")} onClick={pick("carb")} />
              {chartAfter("carb") && chart && metric && (
                <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
              )}
            </div>
            <div style={{ display: "flex", height: 8, borderRadius: 999, overflow: "hidden", background: "var(--track)", gap: 2 }}>
              <div style={{ width: mounted ? `${((fatKcal / macroTot) * 100).toFixed(1)}%` : "0%", background: "var(--fatc)", transition: "width .8s cubic-bezier(.22,1,.36,1)" }} />
              <div style={{ width: mounted ? `${((carbKcal / macroTot) * 100).toFixed(1)}%` : "0%", background: "var(--carbc)", transition: "width .8s cubic-bezier(.22,1,.36,1)" }} />
            </div>
          </div>

          <div style={{ height: 1, background: "var(--line)" }} />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>Terén</span>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(180px,1fr))", gap: 12 }}>
              <TerrainCard
                label="Do kopce"
                color="var(--warn)"
                icon="M4 19h16L14 6l-3.5 6L8 9z"
                value={fmtMin(totUpT)}
                share={totDur ? `${((totUpT / totDur) * 100).toFixed(0)} % jízdy` : "0 % jízdy"}
                active={border("up")}
                onClick={pick("up")}
              />
              {chartAfter("up") && chart && metric && (
                <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
              )}
              <TerrainCard
                label="Z kopce"
                color="var(--ok)"
                icon="M4 6h16L14 19l-3.5-6L8 16z"
                value={fmtMin(totDownT)}
                share={totDur ? `${((totDownT / totDur) * 100).toFixed(0)} % jízdy` : "0 % jízdy"}
                active={border("down")}
                onClick={pick("down")}
              />
              {chartAfter("down") && chart && metric && (
                <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
              )}
              <TerrainCard
                label="Po rovině"
                color="var(--blue)"
                icon="M3 12h18"
                value={fmtMin(totFlatT)}
                share={totDur ? `${((totFlatT / totDur) * 100).toFixed(0)} % jízdy` : "0 % jízdy"}
                active={border("flat")}
                onClick={pick("flat")}
              />
              {chartAfter("flat") && chart && metric && (
                <StatsChart chart={chart} rangeLabel={rangeLabel} mode={chartMode} onSetMode={setChartMode} onClose={() => setMetric(null)} brushMode={brushMode} onToggleBrush={onToggleBrush} scrubIdx={scrubIdx} scrubbing={scrubbing} brushEnd={brushEnd} brushAnchor={brushAnchorRef.current} onScrubStart={onScrubStart} onScrubMove={onScrubMove} onScrubEnd={onScrubEnd} onResetZoom={onResetZoom} isZoomed={isZoomed} compare={compareUI} />
              )}
            </div>
          </div>

          <div style={{ height: 1, background: "var(--line)" }} />

          <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
            <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
              <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>Čas v zónách</span>
              <span style={mono(11, { color: "var(--faint)" })}>{zoneTime.total} · {zoneTime.rides}</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {zoneTime.rows.map((z, i) => (
                <div key={i} style={{ display: "grid", gridTemplateColumns: "96px 1fr 90px 46px", gap: 12, alignItems: "center" }}>
                  <span style={mono(10, { display: "flex", alignItems: "center", gap: 8, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", whiteSpace: "nowrap" })}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: z.color }} />
                    {z.name}
                  </span>
                  <div style={{ height: 8, borderRadius: 999, background: "var(--track)", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: z.w, borderRadius: 999, background: z.color, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
                  </div>
                  <span style={mono(12, { color: "var(--fg)", textAlign: "right", whiteSpace: "nowrap" })}>{z.time}</span>
                  <span style={mono(11, { color: "var(--faint)", textAlign: "right" })}>{z.pct} %</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({
  label,
  big,
  unit,
  bigColor,
  active,
  onClick,
  subLabel,
  subValue,
}: {
  label: string;
  big: string;
  unit: string;
  bigColor?: string;
  active: string;
  onClick: () => void;
  subLabel: string;
  subValue: string;
}) {
  return (
    <div onClick={onClick} style={{ cursor: "pointer", border: `1px solid ${active}`, borderRadius: 18, background: "var(--card)", padding: 20, display: "flex", flexDirection: "column", gap: 6, minWidth: 0, transition: "border-color .2s" }}>
      <span style={mono(9, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>{label}</span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6, minWidth: 0 }}>
        <span style={mono(26, { lineHeight: 1.05, letterSpacing: "-.03em", whiteSpace: "nowrap", color: bigColor })}>{big}</span>
        {unit && <span style={mono(10, { color: "var(--faint)" })}>{unit}</span>}
      </div>
      {subValue !== "" ? (
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)", minWidth: 0 }}>
          <span style={mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", whiteSpace: "nowrap" })}>{subLabel}</span>
          <span style={mono(12, { color: "var(--fg2)", marginLeft: "auto", whiteSpace: "nowrap" })}>{subValue}</span>
        </div>
      ) : (
        <div style={{ display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)", minWidth: 0 }}>
          <span style={mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", whiteSpace: "nowrap" })}>{subLabel}</span>
        </div>
      )}
    </div>
  );
}

function NutrientCard({
  label,
  dot,
  value,
  active,
  onClick,
}: {
  label: string;
  dot: string;
  value: string;
  active: string;
  onClick: () => void;
}) {
  return (
    <div onClick={onClick} style={{ cursor: "pointer", display: "flex", flexDirection: "column", gap: 8, padding: 10, margin: -8, borderRadius: 14, border: `1px solid ${active}`, transition: "border-color .2s" }}>
      <span style={mono(9, { display: "flex", alignItems: "center", gap: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>
        <span style={{ width: 8, height: 8, borderRadius: 2, background: dot }} />
        {label}
      </span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={mono(26, { letterSpacing: "-.03em" })}>{value}</span>
        <span style={mono(11, { color: "var(--faint)" })}>g</span>
      </div>
    </div>
  );
}

function TerrainCard({
  label,
  color,
  icon,
  value,
  share,
  active,
  onClick,
}: {
  label: string;
  color: string;
  icon: string;
  value: string;
  share: string;
  active: string;
  onClick: () => void;
}) {
  return (
    <div onClick={onClick} style={{ cursor: "pointer", display: "flex", flexDirection: "column", gap: 10, padding: 16, border: `1px solid ${active}`, borderRadius: 16, background: "var(--card)", transition: "border-color .2s" }}>
      <span style={mono(9, { display: "flex", alignItems: "center", gap: 8, letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>
        <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d={icon} />
        </svg>
        {label}
      </span>
      <span style={mono(22, { color })}>{value}</span>
      <span style={mono(12, { color: "var(--faint)" })}>{share}</span>
    </div>
  );
}
