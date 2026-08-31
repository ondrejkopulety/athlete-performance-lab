import type { PointerEvent as ReactPointerEvent } from "react";

import type { StatsChartData } from "../derive/statsChart";
import { weekRangeLabel } from "../derive/statsChart";
import { tick } from "../format";
import { Dropdown, type DropdownItem } from "./Dropdown";
import { mono } from "./ui";

export interface CompareUI {
  items: DropdownItem[];
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  buttonLabel: string;
  active: boolean;
  path: string;
  total: string;
  diffLabel: string;
  diffColor: string;
}

const CLOSE_ICON =
  "M6 6l12 12M18 6L6 18";
const BRUSH_ICON_CIRCLE = { cx: 10.5, cy: 10.5, r: 6.5 };

function idxFromEvent(e: ReactPointerEvent<HTMLDivElement>, n: number): number | null {
  const rect = e.currentTarget.getBoundingClientRect();
  if (!rect.width || n < 1) return null;
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  return Math.round(frac * (n - 1));
}

/**
 * Rozbalený graf jedné metriky na stránce Stats – jedna implementace pro
 * všech deset karet (design měl tenhle blok desetkrát doslova zkopírovaný).
 */
export function StatsChart({
  chart,
  rangeLabel,
  mode,
  onSetMode,
  onClose,
  brushMode,
  onToggleBrush,
  scrubIdx,
  scrubbing,
  brushEnd,
  brushAnchor,
  onScrubStart,
  onScrubMove,
  onScrubEnd,
  onResetZoom,
  isZoomed,
  compare,
}: {
  chart: StatsChartData;
  rangeLabel: string;
  mode: "weekly" | "cumulative";
  onSetMode: (m: "weekly" | "cumulative") => void;
  onClose: () => void;
  brushMode: boolean;
  onToggleBrush: () => void;
  scrubIdx: number | null;
  scrubbing: boolean;
  brushEnd: number | null;
  brushAnchor: number | null;
  onScrubStart: (idx: number) => void;
  onScrubMove: (idx: number, isMouse: boolean) => void;
  onScrubEnd: () => void;
  onResetZoom: () => void;
  isZoomed: boolean;
  compare: CompareUI | null;
}) {
  const n = chart.points.length;
  const sIdx = scrubIdx != null && scrubIdx < n ? scrubIdx : null;
  const scrubbingActive = scrubbing && sIdx != null;
  const scrubX = sIdx == null ? -20 : chart.points[sIdx].x;
  const scrubDotY = sIdx == null ? -20 : chart.points[sIdx].y;
  const scrubLeft = sIdx == null ? "50%" : `${((chart.points[sIdx]?.x ?? 0) / 700) * 100}%`;
  const scrubLabel =
    sIdx == null
      ? ""
      : brushMode
        ? "Táhněte pro výběr →"
        : `${chart.daily ? tick(chart.keys[sIdx]) : weekRangeLabel(chart.keys[sIdx])} · ${chart.fmtVals[sIdx]}`;

  const brushActive = brushMode && brushAnchor != null && brushEnd != null;
  const bA = brushActive ? Math.min(brushAnchor as number, brushEnd as number) : 0;
  const bB = brushActive ? Math.max(brushAnchor as number, brushEnd as number) : 0;
  const brushX = brushActive ? chart.points[bA]?.x ?? 0 : 0;
  const brushW = brushActive ? Math.max(1, (chart.points[bB]?.x ?? 0) - brushX) : 0;

  const pointerHandlers = {
    onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => {
      try {
        e.currentTarget.setPointerCapture(e.pointerId);
      } catch {
        /* Safari občas capture odmítne */
      }
      const idx = idxFromEvent(e, n);
      if (idx != null) onScrubStart(idx);
    },
    onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => {
      const idx = idxFromEvent(e, n);
      if (idx != null) onScrubMove(idx, e.pointerType === "mouse");
    },
    onPointerUp: (e: ReactPointerEvent<HTMLDivElement>) => {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        /* viz výše */
      }
      onScrubEnd();
    },
    onPointerCancel: onScrubEnd,
    onPointerLeave: onScrubEnd,
    onDoubleClick: onResetZoom,
  };

  return (
    <div
      style={{
        gridColumn: "1/-1",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        padding: 20,
        border: "1px solid var(--line2)",
        borderRadius: 18,
        background: "var(--card2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: chart.color })}>
            {chart.label} · po týdnech
          </span>
          <span style={mono(11, { color: "var(--faint)" })}>{rangeLabel}</span>
          {compare?.active && (
            <span style={mono(10, { display: "flex", alignItems: "center", gap: 6, color: "var(--faint)" })}>
              <span style={{ width: 11, height: 0, borderTop: `1.5px dashed ${chart.color}` }} />
              {compare.buttonLabel} · {compare.total}
              <span style={{ color: compare.diffColor }}>{compare.diffLabel}</span>
            </span>
          )}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ display: "flex", gap: 2, padding: 2, borderRadius: 999, background: "var(--track)" }}>
            <button
              type="button"
              onClick={() => onSetMode("weekly")}
              style={mono(10, {
                letterSpacing: ".04em",
                padding: "6px 12px",
                borderRadius: 999,
                border: "none",
                cursor: "pointer",
                whiteSpace: "nowrap",
                background: mode === "weekly" ? "var(--fg)" : "transparent",
                color: mode === "weekly" ? "var(--card)" : "var(--fg2)",
              })}
            >
              Týdny
            </button>
            <button
              type="button"
              onClick={() => onSetMode("cumulative")}
              style={mono(10, {
                letterSpacing: ".04em",
                padding: "6px 12px",
                borderRadius: 999,
                border: "none",
                cursor: "pointer",
                whiteSpace: "nowrap",
                background: mode === "cumulative" ? "var(--fg)" : "transparent",
                color: mode === "cumulative" ? "var(--card)" : "var(--fg2)",
              })}
            >
              Kumulativně
            </button>
          </div>
          {compare && (
            <Dropdown
              open={compare.open}
              onToggle={compare.onToggle}
              onClose={compare.onClose}
              buttonLabel={compare.buttonLabel}
              items={compare.items}
              minWidth={120}
              buttonStyle={{
                background: compare.active ? "var(--blue)" : "var(--track)",
                color: compare.active ? "var(--card)" : "var(--fg2)",
                border: "none",
                padding: "6px 12px",
              }}
            />
          )}
          <button
            type="button"
            onClick={onToggleBrush}
            title="Přiblížit výběrem v grafu"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 26,
              height: 26,
              borderRadius: 999,
              border: "1px solid var(--line2)",
              background: brushMode ? "var(--fg2)" : "transparent",
              color: brushMode ? "var(--card)" : "var(--mut)",
              cursor: "pointer",
            }}
          >
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx={BRUSH_ICON_CIRCLE.cx} cy={BRUSH_ICON_CIRCLE.cy} r={BRUSH_ICON_CIRCLE.r} />
              <path d="M15.5 15.5L20 20" />
            </svg>
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Zavřít"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 26,
              height: 26,
              borderRadius: 999,
              border: "1px solid var(--line2)",
              background: "transparent",
              color: "var(--mut)",
              cursor: "pointer",
            }}
          >
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
              <path d={CLOSE_ICON} />
            </svg>
          </button>
        </div>
      </div>

      <div style={{ position: "relative", height: 120, paddingLeft: 44 }}>
        <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}>
          {chart.max}
        </span>
        <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", bottom: -1 })}>
          0
        </span>

        {chart.isBar && (
          <div style={{ position: "absolute", left: 44, right: 0, top: 0, bottom: 0, display: "flex", alignItems: "flex-end", gap: 2 }}>
            {chart.bars.map((b, i) => (
              <div key={i} style={{ flex: 1, height: "100%", display: "flex", flexDirection: "column", justifyContent: "flex-end" }}>
                <div
                  style={{
                    height: b.h,
                    minHeight: 2,
                    borderRadius: 3,
                    background: chart.color,
                    transition: "height .55s cubic-bezier(.22,1,.36,1)",
                  }}
                />
              </div>
            ))}
          </div>
        )}

        {chart.isLine && (
          <svg
            viewBox="0 0 700 120"
            preserveAspectRatio="none"
            style={{ position: "absolute", left: 44, right: 0, top: 0, width: "calc(100% - 44px)", height: "100%", display: "block", overflow: "visible" }}
          >
            <defs>
              <linearGradient id="statsGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" style={{ stopColor: chart.color }} stopOpacity="0.28" />
                <stop offset="100%" style={{ stopColor: chart.color }} stopOpacity="0" />
              </linearGradient>
            </defs>
            <path d={chart.areaPath} fill="url(#statsGrad)" />
            <path d={chart.linePath} fill="none" style={{ stroke: chart.color }} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
          </svg>
        )}

        <svg
          viewBox="0 0 700 120"
          preserveAspectRatio="none"
          style={{ position: "absolute", left: 44, right: 0, top: 0, width: "calc(100% - 44px)", height: "100%", display: "block", overflow: "visible", pointerEvents: "none" }}
        >
          {compare && (
            <path d={compare.path} fill="none" style={{ stroke: chart.color }} strokeWidth="1.8" strokeDasharray="5 4" strokeLinejoin="round" strokeLinecap="round" opacity="0.6" vectorEffect="non-scaling-stroke" />
          )}
          <line x1={scrubX} y1={0} x2={scrubX} y2={120} style={{ stroke: "var(--fg2)" }} strokeWidth={1} vectorEffect="non-scaling-stroke" opacity={scrubbingActive ? 1 : 0} />
          <circle cx={scrubX} cy={scrubDotY} r={4} style={{ fill: "var(--card)", stroke: chart.color }} strokeWidth={2} opacity={scrubbingActive && chart.isLine ? 1 : 0} />
          <rect x={brushX} y={0} width={brushW} height={120} style={{ fill: "var(--fg2)" }} opacity={brushActive ? 1 : 0} fillOpacity={0.14} />
        </svg>

        <div
          style={{
            position: "absolute",
            top: 2,
            left: scrubLeft,
            transform: "translateX(-50%)",
            padding: "4px 8px",
            borderRadius: 999,
            background: "var(--track)",
            color: "var(--fg)",
            pointerEvents: "none",
            opacity: scrubbingActive || (brushMode && sIdx != null) ? 1 : 0,
            transition: "opacity .2s",
            ...mono(9, { whiteSpace: "nowrap" }),
          }}
        >
          {scrubLabel}
        </div>

        <div
          {...pointerHandlers}
          style={{ position: "absolute", left: 44, right: 0, top: 0, bottom: 0, touchAction: "none", cursor: "crosshair" }}
        />
      </div>

      <div style={{ display: "flex", gap: 2, paddingLeft: 44 }}>
        {chart.bars.map((b, i) => (
          <span key={i} style={{ flex: 1, position: "relative", height: 12 }}>
            <span style={mono(9, { position: "absolute", left: "50%", top: 0, transform: "translateX(-50%)", color: "var(--faint)", whiteSpace: "nowrap" })}>
              {b.tick}
            </span>
          </span>
        ))}
      </div>

      {isZoomed && (
        <button
          type="button"
          onClick={onResetZoom}
          style={mono(10, { alignSelf: "flex-start", color: "var(--faint)", background: "none", border: "none", cursor: "pointer", padding: 0 })}
        >
          ← zpět z přiblížení (nebo dvojklik do grafu)
        </button>
      )}
    </div>
  );
}
