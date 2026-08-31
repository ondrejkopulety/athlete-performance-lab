import { useRef, useState, type PointerEvent as ReactPointerEvent } from "react";

import type { Pmc } from "../derive/pmc";
import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { RangeDropdown } from "./RangeDropdown";
import { CARD, mono } from "./ui";

export interface CompareOption {
  off: number;
  label: string;
}

/**
 * „Bilance zátěže" – 1:1 s Readiness Dashboard.dc.html: CTL/ATL křivky,
 * srovnání s předchozím rokem (čárkovaně), výběr úseku (brush) a sloupce
 * denní zátěže.
 */
export function LoadBalance({
  pmc,
  ranges,
  range,
  rangeLabel,
  theme,
  onPickRange,
  onSelectDay,
  onScrub,
  scrubbing,
  onScrubbingChange,
  compareOptions,
  compareYear,
  onPickCompare,
  brushMode,
  onToggleBrushMode,
  onBrushChange,
  onCommitZoom,
  windowLen,
}: {
  pmc: Pmc;
  ranges: RangeOption[];
  range: Range;
  rangeLabel: string;
  theme: Theme;
  onPickRange: (value: Range) => void;
  onSelectDay: (index: number) => void;
  onScrub: (fraction: number) => void;
  scrubbing: boolean;
  onScrubbingChange: (active: boolean) => void;
  compareOptions: CompareOption[];
  compareYear: number | null;
  onPickCompare: (off: number | null) => void;
  brushMode: boolean;
  onToggleBrushMode: () => void;
  /** aktuální rozsah brushe v relativních indexech okna (nebo null) */
  onBrushChange: (range: [number, number] | null) => void;
  onCommitZoom: (fromRel: number, toRel: number) => void;
  windowLen: number;
}) {
  const anchorRef = useRef<number | null>(null);
  const endRef = useRef<number | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);

  const idxFromEvent = (e: ReactPointerEvent<HTMLDivElement>): number | null => {
    if (windowLen < 2) return null;
    const r = e.currentTarget.getBoundingClientRect();
    if (r.width === 0) return null;
    const fr = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    return Math.round(fr * (windowLen - 1));
  };

  const onDown = (e: ReactPointerEvent<HTMLDivElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch {
      /* Safari */
    }
    if (brushMode) {
      const idx = idxFromEvent(e);
      if (idx == null) return;
      anchorRef.current = idx;
      endRef.current = idx;
      onBrushChange([idx, idx]);
      return;
    }
    onScrubbingChange(true);
    scrubAt(e);
  };
  const onMove = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (brushMode) {
      if (anchorRef.current == null) return;
      const idx = idxFromEvent(e);
      if (idx == null) return;
      endRef.current = idx;
      onBrushChange([Math.min(anchorRef.current, idx), Math.max(anchorRef.current, idx)]);
      return;
    }
    if (scrubbing || e.pointerType === "mouse") scrubAt(e);
  };
  const onUp = (e: ReactPointerEvent<HTMLDivElement>) => {
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      /* Safari */
    }
    if (brushMode) {
      const a = anchorRef.current;
      const b = endRef.current;
      anchorRef.current = null;
      endRef.current = null;
      if (a != null && b != null && Math.abs(b - a) >= 1) {
        onCommitZoom(Math.min(a, b), Math.max(a, b));
      } else {
        onBrushChange(null);
      }
      return;
    }
    onScrubbingChange(false);
  };
  const scrubAt = (e: ReactPointerEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    if (r.width === 0) return;
    onScrub(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
  };

  const compareLabel = compareYear != null
    ? String(compareOptions.find((o) => o.off === compareYear)?.label ?? compareYear)
    : "Srovnat";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, minWidth: 0 }}>
      <div style={{ ...CARD, flex: 1, padding: "24px 28px 20px", gap: 16, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={mono(10, { letterSpacing: ".16em", textTransform: "uppercase", color: "var(--mut)" })}>Bilance zátěže</span>
            <span style={{ fontSize: 12, color: "var(--faint)" }}>Kondice vs únava · {rangeLabel}</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {compareOptions.length > 0 && (
              <div style={{ position: "relative" }}>
                <button
                  type="button"
                  onClick={() => setMenuOpen((o) => !o)}
                  style={mono(10, {
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    letterSpacing: ".06em",
                    padding: "6px 10px 6px 12px",
                    borderRadius: 999,
                    border: "1px solid var(--line2)",
                    background: compareYear != null ? theme.blue : "var(--track)",
                    color: compareYear != null ? theme.card : "var(--fg2)",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                  })}
                >
                  <span style={{ width: 11, height: 0, borderTop: "1.5px dashed currentColor" }} />
                  <span>{compareLabel}</span>
                  <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M6 9l6 6 6-6" />
                  </svg>
                </button>
                {menuOpen && (
                  <>
                    <div onClick={() => setMenuOpen(false)} style={{ position: "fixed", inset: 0, zIndex: 20 }} />
                    <div style={{ position: "absolute", top: "calc(100% + 7px)", left: 0, zIndex: 21, display: "flex", flexDirection: "column", gap: 2, padding: 6, minWidth: 110, border: "1px solid var(--line2)", borderRadius: 14, background: "var(--card2)", boxShadow: "0 14px 34px rgba(0,0,0,.28)" }}>
                      {[{ off: null as number | null, label: "Vypnuto" }, ...compareOptions].map((o) => {
                        const on = compareYear === o.off;
                        return (
                          <button
                            key={String(o.off)}
                            type="button"
                            onClick={() => {
                              onPickCompare(o.off);
                              setMenuOpen(false);
                            }}
                            style={mono(11, {
                              letterSpacing: ".04em",
                              textAlign: "left",
                              padding: "8px 10px",
                              borderRadius: 9,
                              border: "none",
                              cursor: "pointer",
                              whiteSpace: "nowrap",
                              background: on ? theme.fg : "transparent",
                              color: on ? theme.card : theme.fg2,
                            })}
                          >
                            {o.label}
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            )}
            <button
              type="button"
              onClick={onToggleBrushMode}
              title="Vybrat časový úsek v grafu"
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 26,
                height: 26,
                borderRadius: 999,
                border: "1px solid var(--line2)",
                background: brushMode ? theme.fg2 : "var(--track)",
                color: brushMode ? theme.card : "var(--fg2)",
                cursor: "pointer",
                flex: "none",
              }}
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 12h4M16 12h4M9 5v14M15 5v14" />
              </svg>
            </button>
            <RangeDropdown options={ranges} value={range} onPick={onPickRange} theme={theme} />
          </div>
        </div>

        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <Stat label="CTL" dot="var(--blue)" value={pmc.sel.ctl} />
          <Stat label="ATL" dot="var(--orange)" value={pmc.sel.atl} />
          {pmc.showCompareLegend && (
            <Stat label={`CTL ${pmc.compareLegendYear}`} dash value={pmc.sel.ctlPrev} color="var(--faint)" />
          )}
          <Stat label="TSB" value={pmc.sel.tsb} color={pmc.tsbColor} />
          <Stat label="ACWR" value={pmc.sel.acwr} color="var(--mut)" />
        </div>

        <div style={{ position: "relative", width: "100%", flex: 1, minHeight: 190, display: "flex", paddingLeft: 40 }}>
          {pmc.yTicks.map((t) => (
            <span key={t.top} style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: t.top, transform: "translateY(-50%)" })}>
              {t.label}
            </span>
          ))}

          <svg viewBox="0 0 700 220" preserveAspectRatio="none" style={{ width: "100%", height: "100%", minHeight: 190, display: "block", overflow: "visible" }}>
            <defs>
              <linearGradient id="ctlFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" style={{ stopColor: "var(--blue)" }} stopOpacity="0.22" />
                <stop offset="100%" style={{ stopColor: "var(--blue)" }} stopOpacity="0" />
              </linearGradient>
            </defs>
            {pmc.gridLines.map((g) => (
              <line key={g.y} x1="0" y1={g.y} x2="700" y2={g.y} style={{ stroke: "var(--line)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" />
            ))}
            <path d={pmc.ctlArea} fill="url(#ctlFill)" />
            {pmc.ctlLineCompare && (
              <path d={pmc.ctlLineCompare} fill="none" style={{ stroke: "var(--blue)" }} strokeWidth="2" strokeDasharray="6 5" strokeLinejoin="round" strokeLinecap="round" opacity="0.55" vectorEffect="non-scaling-stroke" />
            )}
            <path d={pmc.atlLine} fill="none" style={{ stroke: "var(--orange)" }} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            <path d={pmc.ctlLine} fill="none" style={{ stroke: "var(--blue)" }} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
            <line x1={pmc.selX} y1="0" x2={pmc.selX} y2="196" style={{ stroke: "var(--grey)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" />
            <circle cx={pmc.selX} cy={pmc.selCtlY} r="4" style={{ fill: "var(--card)", stroke: "var(--blue)" }} strokeWidth="2" vectorEffect="non-scaling-stroke" />
            <circle cx={pmc.selX} cy={pmc.selAtlY} r="4" style={{ fill: "var(--card)", stroke: "var(--orange)" }} strokeWidth="2" vectorEffect="non-scaling-stroke" />
            <rect x={pmc.brushX} y="0" width={pmc.brushW} height="196" style={{ fill: "var(--fg2)" }} opacity={pmc.brushOn} fillOpacity="0.14" />
          </svg>

          <div
            onPointerDown={onDown}
            onPointerMove={onMove}
            onPointerUp={onUp}
            onPointerCancel={onUp}
            onPointerLeave={onUp}
            style={{ position: "absolute", inset: "0 0 0 40px", touchAction: "none", cursor: "crosshair" }}
          >
            <div
              style={mono(9, {
                position: "absolute",
                top: 0,
                left: pmc.selPct,
                transform: "translateX(-50%)",
                padding: "4px 8px",
                borderRadius: 999,
                background: "var(--track)",
                color: "var(--fg)",
                letterSpacing: ".06em",
                whiteSpace: "nowrap",
                pointerEvents: "none",
                opacity: scrubbing ? "1" : "0.55",
                transition: "opacity .2s",
              })}
            >
              {pmc.sel.label}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <span style={mono(9, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>{pmc.barsLabel}</span>
          <div style={{ position: "relative", display: "flex", alignItems: "flex-end", gap: 4, height: 68, paddingLeft: 40 }}>
            <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}>{pmc.barMax}</span>
            <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", bottom: -1 })}>0</span>
            {pmc.bars.map((b, i) => (
              <div
                key={i}
                onMouseEnter={() => onSelectDay(b.dayIndex)}
                onClick={() => onSelectDay(b.dayIndex)}
                style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "flex-end", height: "100%", cursor: "pointer", gap: 6 }}
              >
                <div style={{ height: b.barH, borderRadius: 4, background: b.barColor, transition: "height .5s cubic-bezier(.22,1,.36,1),opacity .2s", opacity: b.barOpacity, minHeight: 3 }} />
              </div>
            ))}
          </div>
          <div style={{ display: "flex", gap: 4, paddingLeft: 40 }}>
            {pmc.bars.map((b, i) => (
              <span key={i} style={{ flex: 1, position: "relative", overflow: "visible", height: 12 }}>
                <span style={mono(9, { position: "absolute", left: "50%", top: 0, transform: "translateX(-50%)", color: b.tickColor, whiteSpace: "nowrap" })}>{b.tick}</span>
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({
  label,
  value,
  dot,
  dash,
  color,
}: {
  label: string;
  value: string;
  dot?: string;
  dash?: boolean;
  color?: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={mono(9, { display: "flex", alignItems: "center", gap: 6, letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>
        {dot && <span style={{ width: 8, height: 8, borderRadius: 2, background: dot }} />}
        {dash && <span style={{ width: 11, height: 0, borderTop: "1.5px dashed var(--blue)" }} />}
        {label}
      </span>
      <span style={mono(20, color ? { color } : {})}>{value}</span>
    </div>
  );
}
