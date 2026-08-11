import type { Pmc } from "../derive/pmc";
import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { RangeTabs } from "./RangeTabs";
import { CARD, mono, SECTION_LABEL, SMALL_LABEL } from "./ui";
import { useScrub } from "./useScrub";

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
}) {
  const scrub = useScrub(scrubbing, onScrubbingChange, onScrub);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20, minWidth: 0 }}>
      <div style={{ ...CARD, flex: 1, padding: "24px 26px 20px", gap: 16, minWidth: 0 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <span style={SECTION_LABEL}>Bilance zátěže</span>
            <span style={{ fontSize: 12, color: "var(--faint)" }}>
              Kondice vs únava · {rangeLabel}
            </span>
          </div>
          <RangeTabs options={ranges} value={range} onPick={onPickRange} theme={theme} />
        </div>

        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <Stat label="Kondice" dot="var(--blue)" value={pmc.sel.ctl} />
          <Stat label="Únava" dot="var(--orange)" value={pmc.sel.atl} />
          <Stat label="Forma · TSB" value={pmc.sel.tsb} color={pmc.tsbColor} />
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 2,
              marginLeft: "auto",
              textAlign: "right",
            }}
          >
            <span style={SMALL_LABEL}>{pmc.sel.label}</span>
            <span style={mono(20, { color: "var(--mut)" })}>ACWR {pmc.sel.acwr}</span>
          </div>
        </div>

        <div
          style={{
            position: "relative",
            width: "100%",
            flex: 1,
            minHeight: 190,
            display: "flex",
            paddingLeft: 40,
          }}
        >
          {pmc.yTicks.map((t) => (
            <span
              key={t.top}
              style={mono(9, {
                position: "absolute",
                left: 0,
                color: "var(--faint)",
                whiteSpace: "nowrap",
                top: t.top,
                transform: "translateY(-50%)",
              })}
            >
              {t.label}
            </span>
          ))}

          <svg
            viewBox="0 0 700 220"
            preserveAspectRatio="none"
            style={{ width: "100%", height: "100%", minHeight: 190, display: "block", overflow: "visible" }}
          >
            <defs>
              <linearGradient id="ctlFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" style={{ stopColor: "var(--blue)" }} stopOpacity="0.22" />
                <stop offset="100%" style={{ stopColor: "var(--blue)" }} stopOpacity="0" />
              </linearGradient>
              <linearGradient id="atlFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" style={{ stopColor: "var(--orange)" }} stopOpacity="0.2" />
                <stop offset="100%" style={{ stopColor: "var(--orange)" }} stopOpacity="0" />
              </linearGradient>
            </defs>

            {pmc.gridLines.map((g) => (
              <line
                key={g.y}
                x1="0"
                y1={g.y}
                x2="700"
                y2={g.y}
                style={{ stroke: "var(--line)" }}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
              />
            ))}

            <path d={pmc.atlArea} fill="url(#atlFill)" />
            <path d={pmc.ctlArea} fill="url(#ctlFill)" />
            <path
              d={pmc.atlLine}
              fill="none"
              style={{ stroke: "var(--orange)" }}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            <path
              d={pmc.ctlLine}
              fill="none"
              style={{ stroke: "var(--blue)" }}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
            <line
              x1={pmc.selX}
              y1="0"
              x2={pmc.selX}
              y2="196"
              style={{ stroke: "var(--grey)" }}
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={pmc.selX}
              cy={pmc.selCtlY}
              r="4"
              style={{ fill: "var(--card)", stroke: "var(--blue)" }}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={pmc.selX}
              cy={pmc.selAtlY}
              r="4"
              style={{ fill: "var(--card)", stroke: "var(--orange)" }}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
          </svg>

          <div
            {...scrub}
            style={{ position: "absolute", inset: "0 0 0 40px", touchAction: "none", cursor: "crosshair" }}
          >
            <div
              style={mono(9.5, {
                position: "absolute",
                top: 0,
                left: pmc.selPct,
                transform: "translateX(-50%)",
                padding: "3px 8px",
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

        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          <span style={SMALL_LABEL}>{pmc.barsLabel}</span>
          <div
            style={{
              position: "relative",
              display: "flex",
              alignItems: "flex-end",
              gap: 3,
              height: 68,
              paddingLeft: 40,
            }}
          >
            <span
              style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}
            >
              {pmc.barMax}
            </span>
            <span
              style={mono(9, {
                position: "absolute",
                left: 0,
                color: "var(--faint)",
                whiteSpace: "nowrap",
                bottom: -1,
              })}
            >
              0
            </span>
            {pmc.bars.map((b, i) => (
              <div
                key={i}
                onMouseEnter={() => onSelectDay(b.dayIndex)}
                onClick={() => onSelectDay(b.dayIndex)}
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "flex-end",
                  height: "100%",
                  cursor: "pointer",
                  gap: 5,
                }}
              >
                <div
                  style={{
                    height: b.barH,
                    borderRadius: 4,
                    background: b.barColor,
                    transition: "height .5s cubic-bezier(.22,1,.36,1),opacity .2s",
                    opacity: b.barOpacity,
                    minHeight: 3,
                  }}
                />
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 3, paddingLeft: 40 }}>
            {pmc.bars.map((b, i) => (
              <span key={i} style={{ flex: 1, position: "relative", overflow: "visible", height: 12 }}>
                <span
                  style={mono(9, {
                    position: "absolute",
                    left: "50%",
                    top: 0,
                    transform: "translateX(-50%)",
                    color: b.tickColor,
                    whiteSpace: "nowrap",
                  })}
                >
                  {b.tick}
                </span>
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
  color,
}: {
  label: string;
  value: string;
  dot?: string;
  color?: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={{ ...SMALL_LABEL, display: "flex", alignItems: "center", gap: 6 }}>
        {dot && <span style={{ width: 8, height: 8, borderRadius: 2, background: dot }} />}
        {label}
      </span>
      <span style={mono(20, color ? { color } : {})}>{value}</span>
    </div>
  );
}
