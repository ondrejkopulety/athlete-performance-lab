import type { Climb } from "../derive/climb";
import type { Hrr } from "../derive/hrr";
import type { Polarization, ZoneTime } from "../derive/quality";
import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { RangeTabs } from "./RangeTabs";
import { CARD, CARD_LABEL, mono, NOTE, SECTION_LABEL } from "./ui";
import { useScrub } from "./useScrub";

export function Quality({
  ranges,
  range,
  rangeLabel,
  onPickRange,
  theme,
  polarization,
  zoneTime,
  climb,
  hrr,
  onHrrScrub,
  hrrScrubbing,
  onHrrScrubbingChange,
}: {
  ranges: RangeOption[];
  range: Range;
  rangeLabel: string;
  onPickRange: (value: Range) => void;
  theme: Theme;
  polarization: Polarization;
  zoneTime: ZoneTime;
  climb: Climb;
  hrr: Hrr;
  onHrrScrub: (fraction: number) => void;
  hrrScrubbing: boolean;
  onHrrScrubbingChange: (active: boolean) => void;
}) {
  const scrub = useScrub(hrrScrubbing, onHrrScrubbingChange, onHrrScrub);
  const pct = (v: number) => v.toFixed(0);

  return (
    <section style={{ ...CARD, padding: "24px 26px", gap: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <span style={SECTION_LABEL}>Kvalita tréninku</span>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={mono(11, { color: "var(--faint)" })}>průměr · {rangeLabel}</span>
          <RangeTabs options={ranges} value={range} onPick={onPickRange} theme={theme} />
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))",
          gap: "32px 40px",
          alignItems: "start",
        }}
      >
        {/* ── Polarizace ─────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
          <span style={CARD_LABEL}>Polarizace · Z1–Z2</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
            <span style={mono(30, { lineHeight: 1, color: polarization.polColor })}>
              {polarization.hasData ? pct(polarization.low) : "–"}
            </span>
            <span style={mono(13, { color: "var(--faint)" })}>%</span>
            <span style={mono(10, { marginLeft: "auto", color: "var(--mut)" })}>cíl ≥ 75 %</span>
          </div>
          <div
            style={{
              display: "flex",
              height: 6,
              borderRadius: 99,
              overflow: "hidden",
              background: "var(--track)",
              gap: 1.5,
            }}
          >
            <div style={{ width: polarization.lowW, background: "var(--blue)" }} />
            <div style={{ width: polarization.junkW, background: "var(--warn)" }} />
            <div style={{ width: polarization.highW, background: "var(--bad)" }} />
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <Legend color="var(--blue)" text={`Z1–2 ${pct(polarization.low)} %`} />
            <Legend color="var(--warn)" text={`Z3 ${pct(polarization.junk)} %`} />
            <Legend color="var(--bad)" text={`Z4–5 ${pct(polarization.high)} %`} />
          </div>
          <p style={NOTE}>Čas ve striktní Z1–Z2 — základ vytrvalosti.</p>
        </div>

        {/* ── Nezáměrná Z3 ───────────────────────────────────────────────
            Dřív "Junk miles · Z3" = veškerý čas v Z3. Jenže Z3 v souvislém
            bloku je sweet spot trénink, ne odpad; odpad je Z3, do které se
            spadne kvůli kopci. Teď se počítá jen čas v Z3 v úsecích kratších
            než tři minuty. */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
          <span style={CARD_LABEL}>Nezáměrná Z3</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
            <span style={mono(30, { lineHeight: 1, color: polarization.unintendedColor })}>
              {polarization.unintended == null ? "–" : pct(polarization.unintended)}
            </span>
            <span style={mono(13, { color: "var(--faint)" })}>%</span>
            <span style={mono(10, { marginLeft: "auto", color: "var(--mut)" })}>cíl ≤ 8 %</span>
          </div>
          <div
            style={{
              position: "relative",
              height: 6,
              borderRadius: 99,
              background: "var(--track)",
              overflow: "visible",
            }}
          >
            <div
              style={{
                height: "100%",
                width: polarization.unintendedW30,
                borderRadius: 99,
                background: polarization.unintendedColor,
                transition: "width .8s cubic-bezier(.22,1,.36,1)",
              }}
            />
            <span
              style={{
                position: "absolute",
                top: -4,
                left: "27%",
                width: 1.5,
                height: 15,
                background: "var(--faint)",
              }}
            />
          </div>
          <div style={mono(10.5, { display: "flex", justifyContent: "space-between", gap: 8 })}>
            <span style={{ color: "var(--fg2)" }}>{polarization.unintendedTime}</span>
            <span style={{ color: "var(--faint)" }}>{polarization.unintendedShare}</span>
          </div>
          <p style={NOTE}>
            Čas v Z3 v úsecích kratších než tři minuty — šedá zóna, do které se spadne
            kvůli kopci. Souvislá Z3 je sweet spot trénink a nepočítá se sem.
          </p>
        </div>

        {/* ── Tepová regenerace ──────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
          <span style={CARD_LABEL}>Tepová regenerace · 60 s</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
            <span style={mono(30, { lineHeight: 1, color: hrr.color })}>{hrr.last}</span>
            <span style={mono(13, { color: "var(--faint)" })}>{hrr.unit}</span>
            <span style={mono(10, { marginLeft: "auto", color: hrr.trendColor })}>{hrr.trend}</span>
          </div>
          <div style={{ position: "relative", width: "100%", height: 72, paddingLeft: 30 }}>
            <span
              style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}
            >
              {hrr.hi}
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
              {hrr.lo}
            </span>
            <svg
              viewBox="0 0 320 72"
              preserveAspectRatio="none"
              style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}
            >
              <defs>
                <linearGradient id="hrrFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" style={{ stopColor: "var(--ok)" }} stopOpacity="0.2" />
                  <stop offset="100%" style={{ stopColor: "var(--ok)" }} stopOpacity="0" />
                </linearGradient>
              </defs>
              <rect
                x="0"
                y={hrr.goodY}
                width="320"
                height={hrr.goodH}
                style={{ fill: "var(--ok)", fillOpacity: 0.08 }}
              />
              <path d={hrr.area} fill="url(#hrrFill)" />
              <path
                d={hrr.line}
                fill="none"
                style={{ stroke: "var(--grey)" }}
                strokeWidth="1.5"
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
              <path
                d={hrr.trendLine}
                fill="none"
                style={{ stroke: "var(--ok)" }}
                strokeWidth="2"
                strokeLinejoin="round"
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
              <line
                x1={hrr.selX}
                y1="0"
                x2={hrr.selX}
                y2="72"
                style={{ stroke: "var(--grey)" }}
                strokeWidth="1"
                vectorEffect="non-scaling-stroke"
                opacity={hrr.crossOpacity}
              />
              <circle
                cx={hrr.lastX}
                cy={hrr.lastY}
                r="3.5"
                style={{ fill: "var(--card)", stroke: "var(--ok)" }}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            </svg>
            <div
              {...scrub}
              style={{ position: "absolute", inset: "0 0 0 30px", touchAction: "none", cursor: "crosshair" }}
            >
              <div
                style={mono(9.5, {
                  position: "absolute",
                  top: -2,
                  left: hrr.selPct,
                  transform: "translateX(-50%)",
                  padding: "3px 8px",
                  borderRadius: 999,
                  background: "var(--track)",
                  color: "var(--fg)",
                  whiteSpace: "nowrap",
                  pointerEvents: "none",
                  opacity: hrr.crossOpacity,
                  transition: "opacity .2s",
                })}
              >
                {hrr.selLabel}
              </div>
            </div>
          </div>
          <div
            style={mono(9, {
              display: "flex",
              justifyContent: "space-between",
              paddingLeft: 30,
              color: "var(--mut)",
            })}
          >
            <span>{hrr.from}</span>
            <span>klouzavý průměr · dobré ≥ 50</span>
            <span>{hrr.to}</span>
          </div>
          <p style={NOTE}>Pokles tepu první minutu po zátěži.</p>
        </div>

        {/* ── Stoupání ───────────────────────────────────────────────── */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
          <span style={CARD_LABEL}>Stoupání · VAM</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 7 }}>
            <span style={mono(30, { lineHeight: 1, color: climb.vamColor })}>
              {climb.hasData ? climb.vam : "–"}
            </span>
            <span style={mono(13, { color: "var(--faint)" })}>m/h</span>
            <span style={mono(10, { marginLeft: "auto", color: climb.vamTrendColor })}>
              {climb.vamTrend}
            </span>
          </div>
          <div
            style={{
              position: "relative",
              display: "flex",
              alignItems: "flex-end",
              gap: 2,
              height: 44,
              paddingLeft: 40,
            }}
          >
            <span
              style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}
            >
              {climb.vamMax}
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
            {climb.bars.map((b, i) => (
              <div
                key={i}
                style={{
                  flex: 1,
                  height: b.h,
                  minHeight: 3,
                  borderRadius: 3,
                  background: b.color,
                  transition: "height .6s cubic-bezier(.22,1,.36,1)",
                }}
              />
            ))}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, paddingTop: 2 }}>
            <Metric label="Do kopce" value={climb.uphill} />
            <Metric label="Prům. sklon" value={`${climb.grad} %`} />
          </div>
          <p style={NOTE}>Rychlost stoupání — odraz poměru výkon/váha.</p>
        </div>
      </div>

      <div style={{ height: 1, background: "var(--line)" }} />

      {/* ── Čas v zónách ─────────────────────────────────────────────── */}
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            justifyContent: "space-between",
            gap: 12,
            flexWrap: "wrap",
          }}
        >
          <span style={CARD_LABEL}>Čas v zónách</span>
          <span style={mono(11, { color: "var(--faint)" })}>
            {zoneTime.total} · {zoneTime.rides}
          </span>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          {zoneTime.rows.map((z) => (
            <div
              key={z.name}
              style={{
                display: "grid",
                gridTemplateColumns: "96px 1fr 74px 46px",
                gap: 12,
                alignItems: "center",
              }}
            >
              <span
                style={mono(10, {
                  display: "flex",
                  alignItems: "center",
                  gap: 7,
                  letterSpacing: ".08em",
                  textTransform: "uppercase",
                  color: "var(--mut)",
                  whiteSpace: "nowrap",
                })}
              >
                <span style={{ width: 7, height: 7, borderRadius: 2, background: z.color }} />
                {z.name}
              </span>
              <div style={{ height: 8, borderRadius: 99, background: "var(--track)", overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: z.w,
                    borderRadius: 99,
                    background: z.color,
                    transition: "width .7s cubic-bezier(.22,1,.36,1)",
                  }}
                />
              </div>
              <span style={mono(12.5, { color: "var(--fg)", textAlign: "right", whiteSpace: "nowrap" })}>
                {z.time}
              </span>
              <span style={mono(11, { color: "var(--faint)", textAlign: "right" })}>{z.pct} %</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function Legend({ color, text }: { color: string; text: string }) {
  return (
    <span style={mono(9.5, { display: "flex", alignItems: "center", gap: 5, color: "var(--mut)" })}>
      <span style={{ width: 7, height: 7, borderRadius: 2, background: color }} />
      {text}
    </span>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>
        {label}
      </span>
      <span style={mono(14, { whiteSpace: "nowrap" })}>{value}</span>
    </div>
  );
}
