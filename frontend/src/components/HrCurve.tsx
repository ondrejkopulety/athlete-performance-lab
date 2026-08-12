import type { HrCurveView } from "../derive/hrcurve";
import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { CompleteOnlyToggle } from "./CompleteOnlyToggle";
import { RangeTabs } from "./RangeTabs";
import { CARD, CARD_LABEL, mono, NOTE, SECTION_LABEL } from "./ui";

/**
 * Tepová křivka: maximální průměrný tep proti délce okna, logaritmická osa x.
 * Referenční období se kreslí čárkovaně pod hlavní křivkou.
 *
 * Bod, který v období nevyšel, se nekreslí a čára se přeruší – tabulka pod
 * grafem u něj ukáže pomlčku. Nula by tvrdila, že se hodina odjela na nule.
 */
export function HrCurve({
  view,
  ranges,
  range,
  rangeLabel,
  onPickRange,
  completeOnly,
  onToggleComplete,
  compare,
  onToggleCompare,
  loading,
  error,
  theme,
}: {
  view: HrCurveView;
  ranges: RangeOption[];
  range: Range;
  rangeLabel: string;
  onPickRange: (value: Range) => void;
  completeOnly: boolean;
  onToggleComplete: (value: boolean) => void;
  compare: "prev" | "year";
  onToggleCompare: (value: "prev" | "year") => void;
  loading: boolean;
  error: string | null;
  theme: Theme;
}) {
  const effort = view.lastEffort;

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
        <span style={SECTION_LABEL}>Tepová křivka</span>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={mono(11, { color: "var(--faint)" })}>{rangeLabel}</span>
          <CompleteOnlyToggle value={completeOnly} onChange={onToggleComplete} theme={theme} />
          <RangeTabs options={ranges} value={range} onPick={onPickRange} theme={theme} />
        </div>
      </div>

      {/* Kvůli tomuhle řádku ten panel hlavně je. */}
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <span style={CARD_LABEL}>Poslední maximální výkon</span>
          <span
            style={mono(15, {
              color: effort?.stale ? theme.warn : "var(--fg)",
              fontWeight: effort?.stale ? 600 : 400,
            })}
          >
            {effort ? effort.text : "–"}
          </span>
        </div>
        <span style={mono(10.5, { color: "var(--faint)" })}>{view.ridesNote}</span>
      </div>

      {error ? (
        <p style={{ ...NOTE, padding: "18px 0", color: theme.bad }}>
          Panel se nepodařilo načíst: {error}
        </p>
      ) : view.emptyNote ? (
        <p style={{ ...NOTE, padding: "18px 0" }}>{view.emptyNote}</p>
      ) : (
        // SVG nese jen čáry; popisky a body jsou HTML, aby je roztažený
        // viewBox nedeformoval (stejně jako u ostatních grafů dashboardu).
        <div
          style={{
            position: "relative",
            height: 190,
            marginLeft: 30,
            opacity: loading ? 0.45 : 1,
            transition: "opacity .2s",
          }}
        >
          {view.yTicks.map((t) => (
            <div
              key={t.label}
              style={{
                position: "absolute",
                left: 0,
                right: 0,
                top: t.top,
                borderTop: `1px solid ${theme.line}`,
              }}
            >
              <span
                style={mono(9, {
                  position: "absolute",
                  left: -30,
                  top: -6,
                  color: "var(--faint)",
                })}
              >
                {t.label}
              </span>
            </div>
          ))}

          <svg
            viewBox={`0 0 ${view.width} ${view.height}`}
            preserveAspectRatio="none"
            style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}
          >
            {view.reference?.paths.map((d, i) => (
              <path
                key={`ref-${i}`}
                d={d}
                fill="none"
                stroke={theme.grey}
                strokeWidth={1.5}
                strokeDasharray="4 3"
                vectorEffect="non-scaling-stroke"
              />
            ))}

            {view.period.paths.map((d, i) => (
              <path
                key={`cur-${i}`}
                d={d}
                fill="none"
                stroke={theme.orange}
                strokeWidth={2}
                strokeLinejoin="round"
                vectorEffect="non-scaling-stroke"
              />
            ))}
          </svg>

          {view.period.dots.map((p) => (
            <span
              key={p.left}
              title={`${Math.round(p.hr)} tepů`}
              style={{
                position: "absolute",
                left: p.left,
                top: p.top,
                width: 5,
                height: 5,
                marginLeft: -2.5,
                marginTop: -2.5,
                borderRadius: "50%",
                background: theme.orange,
              }}
            />
          ))}

          {view.xTicks.map((t) => (
            <span
              key={t.label}
              style={mono(9, {
                position: "absolute",
                left: t.left,
                bottom: -4,
                transform: "translateX(-50%)",
                color: "var(--faint)",
                whiteSpace: "nowrap",
              })}
            >
              {t.label}
            </span>
          ))}
        </div>
      )}

      {view.reference && view.hasData && (
        <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <Legend color={theme.orange} label="zvolené období" dashed={false} />
          <Legend color={theme.grey} label={view.referenceLabel} dashed />
          <button
            type="button"
            onClick={() => onToggleCompare(compare === "prev" ? "year" : "prev")}
            style={mono(10, {
              marginLeft: "auto",
              border: `1px solid ${theme.line}`,
              background: "transparent",
              color: theme.mut,
              borderRadius: 999,
              padding: "4px 11px",
              cursor: "pointer",
            })}
          >
            {compare === "prev" ? "srovnat s loňskem" : "srovnat s předchozím obdobím"}
          </button>
        </div>
      )}

      {view.hasData && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(210px,1fr))",
            gap: "6px 22px",
          }}
        >
          {view.rows.map((r) => (
            <div
              key={r.duration}
              style={{ display: "flex", alignItems: "baseline", gap: 8, minWidth: 0 }}
            >
              <span style={mono(10, { color: "var(--mut)", width: 46 })}>{r.duration}</span>
              <span style={mono(13, { width: 34, textAlign: "right" })}>{r.value}</span>
              <span
                style={mono(10, {
                  color: "var(--faint)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                })}
              >
                {r.source}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function Legend({ color, label, dashed }: { color: string; label: string; dashed: boolean }) {
  return (
    <span style={mono(10, { display: "flex", alignItems: "center", gap: 7, color: "var(--mut)" })}>
      <span
        style={{
          width: 16,
          height: 0,
          borderTop: `2px ${dashed ? "dashed" : "solid"} ${color}`,
        }}
      />
      {label}
    </span>
  );
}
