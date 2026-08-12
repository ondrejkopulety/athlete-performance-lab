import type { HrBlocksView } from "../derive/hrblocks";
import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { CompleteOnlyToggle } from "./CompleteOnlyToggle";
import { RangeTabs } from "./RangeTabs";
import { CARD, CARD_LABEL, mono, NOTE, SECTION_LABEL } from "./ui";

/**
 * Souvislé bloky nad prahem: hlavní číslo je nejdelší blok za období,
 * vedle něj rozdělení délek úseků.
 *
 * Přepínač tolerance přemostění (0 / 15 s) nemění data – obě varianty jsou
 * uložené vedle sebe, mění se jen který řádek se čte.
 */
export function HrBlocks({
  view,
  ranges,
  range,
  rangeLabel,
  onPickRange,
  completeOnly,
  onToggleComplete,
  tolerance,
  onPickTolerance,
  loading,
  error,
  theme,
}: {
  view: HrBlocksView;
  ranges: RangeOption[];
  range: Range;
  rangeLabel: string;
  onPickRange: (value: Range) => void;
  completeOnly: boolean;
  onToggleComplete: (value: boolean) => void;
  tolerance: number;
  onPickTolerance: (value: number) => void;
  loading: boolean;
  error: string | null;
  theme: Theme;
}) {
  return (
    <section style={{ ...CARD, padding: "24px 26px", gap: 18 }}>
      <div
        style={{
          display: "flex",
          alignItems: "baseline",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        <span style={SECTION_LABEL}>Souvislé bloky</span>
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={mono(11, { color: "var(--faint)" })}>{rangeLabel}</span>
          <ToleranceTabs value={tolerance} onPick={onPickTolerance} theme={theme} />
          <CompleteOnlyToggle value={completeOnly} onChange={onToggleComplete} theme={theme} />
          <RangeTabs options={ranges} value={range} onPick={onPickRange} theme={theme} />
        </div>
      </div>

      {error ? (
        <p style={{ ...NOTE, padding: "8px 0", color: theme.bad }}>
          Panel se nepodařilo načíst: {error}
        </p>
      ) : view.emptyNote ? (
        <>
          <span style={mono(10.5, { color: "var(--faint)" })}>{view.ridesNote}</span>
          <p style={{ ...NOTE, padding: "8px 0" }}>{view.emptyNote}</p>
        </>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))",
            gap: "26px 40px",
            alignItems: "start",
            opacity: loading ? 0.45 : 1,
            transition: "opacity .2s",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <span style={CARD_LABEL}>Nejdelší blok {view.thresholdLabel}</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
              <span style={mono(34, { lineHeight: 1, letterSpacing: "-.02em" })}>
                {view.headline}
              </span>
              {view.trend && (
                <span style={mono(11.5, { color: view.trendColor })}>{view.trend}</span>
              )}
            </div>
            <span style={mono(10.5, { color: "var(--faint)" })}>{view.headlineNote}</span>
            <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
              <span style={mono(10.5, { color: "var(--mut)" })}>{view.totals}</span>
              <span style={mono(10.5, { color: "var(--mut)" })}>{view.intentional}</span>
              <span style={mono(10, { color: "var(--faint)" })}>{view.ridesNote}</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
            <span style={CARD_LABEL}>Rozdělení délek úseků</span>
            {view.hist.map((h) => (
              <div key={h.bucket} style={{ display: "flex", flexDirection: "column", gap: 3 }}>
                <div
                  style={mono(10.5, {
                    display: "flex",
                    justifyContent: "space-between",
                    gap: 10,
                    color: "var(--mut)",
                  })}
                >
                  <span>{h.bucket}</span>
                  {/* Počet i čas: 199 krátkých úseků vypadá jinak než
                      18 minut, které dohromady dají. */}
                  <span style={{ display: "flex", gap: 12 }}>
                    <span style={{ color: "var(--faint)" }}>{h.count}</span>
                    <span>{h.time}</span>
                  </span>
                </div>
                <div style={{ height: 5, borderRadius: 99, background: "var(--track)" }}>
                  <div
                    style={{
                      width: h.w,
                      height: "100%",
                      borderRadius: 99,
                      background: theme.orange,
                      transition: "width .6s cubic-bezier(.22,1,.36,1)",
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function ToleranceTabs({
  value,
  onPick,
  theme,
}: {
  value: number;
  onPick: (value: number) => void;
  theme: Theme;
}) {
  return (
    <div
      style={{ display: "flex", gap: 2, padding: 2, borderRadius: 999, background: "var(--track)" }}
      title="Přemostění krátkého propadu pod práh – 0 s ukazuje surovou fragmentaci"
    >
      {[0, 15].map((t) => (
        <button
          key={t}
          type="button"
          onClick={() => onPick(t)}
          style={mono(10.5, {
            padding: "6px 12px",
            borderRadius: 999,
            border: "none",
            cursor: "pointer",
            whiteSpace: "nowrap",
            background: value === t ? theme.fg : "transparent",
            color: value === t ? theme.card : theme.mut,
          })}
        >
          {t} s
        </button>
      ))}
    </div>
  );
}
