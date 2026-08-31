import type { HrBlocksView } from "../derive/hrblocks";
import { mono } from "./ui";

const INFO_ICON = (
  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="var(--mut)" strokeWidth="1.7">
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" strokeLinecap="round" />
  </svg>
);

const LINK_ICON = (
  <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M7 17 17 7M9 7h8v8" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

/**
 * Blok „Souvislé bloky · nad ZX" v mřížce Kvality tréninku – přesně dle
 * Trénink.dc.html: eyebrow + info, pillToggleGroup tolerance, filtr
 * „jen kompletní data", pak stav hasData / isEmpty / isExcluded.
 */
export function BlockPanel({
  title,
  view,
  barColor,
  tolerance,
  onPickTolerance,
  completeOnly,
  onToggleComplete,
  note,
  onOpenActivity,
  activityId,
}: {
  title: string;
  view: HrBlocksView;
  barColor: string;
  tolerance: number;
  onPickTolerance: (v: number) => void;
  completeOnly: boolean;
  onToggleComplete: (v: boolean) => void;
  note: string;
  onOpenActivity: (id: string) => void;
  activityId: string | null;
}) {
  const excluded = view.emptyNote != null && /neúplná data|filtr/i.test(view.emptyNote);
  const isEmpty = view.emptyNote != null && !excluded;
  const hasData = view.hasData;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" })}>{title}</span>
        {INFO_ICON}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: 3, padding: 3, borderRadius: 999, background: "var(--track)" }}>
          {[0, 15].map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => onPickTolerance(v)}
              style={mono(10, {
                appearance: "none",
                border: "none",
                cursor: "pointer",
                padding: "6px 13px",
                borderRadius: 999,
                letterSpacing: ".06em",
                whiteSpace: "nowrap",
                transition: "background .18s,color .18s",
                background: tolerance === v ? "var(--card2)" : "transparent",
                color: tolerance === v ? barColor : "var(--mut)",
              })}
            >
              {v} s
            </button>
          ))}
        </div>
        <button
          type="button"
          onClick={() => onToggleComplete(!completeOnly)}
          style={mono(10, {
            display: "flex",
            alignItems: "center",
            gap: 7,
            appearance: "none",
            cursor: "pointer",
            padding: "7px 12px",
            borderRadius: 999,
            letterSpacing: ".06em",
            whiteSpace: "nowrap",
            transition: "border-color .18s,color .18s",
            border: `1px solid ${completeOnly ? "var(--line2)" : "var(--line)"}`,
            background: "var(--track)",
            color: completeOnly ? "var(--fg2)" : "var(--mut)",
          })}
        >
          <span style={{ width: 9, height: 9, borderRadius: 3, background: completeOnly ? "var(--ok)" : "var(--grey)" }} />
          jen kompletní data
        </button>
      </div>

      {hasData && (
        <>
          <span style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span style={mono(30, { lineHeight: 1, letterSpacing: "-.035em", color: "var(--fg)", fontVariantNumeric: "tabular-nums" })}>
              {view.headline}
            </span>
            <span style={mono(13, { color: "var(--mut)" })}>min</span>
          </span>

          <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={mono(11, { color: "var(--mut)" })}>Nejdelší úsek, {view.headlineNote}</span>
            {activityId && (
              <button
                type="button"
                onClick={() => onOpenActivity(activityId)}
                className="hover-fg"
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: 20,
                  height: 20,
                  borderRadius: 999,
                  border: "1px solid var(--line2)",
                  background: "none",
                  color: "var(--mut)",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                {LINK_ICON}
              </button>
            )}
          </span>

          <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
            {view.hist.map((b) => (
              <div key={b.bucket} style={{ display: "grid", gridTemplateColumns: "82px 34px 1fr 58px", gap: 10, alignItems: "center" }}>
                <span style={mono(10, { color: "var(--mut)", whiteSpace: "nowrap", textAlign: "right" })}>{b.bucket}</span>
                <span style={mono(10, { color: "var(--faint)", textAlign: "right" })}>{b.count}</span>
                <span style={{ display: "block", height: 7, borderRadius: 999, background: "var(--track)", overflow: "hidden" }}>
                  <span style={{ display: "block", height: "100%", width: b.w, borderRadius: 999, background: barColor, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
                </span>
                <span style={mono(10, { color: "var(--fg2)", textAlign: "right", whiteSpace: "nowrap" })}>{b.time}</span>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
            <span style={mono(11, { color: "var(--fg2)" })}>{view.totals}</span>
            <span style={mono(10, { color: "var(--faint)" })}>{view.ridesNote}</span>
          </div>
        </>
      )}

      {isEmpty && (
        <>
          <span style={mono(30, { lineHeight: 1, color: "var(--faint)" })}>—</span>
          <span style={mono(10, { color: "var(--faint)" })}>{view.ridesNote || view.emptyNote}</span>
        </>
      )}

      {excluded && (
        <>
          <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <span style={{ width: 8, height: 8, borderRadius: 999, background: "var(--warn)" }} />
            <span style={mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--warn)" })}>Vyloučeno filtrem</span>
          </div>
          <button
            type="button"
            onClick={() => onToggleComplete(false)}
            style={mono(11, {
              alignSelf: "flex-start",
              appearance: "none",
              cursor: "pointer",
              padding: "9px 15px",
              borderRadius: 999,
              border: "1px solid var(--line2)",
              background: "var(--track)",
              color: "var(--fg)",
              letterSpacing: ".04em",
            })}
          >
            Vypnout filtr „jen kompletní data“
          </button>
        </>
      )}

      <p style={{ margin: 0, fontSize: 11, lineHeight: 1.5, color: "var(--faint)", textWrap: "pretty" }}>{note}</p>
    </div>
  );
}
