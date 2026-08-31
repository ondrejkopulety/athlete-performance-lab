import { useEffect, useMemo, useState } from "react";

import { fetchActivityHistory, type ActivityListRow } from "./api";
import { Shell } from "./components/Shell";
import { SportIcon } from "./components/SportIcon";
import { StateScreen } from "./components/StateScreen";
import { mono } from "./components/ui";
import { buildMonth, buildWeek, mondayOf } from "./derive/calendar";
import { THEMES, type ThemeName } from "./theme";

const HISTORY_START_YEAR = 2021;

export function Aktivity({
  theme,
  hidden,
  onOpenActivity,
}: {
  theme: ThemeName;
  hidden?: boolean;
  onOpenActivity: (id: string) => void;
}) {
  const T = THEMES[theme];
  const light = theme === "light";
  const [rows, setRows] = useState<ActivityListRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const today = new Date().toISOString().slice(0, 10);
  const [view, setView] = useState(() => {
    const d = new Date();
    return { year: d.getFullYear(), month: d.getMonth() };
  });
  const [selected, setSelected] = useState<string>(today);

  useEffect(() => {
    if (hidden || rows) return;
    const ctrl = new AbortController();
    fetchActivityHistory(HISTORY_START_YEAR, new Date().getFullYear(), ctrl.signal)
      .then(setRows)
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      });
    return () => ctrl.abort();
  }, [hidden, rows]);

  const month = useMemo(
    () => (rows ? buildMonth(view.year, view.month, rows, selected, today, T) : null),
    [rows, view, selected, today, T],
  );
  const week = useMemo(
    () => (rows ? buildWeek(mondayOf(selected), rows, T, light) : null),
    [rows, selected, T, light],
  );

  if (error) return <StateScreen title="Aktivity se nepodařilo načíst" detail={error} />;

  const stepMonth = (delta: number) => {
    setView((v) => {
      const m = v.month + delta;
      return { year: v.year + Math.floor(m / 12), month: ((m % 12) + 12) % 12 };
    });
  };
  const goToday = () => {
    const d = new Date();
    setView({ year: d.getFullYear(), month: d.getMonth() });
    setSelected(today);
  };

  return (
    <Shell hidden={hidden}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap", padding: "2px 2px 6px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>AKTIVITY</span>
          <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>Kalendář aktivit</h1>
        </div>
      </header>

      {/* Kalendář */}
      <section style={CARD}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12 }}>
          <span style={EYEBROW}>{month?.label ?? "…"}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <RoundBtn label="Předchozí měsíc" onClick={() => stepMonth(-1)} d="M15 6l-6 6 6 6" />
            <button
              type="button"
              onClick={goToday}
              style={mono(10, { padding: "6px 12px", borderRadius: 999, border: "1px solid var(--line2)", background: "var(--track)", color: "var(--mut)", textTransform: "uppercase", letterSpacing: ".06em", cursor: "pointer" })}
            >
              Dnes
            </button>
            <RoundBtn label="Další měsíc" onClick={() => stepMonth(1)} d="M9 6l6 6-6 6" />
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 6, padding: "0 2px" }}>
          {(month?.weekdayLabels ?? WD_FALLBACK).map((w) => (
            <span key={w} style={mono(9, { textAlign: "center", letterSpacing: ".08em", textTransform: "uppercase", color: "var(--faint)" })}>{w}</span>
          ))}
        </div>

        {!month ? (
          <p style={{ ...NOTE, padding: "24px 0", textAlign: "center" }}>Načítám…</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {month.weeks.map((wk) => (
              <div key={wk.key} style={{ display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 6, borderRadius: 14, background: wk.bg, padding: 4, transition: "background .2s ease" }}>
                {wk.days.map((d) => (
                  <button
                    key={d.iso}
                    type="button"
                    onClick={() => setSelected(d.iso)}
                    style={{
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 3,
                      minHeight: 46,
                      padding: "5px 0",
                      borderRadius: 10,
                      border: `1px solid ${d.isSelected ? "var(--line2)" : d.isToday ? T.line : "transparent"}`,
                      background: d.isSelected ? "var(--card2)" : "none",
                      cursor: "pointer",
                      color: d.inMonth ? "var(--fg)" : "var(--faint)",
                      fontFamily: "var(--font-sans),'DM Sans',sans-serif",
                      opacity: d.inMonth ? 1 : 0.4,
                      transition: "background .15s,border-color .15s",
                    }}
                  >
                    <span style={mono(11, { lineHeight: 1 })}>{d.num}</span>
                    {d.sport && <SportIcon sport={d.sport} size={18} color={d.iconColor} />}
                    {d.durLabel && <span style={mono(8, { color: "var(--faint)" })}>{d.durLabel}</span>}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Vybraný týden */}
      <section style={{ ...CARD, gap: 14 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <span style={EYEBROW}>{week?.title ?? "VYBRANÝ TÝDEN"}</span>
          <span style={mono(11, { color: "var(--faint)" })}>{week?.summary ?? ""}</span>
        </div>

        {!week || week.cards.length === 0 ? (
          <div style={{ padding: "32px 24px", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 8, textAlign: "center" }}>
            <span style={mono(11, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--faint)" })}>Žádné aktivity</span>
            <p style={{ margin: 0, maxWidth: 280, fontSize: 13, lineHeight: 1.6, color: "var(--mut)" }}>
              V tomto týdnu nejsou zaznamenané žádné aktivity.
            </p>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(250px,1fr))", gap: 12 }}>
            {week.cards.map((c) => (
              <button
                key={c.id}
                type="button"
                onClick={() => onOpenActivity(c.id)}
                className="hover-lift"
                style={{ border: "1px solid var(--line)", borderRadius: 18, background: "var(--card2)", padding: 20, color: "inherit", textAlign: "left", display: "flex", flexDirection: "column", gap: 14, cursor: "pointer" }}
              >
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                  <span style={mono(10, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>{c.date}</span>
                  <span style={mono(10, { display: "flex", alignItems: "center", gap: 6, padding: "4px 8px", borderRadius: 999, background: c.tagBg, color: c.tagColor })}>
                    <SportIcon sport={c.sport} size={14} color="currentColor" />
                    {c.tag}
                  </span>
                </div>

                {c.showDistance && (
                  <div style={{ display: "flex", alignItems: "flex-end", gap: 6 }}>
                    <span style={mono(26, { lineHeight: 1, letterSpacing: "-.02em" })}>{c.distance}</span>
                    <span style={mono(12, { color: "var(--faint)", paddingBottom: 2 })}>km</span>
                  </div>
                )}

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(70px,1fr))", gap: 8 }}>
                  {c.cells.map((m) => (
                    <div key={m.label} style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                      <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>{m.label}</span>
                      <span style={mono(13, { color: m.color, whiteSpace: "nowrap" })}>{m.value}</span>
                    </div>
                  ))}
                </div>

                {c.showZones && (
                  <div style={{ display: "flex", height: 5, borderRadius: 999, overflow: "hidden", background: "var(--track)", gap: 2 }}>
                    {c.zones.map((z) => (
                      <div key={z.name} style={{ width: z.w, background: z.color }} />
                    ))}
                  </div>
                )}
              </button>
            ))}
          </div>
        )}
      </section>
    </Shell>
  );
}

const CARD = { border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "24px 28px", display: "flex", flexDirection: "column", gap: 16 } as const;
const EYEBROW = mono(10, { letterSpacing: ".16em", textTransform: "uppercase", color: "var(--mut)" });
const NOTE = { margin: 0, fontSize: 13, lineHeight: 1.6, color: "var(--mut)" } as const;
const WD_FALLBACK = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"];

function RoundBtn({ label, onClick, d }: { label: string; onClick: () => void; d: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 26, height: 26, borderRadius: 999, border: "1px solid var(--line2)", background: "var(--track)", color: "var(--fg2)", cursor: "pointer" }}
    >
      <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
        <path d={d} />
      </svg>
    </button>
  );
}
