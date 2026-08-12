import type { RideCard } from "../derive/rides";
import type { Theme } from "../theme";
import { CARD, mono, NOTE, SECTION_LABEL } from "./ui";

export function Rides({
  rides,
  summary,
  openId,
  onToggle,
  theme,
}: {
  rides: RideCard[];
  summary: string;
  openId: string | null;
  onToggle: (id: string) => void;
  theme: Theme;
}) {
  return (
    <section style={{ ...CARD, padding: "24px 26px", gap: 14 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <span style={SECTION_LABEL}>Poslední jízdy</span>
        <span style={mono(11, { color: "var(--faint)" })}>{summary}</span>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit,minmax(250px,1fr))",
          gap: 12,
        }}
      >
        {rides.map((r) => {
          const open = openId === r.id;
          return (
            <div
              key={r.id}
              onClick={() => onToggle(r.id)}
              className="hover-lift"
              style={{
                border: `1px solid ${open ? theme.lineOn : theme.line}`,
                borderRadius: 18,
                background: "var(--card2)",
                padding: 18,
                display: "flex",
                flexDirection: "column",
                gap: 13,
                cursor: "pointer",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
                <span
                  style={mono(10.5, {
                    letterSpacing: ".1em",
                    textTransform: "uppercase",
                    color: "var(--mut)",
                  })}
                >
                  {r.date}
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  {/* Odznak jen při neúplných datech. Zelený "100 % OK"
                      štítek u zbytku by byl šum – ticho znamená v pořádku. */}
                  {r.partial && (
                    <span
                      title="Tep chybí na části záznamu – rozbal kartu pro detail"
                      style={mono(9.5, {
                        display: "flex",
                        alignItems: "center",
                        gap: 4,
                        padding: "3px 7px",
                        borderRadius: 99,
                        border: `1px solid ${theme.line}`,
                        color: theme.mut,
                      })}
                    >
                      ◔ částečná data
                    </span>
                  )}
                  <span
                    style={mono(10, {
                      padding: "3px 8px",
                      borderRadius: 99,
                      background: r.tagBg,
                      color: r.tagColor,
                    })}
                  >
                    {r.tag}
                  </span>
                </span>
              </div>

              <div style={{ display: "flex", alignItems: "flex-end", gap: 6 }}>
                <span style={mono(28, { lineHeight: 1, letterSpacing: "-.02em" })}>{r.distance}</span>
                <span style={mono(12, { color: "var(--faint)", paddingBottom: 2 })}>km</span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 8 }}>
                <Cell label="Čas" value={r.duration} />
                <Cell label="Prům. tep" value={r.avgHr} color="var(--orange)" />
                <Cell label="Převýšení" value={r.ascent} />
              </div>

              <div
                style={{
                  display: "flex",
                  height: 5,
                  borderRadius: 99,
                  overflow: "hidden",
                  background: "var(--track)",
                  gap: 1.5,
                }}
              >
                {r.zones.map((z) => (
                  <div key={z.name} style={{ width: z.w, background: z.color }} />
                ))}
              </div>

              {open && (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    gap: 8,
                    paddingTop: 11,
                    borderTop: "1px solid var(--line2)",
                  }}
                >
                  {r.zones.map((z) => (
                    <div
                      key={z.name}
                      style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}
                    >
                      <span
                        style={mono(10.5, {
                          display: "flex",
                          alignItems: "center",
                          gap: 8,
                          color: "var(--mut)",
                        })}
                      >
                        <span style={{ width: 7, height: 7, borderRadius: 2, background: z.color }} />
                        {z.name}
                      </span>
                      <span style={mono(10.5, { color: "var(--mut)" })}>{z.mins}</span>
                    </div>
                  ))}
                  <div
                    style={mono(10.5, {
                      display: "flex",
                      justifyContent: "space-between",
                      color: "var(--mut)",
                    })}
                  >
                    <span>TRIMP {r.trimp}</span>
                    <span>Max HR {r.maxHr}</span>
                    <span>{r.kcal} kcal</span>
                  </div>
                  {r.coverageNote && (
                    <p
                      style={{
                        ...NOTE,
                        marginTop: 2,
                        paddingTop: 9,
                        borderTop: "1px solid var(--line2)",
                      }}
                    >
                      {r.coverageNote}
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function Cell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>
        {label}
      </span>
      <span style={mono(13.5, { whiteSpace: "nowrap", ...(color ? { color } : {}) })}>{value}</span>
    </div>
  );
}
