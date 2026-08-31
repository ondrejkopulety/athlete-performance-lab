import type { Today } from "../api";
import { czDate } from "../format";
import { mono } from "./ui";

/**
 * Hlavička Přehledu podle designu 2.0: datum (eyebrow), pozdrav a stavová
 * tečka. Přepínač motivu je teď na Profilu, práh LTHR taky – hlavička je
 * čistě informační.
 */
export function Header({
  today,
  readiness,
  statusDot,
}: {
  today: Today | null;
  readiness: number | null;
  statusDot: string;
}) {
  const statusLabel =
    readiness == null
      ? "Bez dat"
      : readiness >= 66
        ? "Nabito"
        : readiness >= 40
          ? "Střední"
          : "Regenerace";

  return (
    <header
      style={{
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "space-between",
        gap: 16,
        flexWrap: "wrap",
        padding: "2px 2px 6px",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span
          style={mono(11, {
            letterSpacing: ".14em",
            textTransform: "uppercase",
            color: "var(--mut)",
          })}
        >
          {today
            ? czDate(today.date, { weekday: "long", day: "numeric", month: "long" }).toUpperCase()
            : ""}
        </span>
        <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>
          Dobré ráno, Ondřeji
        </h1>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: statusDot,
            boxShadow: `0 0 10px ${statusDot}`,
          }}
        />
        <span
          style={mono(11, {
            letterSpacing: ".08em",
            textTransform: "uppercase",
            color: "var(--mut)",
          })}
        >
          {statusLabel}
        </span>
      </div>
    </header>
  );
}
