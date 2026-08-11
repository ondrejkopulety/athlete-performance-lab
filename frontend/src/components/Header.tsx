import type { Today } from "../api";
import { czDate } from "../format";
import type { ThemeName } from "../theme";
import { mono } from "./ui";

const SUN =
  "M12 3v1.5M12 19.5V21M4.2 4.2l1.1 1.1M18.7 18.7l1.1 1.1M3 12h1.5M19.5 12H21M4.2 19.8l1.1-1.1M18.7 5.3l1.1-1.1M12 7.5a4.5 4.5 0 100 9 4.5 4.5 0 000-9z";
const MOON = "M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z";

export function Header({
  today,
  readiness,
  statusDot,
  theme,
  onToggleTheme,
}: {
  today: Today | null;
  readiness: number | null;
  statusDot: string;
  theme: ThemeName;
  onToggleTheme: () => void;
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

      <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
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

        <button
          type="button"
          onClick={onToggleTheme}
          aria-label="Přepnout motiv"
          className="hover-fg"
          style={mono(10.5, {
            display: "flex",
            alignItems: "center",
            gap: 7,
            padding: "6px 12px 6px 9px",
            border: "1px solid var(--line2)",
            borderRadius: 999,
            background: "var(--card)",
            color: "var(--mut)",
            letterSpacing: ".08em",
            textTransform: "uppercase",
            cursor: "pointer",
          })}
        >
          <svg
            viewBox="0 0 24 24"
            width="14"
            height="14"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
          >
            <path d={theme === "dark" ? SUN : MOON} />
          </svg>
          {theme === "dark" ? "Světlý" : "Tmavý"}
        </button>
      </div>
    </header>
  );
}
