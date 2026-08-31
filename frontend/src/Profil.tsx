import { Shell } from "./components/Shell";
import { mono } from "./components/ui";
import type { ThemeName } from "./theme";

const SUN =
  "M12 3v1.5M12 19.5V21M4.2 4.2l1.1 1.1M18.7 18.7l1.1 1.1M3 12h1.5M19.5 12H21M4.2 19.8l1.1-1.1M18.7 5.3l1.1-1.1M12 7.5a4.5 4.5 0 100 9 4.5 4.5 0 000-9z";
const MOON = "M20 14.5A8.5 8.5 0 019.5 4a8.5 8.5 0 1010.5 10.5z";

/** Profil – 1:1 s Profil.dc.html: karta „Vzhled" (přepínač motivu) + „Připravujeme". */
export function Profil({
  theme,
  hidden,
  onToggleTheme,
}: {
  theme: ThemeName;
  hidden?: boolean;
  onToggleTheme: () => void;
}) {
  const isDark = theme === "dark";

  return (
    <Shell hidden={hidden}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap", padding: "2px 2px 6px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>PROFIL</span>
          <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>Nastavení</h1>
        </div>
      </header>

      <section style={{ border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "20px 20px", display: "flex", flexDirection: "column", gap: 14 }}>
        <span style={mono(11, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--faint)" })}>Vzhled</span>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="var(--mut)" strokeWidth="1.8" strokeLinecap="round">
              <path d={isDark ? SUN : MOON} />
            </svg>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              <span style={{ fontSize: 14, color: "var(--fg)" }}>Motiv</span>
              <span style={{ fontSize: 12, color: "var(--mut)" }}>{isDark ? "Tmavý" : "Světlý"}</span>
            </div>
          </div>
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label="Přepnout motiv"
            className="hover-fg"
            style={mono(10, {
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "8px 14px",
              border: "1px solid var(--line2)",
              borderRadius: 999,
              background: "var(--track)",
              color: "var(--mut)",
              letterSpacing: ".08em",
              textTransform: "uppercase",
              cursor: "pointer",
              transition: ".25s",
              whiteSpace: "nowrap",
            })}
          >
            Přepnout na {isDark ? "světlý" : "tmavý"}
          </button>
        </div>
      </section>

      <section
        style={{
          border: "1px solid var(--line)",
          borderRadius: 22,
          background: "var(--card)",
          padding: "48px 28px",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 10,
          minHeight: 320,
          textAlign: "center",
        }}
      >
        <span style={mono(11, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--faint)" })}>Připravujeme</span>
        <p style={{ margin: 0, maxWidth: 360, fontSize: 14, lineHeight: 1.6, color: "var(--mut)" }}>
          Nastavení účtu, propojená zařízení a preference tréninku budou tady.
        </p>
      </section>
    </Shell>
  );
}
