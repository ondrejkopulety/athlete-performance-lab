import { mono } from "./ui";

/** Načítání, chyba nebo prázdná databáze – design tenhle stav neřešil,
 *  protože měl data napevno v souboru. */
export function StateScreen({ title, detail }: { title: string; detail?: string }) {
  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg)",
        color: "var(--fg)",
        fontFamily: "'DM Sans',system-ui,sans-serif",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: 10,
        padding: 24,
        textAlign: "center",
      }}
    >
      <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>
        Readiness Dashboard
      </span>
      <h1 style={{ margin: 0, fontSize: 20, fontWeight: 500, letterSpacing: "-.02em" }}>{title}</h1>
      {detail && (
        <p style={{ margin: 0, maxWidth: 460, fontSize: 13, lineHeight: 1.55, color: "var(--mut)" }}>
          {detail}
        </p>
      )}
    </div>
  );
}
