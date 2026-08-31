import type { CSSProperties, ReactNode } from "react";

import { mono } from "./ui";

/**
 * Společný obal obrazovky z designu 2.0: vnější `data-theme` vrstva
 * (`padding:28px 20px 96px`) a vnitřní centrovaný sloupec (`max-width:1120px;
 * gap:20px`). Spodní odsazení 96px nechává místo pro fixní nav bar.
 */
export function Shell({
  children,
  screen,
  hidden,
  maxWidth = 1120,
  gap = 20,
}: {
  children: ReactNode;
  /** Nastaví `data-screen` (kvůli screen-specific paletě, viz styles.css). */
  screen?: string;
  /** `display:none` – obrazovka zůstává mountovaná kvůli zachování stavu. */
  hidden?: boolean;
  maxWidth?: number;
  gap?: number;
}) {
  return (
    <div
      data-screen={screen}
      style={{
        display: hidden ? "none" : undefined,
        minHeight: "100vh",
        background: "var(--bg)",
        color: "var(--fg)",
        fontFamily: "var(--font-sans),'DM Sans',system-ui,sans-serif",
        padding: "28px 20px 96px",
        transition: "background .35s ease,color .35s ease",
      }}
    >
      <div
        style={{
          maxWidth,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap,
        }}
      >
        {children}
      </div>
    </div>
  );
}

/** Hlavička obrazovky: eyebrow (mono, prostrkaná) + `<h1>` + volitelný pravý slot. */
export function ScreenHeader({
  eyebrow,
  title,
  right,
  back,
  titleSize = 22,
  align = "flex-end",
}: {
  eyebrow?: string;
  title: string;
  right?: ReactNode;
  back?: () => void;
  titleSize?: number;
  align?: CSSProperties["alignItems"];
}) {
  return (
    <header
      style={{
        display: "flex",
        alignItems: align,
        justifyContent: "space-between",
        gap: 16,
        flexWrap: "wrap",
        padding: "2px 2px 6px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
        {back && (
          <button
            type="button"
            onClick={back}
            aria-label="Zpět"
            className="hover-fg"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 34,
              height: 34,
              flex: "none",
              borderRadius: "50%",
              border: "1px solid var(--line)",
              background: "var(--card)",
              color: "var(--mut)",
              cursor: "pointer",
            }}
          >
            <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M15 18l-6-6 6-6" />
            </svg>
          </button>
        )}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {eyebrow && (
            <span
              style={mono(11, {
                letterSpacing: ".14em",
                textTransform: "uppercase",
                color: "var(--mut)",
              })}
            >
              {eyebrow}
            </span>
          )}
          <h1 style={{ margin: 0, fontSize: titleSize, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>
            {title}
          </h1>
        </div>
      </div>
      {right && <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>{right}</div>}
    </header>
  );
}
