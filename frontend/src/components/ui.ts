import type { CSSProperties } from "react";

/** Popisky v designu jsou vždy JetBrains Mono; tohle šetří opakování. */
export function mono(size: number, extra: CSSProperties = {}): CSSProperties {
  return { fontFamily: "'JetBrains Mono',monospace", fontSize: size, ...extra };
}

export const CARD: CSSProperties = {
  border: "1px solid var(--line)",
  borderRadius: 22,
  background: "var(--card)",
  display: "flex",
  flexDirection: "column",
};

export const SECTION_LABEL: CSSProperties = mono(10.5, {
  letterSpacing: ".16em",
  textTransform: "uppercase",
  color: "var(--mut)",
});

export const CARD_LABEL: CSSProperties = mono(10, {
  letterSpacing: ".12em",
  textTransform: "uppercase",
  color: "var(--mut)",
});

export const SMALL_LABEL: CSSProperties = mono(9.5, {
  letterSpacing: ".12em",
  textTransform: "uppercase",
  color: "var(--mut)",
});

export const NOTE: CSSProperties = {
  margin: 0,
  fontSize: 11.5,
  lineHeight: 1.5,
  color: "var(--faint)",
  textWrap: "pretty",
};
