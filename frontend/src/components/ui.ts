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

/** Vnější karta designu 2.0 – --line / 22px / --card / padding 24×28. */
export const CARD_SURFACE: CSSProperties = {
  border: "1px solid var(--line)",
  borderRadius: 22,
  background: "var(--card)",
  padding: "24px 28px",
  display: "flex",
  flexDirection: "column",
  gap: 16,
};

/** Vnořená karta – --line2 / 18px / --card2 / padding 20. */
export const CARD_INSET: CSSProperties = {
  border: "1px solid var(--line2)",
  borderRadius: 18,
  background: "var(--card2)",
  padding: 20,
  display: "flex",
  flexDirection: "column",
  gap: 12,
};

/** Malý statistický box (2.0) – 18px radius, padding 20. */
export const STAT_BOX: CSSProperties = {
  border: "1px solid var(--line)",
  borderRadius: 18,
  background: "var(--card)",
  padding: 20,
  display: "flex",
  flexDirection: "column",
  gap: 6,
};

/** Eyebrow popisek nad nadpisem sekce/karty. */
export const EYEBROW: CSSProperties = mono(10, {
  letterSpacing: ".16em",
  textTransform: "uppercase",
  color: "var(--mut)",
});

/** Velké „hero" číslo (mono, tabular). */
export function heroNum(size: number, extra: CSSProperties = {}): CSSProperties {
  return mono(size, {
    fontWeight: 500,
    letterSpacing: "-.02em",
    fontVariantNumeric: "tabular-nums",
    ...extra,
  });
}

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
