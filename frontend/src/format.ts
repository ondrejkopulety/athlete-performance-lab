/** Formátování čísel a dat – převzato z designu, včetně české desetinné čárky. */

/** Minuty jako "2:07" nebo "43 min". */
export function fmtShort(m: number): string {
  const h = Math.floor(m / 60);
  const r = Math.round(m % 60);
  return h ? `${h}:${String(r).padStart(2, "0")}` : `${r} min`;
}

/** Minuty jako "2 h 07 min". */
export function fmtMin(m: number): string {
  const h = Math.floor(m / 60);
  const r = Math.round(m % 60);
  return h ? `${h} h ${String(r).padStart(2, "0")} min` : `${r} min`;
}

/** "2026-08-07" → "7.8." */
export function tick(d: string): string {
  const p = d.split("-");
  return `${Number(p[2])}.${Number(p[1])}.`;
}

const MONTHS = ["led", "úno", "bře", "dub", "kvě", "čvn", "čvc", "srp", "zář", "říj", "lis", "pro"];

/** Popisek měsíce; v lednu se místo názvu ukáže rok, ať je zlom vidět. */
export function mon(d: string, withYear: boolean): string {
  const p = d.split("-");
  const m = MONTHS[Number(p[1]) - 1];
  return withYear ? (p[1] === "01" ? p[0] : `${m} ${p[0].slice(2)}`) : m;
}

/** Desetinná čárka místo tečky. */
export function dec(value: number, digits = 1): string {
  return value.toFixed(digits).replace(".", ",");
}

/** Znaménko minus jako typografická pomlčka (jako v designu). */
export function signed(value: number, digits = 1): string {
  return (value > 0 ? "+" : "") + value.toFixed(digits).replace(".", ",").replace("-", "−");
}

export function czDate(iso: string, opts: Intl.DateTimeFormatOptions): string {
  return new Date(`${iso}T12:00:00`).toLocaleDateString("cs-CZ", opts);
}

/** "5. 8." – s rokem, pokud období přesahuje přes Silvestra. */
export function shortDate(d: string, withYear: boolean): string {
  const p = d.split("-");
  return `${Number(p[2])}. ${Number(p[1])}.${withYear ? ` ${p[0]}` : ""}`;
}

/** Skloňování: 1 jízda, 2–4 jízdy, 5+ jízd. */
export function rideCount(n: number): string {
  return `${n} ${n === 1 ? "jízda" : n < 5 ? "jízdy" : "jízd"}`;
}
