/**
 * Barvy, které se počítají v JS (barva čísla podle pásma, barvy zón).
 * Hodnoty jsou 1:1 z designu – CSS proměnné ve styles.css musí sedět
 * na tyhle, jinak se graf rozejde se zbytkem stránky.
 */

export type ThemeName = "dark" | "light";

export interface Theme {
  bg: string;
  card: string;
  line: string;
  lineOn: string;
  track: string;
  fg: string;
  fg2: string;
  mut: string;
  faint: string;
  grey: string;
  zero: string;
  ok: string;
  warn: string;
  bad: string;
  orange: string;
  orange2: string;
  blue: string;
  blue2: string;
  zones: [string, string, string, string, string];
}

export const THEMES: Record<ThemeName, Theme> = {
  dark: {
    bg: "#09090b",
    card: "#0c0c0f",
    line: "#17171a",
    lineOn: "#2e2e35",
    track: "#151518",
    fg: "#fafafa",
    fg2: "#e4e4e7",
    mut: "#8b8b93",
    faint: "#7a7a84",
    grey: "#3f3f46",
    zero: "#232327",
    ok: "#a3e635",
    warn: "#facc15",
    bad: "#f43f5e",
    orange: "#fb923c",
    orange2: "#f97316",
    blue: "#38bdf8",
    blue2: "#60a5fa",
    zones: ["#3f7a8c", "#38bdf8", "#a3e635", "#fb923c", "#f43f5e"],
  },
  light: {
    bg: "#eeedea",
    card: "#f7f6f4",
    line: "#dedcd6",
    lineOn: "#c4c2b9",
    track: "#e3e1db",
    fg: "#33322f",
    fg2: "#44433f",
    mut: "#6f6d66",
    faint: "#6f6d66",
    grey: "#c4c2b9",
    zero: "#dedcd4",
    ok: "#5f8a2a",
    warn: "#b07d1a",
    bad: "#bf5a5a",
    orange: "#c47238",
    orange2: "#bd6a2c",
    blue: "#4a83a6",
    blue2: "#5b7fb5",
    zones: ["#9dc4d6", "#4a83a6", "#5f8a2a", "#c47238", "#bf5a5a"],
  },
};

/**
 * Paleta mapy v detailu jízdy. V designu je screen-specific (mimo sdílené
 * tokeny) – v CSS je jako `[data-screen="activity"]`, tady kvůli SVG stringům
 * skládaným v JS (`derive/activityDetail.ts`).
 */
export interface MapPalette {
  mapbg: string;
  road: string;
  water: string;
  pin: string;
  zones: [string, string, string, string, string];
}

export const MAP_PALETTE: Record<ThemeName, MapPalette> = {
  dark: {
    mapbg: "#101216",
    road: "#1c2027",
    water: "#16232e",
    pin: "#0c0e12",
    zones: ["#4ade80", "#60a5fa", "#facc15", "#fb923c", "#f43f5e"],
  },
  light: {
    mapbg: "#e6e4de",
    road: "#d3d0c8",
    water: "#bcd0da",
    pin: "#f7f6f4",
    zones: ["#5f8a2a", "#4a83a6", "#b07d1a", "#c47238", "#bf5a5a"],
  },
};

export const ZONE_NAMES = [
  "Zóna 1 · Regenerace",
  "Zóna 2 · Vytrvalost",
  "Zóna 3 · Tempo",
  "Zóna 4 · Práh",
  "Zóna 5 · VO2",
];

export const ZONE_SHORT = ["Z1 Reg.", "Z2 Vytrv.", "Z3 Tempo", "Z4 Práh", "Z5 VO2"];

const STORAGE_KEY = "fit-dash-theme";

export function readStoredTheme(fallback: ThemeName = "dark"): ThemeName {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* private mode – prostě zůstane výchozí motiv */
  }
  return fallback;
}

export function storeTheme(theme: ThemeName): void {
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    /* viz výše */
  }
}

export function applyTheme(theme: ThemeName): void {
  document.documentElement.setAttribute("data-theme", theme);
}
