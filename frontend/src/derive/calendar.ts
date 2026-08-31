/**
 * Podklady pro obrazovku Aktivity – 1:1 s Aktivity.dc.html. Měsíční mřížka
 * (pondělní týdny, vybraný týden podbarvený) a karty jízd vybraného týdne.
 * Vše se odvozuje na klientu z `/api/activities`.
 */

import type { ActivityListRow } from "../api";
import { dec, fmtMin, fmtShort, rideCount } from "../format";
import { ZONE_NAMES, type Theme } from "../theme";
import { sportKey, type SportKey } from "./sportIcons";

const WD = ["Po", "Út", "St", "Čt", "Pá", "So", "Ne"];
const MONTHS_FULL = [
  "leden", "únor", "březen", "duben", "květen", "červen",
  "červenec", "srpen", "září", "říjen", "listopad", "prosinec",
];
const CZ_MON = ["led", "úno", "bře", "dub", "kvě", "čvn", "čvc", "srp", "zář", "říj", "lis", "pro"];

export interface CalDay {
  iso: string;
  num: number;
  inMonth: boolean;
  isToday: boolean;
  isSelected: boolean;
  sport: SportKey | null;
  iconColor: string;
  durLabel: string;
}
export interface CalWeek {
  key: string;
  bg: string;
  days: CalDay[];
}
export interface MonthView {
  label: string;
  weekdayLabels: string[];
  weeks: CalWeek[];
}

export function mondayOf(iso: string): string {
  const d = new Date(`${iso}T12:00:00Z`);
  const dow = (d.getUTCDay() + 6) % 7;
  d.setUTCDate(d.getUTCDate() - dow);
  return d.toISOString().slice(0, 10);
}
function addDays(iso: string, n: number): string {
  const d = new Date(`${iso}T12:00:00Z`);
  d.setUTCDate(d.getUTCDate() + n);
  return d.toISOString().slice(0, 10);
}
function czShort(iso: string): string {
  return `${Number(iso.slice(8, 10))}. ${CZ_MON[Number(iso.slice(5, 7)) - 1]}`;
}

/** Barva ikony sportu v kalendáři (dle designu). */
function dayIconColor(key: SportKey, hard: number, theme: Theme): string {
  if (key === "bike") return hard > 0.15 ? theme.orange : hard > 0.05 ? theme.ok : theme.blue;
  if (key === "gym") return theme.blue2;
  return theme.warn;
}

export function buildMonth(
  year: number,
  month: number,
  activities: ActivityListRow[],
  selectedIso: string,
  todayIso: string,
  theme: Theme,
): MonthView {
  const selWeek = mondayOf(selectedIso);
  const byDay = new Map<string, ActivityListRow[]>();
  for (const a of activities) {
    const list = byDay.get(a.date) ?? [];
    list.push(a);
    byDay.set(a.date, list);
  }

  const first = `${year}-${String(month + 1).padStart(2, "0")}-01`;
  const gridStart = mondayOf(first);
  const weeks: CalWeek[] = [];
  for (let w = 0; w < 6; w++) {
    const wkStart = addDays(gridStart, w * 7);
    const days: CalDay[] = [];
    for (let i = 0; i < 7; i++) {
      const iso = addDays(wkStart, i);
      const acts = (byDay.get(iso) ?? [])
        .slice()
        .sort((a, b) => (b.duration_minutes ?? 0) - (a.duration_minutes ?? 0));
      const primary = acts[0];
      const key = primary ? sportKey(primary.sport) : null;
      const z = primary
        ? [primary.time_in_z1, primary.time_in_z2, primary.time_in_z3, primary.time_in_z4, primary.time_in_z5].map((x) => x ?? 0)
        : [0, 0, 0, 0, 0];
      const tot = z.reduce((a, b) => a + b, 0) || 1;
      const hard = (z[3] + z[4]) / tot;
      days.push({
        iso,
        num: Number(iso.slice(8, 10)),
        inMonth: Number(iso.slice(5, 7)) === month + 1,
        isToday: iso === todayIso,
        isSelected: iso === selectedIso,
        sport: key,
        iconColor: key ? dayIconColor(key, hard, theme) : theme.mut,
        durLabel: key === "bike" && primary?.duration_minutes ? fmtShort(primary.duration_minutes) : "",
      });
    }
    weeks.push({ key: wkStart, bg: wkStart === selWeek ? "var(--track)" : "transparent", days });
    if (w >= 4 && days.every((d) => !d.inMonth)) {
      weeks.pop();
      break;
    }
  }

  return { label: `${MONTHS_FULL[month]} ${year}`, weekdayLabels: WD, weeks };
}

// ── karty jízd vybraného týdne ───────────────────────────────────────────

export interface WeekRideCell {
  label: string;
  value: string;
  color: string;
}
export interface WeekRideZone {
  w: string;
  color: string;
  name: string;
  mins: string;
}
export interface WeekRideCard {
  id: string;
  date: string;
  sport: SportKey;
  tag: string;
  tagBg: string;
  tagColor: string;
  showDistance: boolean;
  distance: string;
  cells: WeekRideCell[];
  showZones: boolean;
  zones: WeekRideZone[];
  trimp: string;
  maxHr: string;
  kcal: string;
}
export interface WeekView {
  title: string;
  summary: string;
  cards: WeekRideCard[];
}

export function buildWeek(
  weekStartIso: string,
  activities: ActivityListRow[],
  theme: Theme,
  light: boolean,
): WeekView {
  const end = addDays(weekStartIso, 6);
  const rows = activities
    .filter((a) => a.date >= weekStartIso && a.date <= end)
    .sort((a, b) => (a.date < b.date ? 1 : -1));

  const km = rows.reduce((s, r) => s + (r.distance_km ?? 0), 0);
  const min = rows.reduce((s, r) => s + (r.duration_minutes ?? 0), 0);
  const suffix = light ? "26" : "18";

  const cards: WeekRideCard[] = rows.map((r) => {
    const z = [r.time_in_z1, r.time_in_z2, r.time_in_z3, r.time_in_z4, r.time_in_z5].map((x) => x ?? 0);
    const total = z.reduce((a, b) => a + b, 0);
    const key = sportKey(r.sport);
    const isBike = key === "bike" && (r.distance_km ?? 0) > 0;
    const hard = total ? (z[3] + z[4]) / total : 0;
    const tag = isBike
      ? hard > 0.15 ? "Práh" : hard > 0.05 ? "Tempo" : "Vytrvalost"
      : sportLabel(key);
    const tagColor = isBike
      ? hard > 0.15 ? theme.orange : hard > 0.05 ? theme.ok : theme.blue
      : theme.warn;
    const tagBgBase = isBike
      ? hard > 0.15 ? theme.orange2 : hard > 0.05 ? theme.ok : theme.blue
      : theme.warn;

    return {
      id: r.activity_id,
      date: czShort(r.date),
      sport: key,
      tag,
      tagBg: tagBgBase + suffix,
      tagColor,
      showDistance: isBike,
      distance: dec(r.distance_km ?? 0),
      cells: isBike
        ? [
            { label: "Čas", value: fmtShort(r.duration_minutes ?? 0), color: "var(--fg2)" },
            { label: "Prům. tep", value: r.avg_hr == null ? "–" : `${Math.round(r.avg_hr)}`, color: "var(--fg2)" },
            { label: "Převýšení", value: `${Math.round(r.ascent_m ?? 0)} m`, color: "var(--fg2)" },
          ]
        : [
            { label: "Čas", value: fmtShort(r.duration_minutes ?? 0), color: "var(--fg2)" },
            { label: "Kalorie", value: r.calories == null ? "–" : `${Math.round(r.calories)}`, color: "var(--fg2)" },
          ],
      showZones: total > 0,
      zones: z.map((m, i) => ({
        w: `${total ? (m / total) * 100 : 0}%`,
        color: theme.zones[i],
        name: ZONE_NAMES[i],
        mins: fmtMin(m),
      })),
      trimp: r.total_trimp == null ? "–" : String(Math.round(r.total_trimp)),
      maxHr: r.max_hr == null ? "–" : String(Math.round(r.max_hr)),
      kcal: r.calories == null ? "–" : String(Math.round(r.calories)),
    };
  });

  return {
    title: `${czShort(weekStartIso)} – ${czShort(end)}`.toUpperCase(),
    summary: rows.length ? `${rideCount(rows.length)} · ${km.toFixed(0)} km · ${fmtMin(min)}` : "Žádné aktivity",
    cards,
  };
}

function sportLabel(k: SportKey): string {
  return { bike: "Kolo", run: "Běh", hike: "Túra", gym: "Posilovna", fotbal: "Fotbal", brusleni: "Brusle", hokej: "Hokej", generic: "Aktivita" }[k];
}
