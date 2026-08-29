/**
 * Detail aktivity – Přehled a Grafy nad daty, která už pipeline počítá.
 *
 * Splity, kopce a rozdělení terénu (tab Stoupání) čekají na schválenou
 * migraci, viz konverzace o mapování mockupu – tenhle soubor je proto
 * jen o mapě, gaugech, tepových zónách a kombinovaném grafu tep/rychlost/
 * výška.
 */

import type { ActivityDetail, RecordPoint } from "../api";
import { czDate, dec, fmtMin, fmtShort } from "../format";
import { ZONE_SHORT, type Theme } from "../theme";

// ── Hlavička a chipy ─────────────────────────────────────────────────────

export interface DetailHeader {
  title: string;
  dateLabel: string;
  sourceLabel: string;
}

export function buildHeader(a: ActivityDetail): DetailHeader {
  const rawSport = a.sport ?? "aktivita";
  const primary = rawSport.split("/")[0] || rawSport;
  const title = a.activity_name?.trim() || primary.charAt(0).toUpperCase() + primary.slice(1);
  const time = a.start_time ? a.start_time.slice(11, 16) : null;
  return {
    title,
    dateLabel:
      czDate(a.date, { day: "numeric", month: "long", year: "numeric" }) +
      (time ? ` · ${time}` : ""),
    sourceLabel: a.source === "strava" ? "Strava" : a.source === "garmin" ? "Garmin" : "–",
  };
}

// ── Stat karty ───────────────────────────────────────────────────────────

export interface StatCard {
  label: string;
  value: string;
  unit: string;
  subLabel: string;
  subValue: string;
}

export function buildStats(a: ActivityDetail): StatCard[] {
  return [
    {
      label: "Vzdálenost",
      value: a.distance_km == null ? "–" : dec(a.distance_km, 2),
      unit: "km",
      subLabel: "Převýšení",
      subValue: a.ascent_m == null ? "–" : `${Math.round(a.ascent_m)} m`,
    },
    {
      label: "Průměrná rychlost",
      value: a.avg_speed_kmh == null ? "–" : dec(a.avg_speed_kmh),
      unit: "km/h",
      subLabel: "Maximum",
      subValue: a.max_speed_kmh == null ? "–" : `${dec(a.max_speed_kmh)} km/h`,
    },
    {
      label: "Průměrný tep",
      value: a.avg_hr == null ? "–" : String(Math.round(a.avg_hr)),
      unit: "bpm",
      subLabel: "Maximum",
      subValue: a.max_hr == null ? "–" : `${Math.round(a.max_hr)} bpm`,
    },
    {
      label: "Kalorie",
      value: a.calories == null ? "–" : Math.round(a.calories).toLocaleString("cs-CZ"),
      unit: "kcal",
      subLabel: "Tuky / cukry",
      subValue:
        a.fat_g == null || a.carb_g == null
          ? "–"
          : `${Math.round(a.fat_g)} g / ${Math.round(a.carb_g)} g`,
    },
    {
      label: "Délka aktivity",
      value: a.duration_minutes == null ? "–" : fmtShort(a.duration_minutes),
      unit: "",
      subLabel: "Zdroj",
      subValue: a.source === "strava" ? "Strava" : a.source === "garmin" ? "Garmin" : "–",
    },
  ];
}

// ── Gaugy (TRIMP + strain) a verdikt ────────────────────────────────────

const GAUGE_R = 42;
const GAUGE_C = 2 * Math.PI * GAUGE_R;
const GAUGE_ARC = 0.75 * GAUGE_C;
const GAUGE_TRACK_DASH = `${GAUGE_ARC.toFixed(1)} ${GAUGE_C.toFixed(1)}`;

/** Strop stupnice whoop_strain je pevný 0–21 (Whoopova vlastní škála, ne
 *  vymyšlené číslo) – stejně jako u denního gaugu na dashboardu. */
const STRAIN_MAX = 21;

export interface DetailGauge {
  label: string;
  value: string;
  trackDash: string;
  valDash: string;
  color: string;
}

function gaugeDash(fraction: number, mounted: boolean): string {
  const f = Math.max(0, Math.min(1, fraction));
  return `${(mounted ? f * GAUGE_ARC : 0).toFixed(1)} ${GAUGE_C.toFixed(1)}`;
}

/**
 * Verdikt a strop TRIMP gaugu stojí na percentilu vůči vlastní historii
 * kardio aktivit (``trimp_load_percentile``, spočítaný v pipeline) –
 * ne na vymyšlených pevných prazích jako v mockupu. Pásma 90/75/40 jsou
 * čtení stejného percentilu, který gauge zaplňuje, takže se nikdy
 * nerozejdou.
 */
function verdictFromPercentile(pct: number | null): string {
  if (pct == null) return "Zatím málo aktivit na srovnání";
  if (pct >= 90) return "Velmi velká zátěž · top 10 % historie";
  if (pct >= 75) return "Velká zátěž · top 25 % historie";
  if (pct >= 40) return "Střední zátěž";
  return "Lehká zátěž";
}

export interface LoadGauges {
  trimp: DetailGauge;
  strain: DetailGauge;
  verdict: string;
  /** Whoop strain je denní – null i poznámka, když je den sdílený víc jízdami. */
  strainNote: string;
}

export function buildLoadGauges(a: ActivityDetail, theme: Theme, mounted: boolean): LoadGauges {
  const pct = a.trimp_load_percentile;
  const strain = a.whoop_strain;
  const sharedDay = (a.activities_same_day ?? 1) > 1;

  return {
    trimp: {
      label: "TRIMP",
      value: a.total_trimp == null ? "–" : String(Math.round(a.total_trimp)),
      trackDash: GAUGE_TRACK_DASH,
      valDash: gaugeDash((pct ?? 0) / 100, mounted),
      color: theme.warn,
    },
    strain: {
      label: "STRAIN",
      value: strain == null ? "–" : dec(strain),
      trackDash: GAUGE_TRACK_DASH,
      valDash: gaugeDash((strain ?? 0) / STRAIN_MAX, mounted),
      color: theme.orange,
    },
    verdict: verdictFromPercentile(pct),
    strainNote: strain == null
      ? "Ten den zatím nemá spočítaný strain."
      : sharedDay
        ? `Strain celého dne (${a.activities_same_day} aktivit dohromady) – ne jen téhle jízdy.`
        : "Strain celého dne – žádná jiná aktivita ho ten den nesdílí.",
  };
}

// ── Tepové zóny ──────────────────────────────────────────────────────────

export interface DetailZoneRow {
  name: string;
  color: string;
  time: string;
  pct: number;
  w: string;
}

export function buildZoneTable(
  a: ActivityDetail,
  theme: Theme,
  mounted: boolean,
): { rows: DetailZoneRow[]; totalTime: string } {
  const mins = [a.time_in_z1, a.time_in_z2, a.time_in_z3, a.time_in_z4, a.time_in_z5].map(
    (v) => v ?? 0,
  );
  const total = mins.reduce((x, y) => x + y, 0);
  const max = Math.max(...mins, 1e-9);

  const rows: DetailZoneRow[] = mins.map((m, i) => ({
    name: ZONE_SHORT[i],
    color: theme.zones[i],
    time: fmtMin(m),
    pct: total > 0 ? Math.round((m / total) * 100) : 0,
    w: `${(mounted ? (m / max) * 100 : 0).toFixed(1)}%`,
  }));

  return { rows, totalTime: fmtMin(total) };
}

// ── Mapa trasy (SVG polyline, bez podkladu) ─────────────────────────────

const ZONE_INDEX: Record<string, number> = { Z1: 0, Z2: 1, Z3: 2, Z4: 3, Z5: 4 };

export interface MapSegment {
  d: string;
  color: string;
}

export interface DetailMap {
  segments: MapSegment[];
  empty: boolean;
}

/**
 * Rovnoběžková projekce (equirectangular) se škálováním osy X podle
 * cos(zeměpisná šířka) – na délce jedné jízdy je zkreslení zanedbatelné a
 * nemá smysl tahat do frontendu plnou mapovou projekci kvůli jedné trase.
 */
export function buildMap(records: RecordPoint[], theme: Theme): DetailMap {
  const pts = records
    .filter((r) => r.position_lat != null && r.position_long != null)
    .map((r) => ({ lat: r.position_lat as number, lon: r.position_long as number, zone: r.hr_zone }));

  if (pts.length < 2) return { segments: [], empty: true };

  const lats = pts.map((p) => p.lat);
  const lons = pts.map((p) => p.lon);
  const latMin = Math.min(...lats);
  const latMax = Math.max(...lats);
  const lonMin = Math.min(...lons);
  const lonMax = Math.max(...lons);
  const cosLat = Math.cos((((latMin + latMax) / 2) * Math.PI) / 180);

  const W = 860;
  const H = 300;
  const PAD = 24;
  const spanLon = Math.max((lonMax - lonMin) * cosLat, 1e-6);
  const spanLat = Math.max(latMax - latMin, 1e-6);
  const scale = Math.min((W - PAD * 2) / spanLon, (H - PAD * 2) / spanLat);
  const offX = (W - PAD * 2 - spanLon * scale) / 2;
  const offY = (H - PAD * 2 - spanLat * scale) / 2;

  const x = (lon: number) => PAD + offX + (lon - lonMin) * cosLat * scale;
  const y = (lat: number) => H - PAD - offY - (lat - latMin) * scale;

  const colorFor = (zone: string | null) => {
    const idx = zone ? ZONE_INDEX[zone] : undefined;
    return idx == null ? theme.mut : theme.zones[idx];
  };

  const segments: MapSegment[] = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i];
    const p1 = pts[i + 1];
    segments.push({
      d: `M${x(p0.lon).toFixed(1)} ${y(p0.lat).toFixed(1)} L${x(p1.lon).toFixed(1)} ${y(p1.lat).toFixed(1)}`,
      color: colorFor(p0.zone),
    });
  }
  return { segments, empty: false };
}

// ── Graf tep × rychlost × výška ──────────────────────────────────────────

export interface SeriesPath {
  line: string;
  area: string;
}

/** Body s chybějící hodnotou se přeskočí – spojnice tím nikdy neproletí
 *  nulou, jen díru v datech nakreslí jako rovný úsek mezi sousedy. */
function buildSeriesPath(
  vals: (number | null)[],
  lo: number,
  hi: number,
  W: number,
  H: number,
  pad: number,
): SeriesPath | null {
  const known = vals
    .map((v, i) => (v == null ? null : { i, v }))
    .filter((p): p is { i: number; v: number } => p !== null);
  if (known.length < 2 || vals.length < 2) return null;

  const x = (i: number) => (i / (vals.length - 1)) * W;
  const y = (v: number) => pad + (1 - (v - lo) / (hi - lo)) * (H - pad * 2);

  const line = known
    .map((p, idx) => `${idx ? "L" : "M"}${x(p.i).toFixed(1)} ${y(p.v).toFixed(1)}`)
    .join(" ");
  const area = `${line} L${W} ${H} L0 ${H} Z`;
  return { line, area };
}

export interface ChartAxisTick {
  label: string;
  top: string;
}

export interface DetailChart {
  hr: SeriesPath | null;
  spd: SeriesPath | null;
  el: SeriesPath | null;
  hrAxis: ChartAxisTick[];
  spdAxis: ChartAxisTick[];
  timeAxis: string[];
  minEl: number | null;
  maxEl: number | null;
}

const HR_LO = 60;
const HR_HI = 200;
const SPD_LO = 0;
const SPD_HI = 60;
const CHART_W = 700;
const CHART_H = 200;
const CHART_PAD = 6;

function axisTicks(lo: number, hi: number, ticks: number[]): ChartAxisTick[] {
  return ticks.map((v) => ({
    label: String(v),
    top: `${(
      ((CHART_PAD + (1 - (v - lo) / (hi - lo)) * (CHART_H - CHART_PAD * 2)) / CHART_H) *
      100
    ).toFixed(2)}%`,
  }));
}

export function buildChart(records: RecordPoint[], durationMin: number | null): DetailChart {
  const hrVals = records.map((r) => r.heart_rate);
  const spdVals = records.map((r) => (r.speed == null ? null : r.speed * 3.6));
  const elVals = records.map((r) => r.altitude);

  const elKnown = elVals.filter((v): v is number => v != null);
  const minEl = elKnown.length ? Math.min(...elKnown) : null;
  const maxEl = elKnown.length ? Math.max(...elKnown) : null;
  const elLo = minEl == null ? 0 : minEl - 10;
  const elHi = maxEl == null ? 100 : maxEl + 10;

  return {
    hr: buildSeriesPath(hrVals, HR_LO, HR_HI, CHART_W, CHART_H, CHART_PAD),
    spd: buildSeriesPath(spdVals, SPD_LO, SPD_HI, CHART_W, CHART_H, CHART_PAD),
    el: buildSeriesPath(elVals, elLo, elHi, CHART_W, CHART_H, CHART_PAD),
    hrAxis: axisTicks(HR_LO, HR_HI, [180, 150, 120, 90]),
    spdAxis: axisTicks(SPD_LO, SPD_HI, [45, 30, 15, 0]),
    timeAxis: [0, 0.25, 0.5, 0.75, 1].map((f) => fmtShort((durationMin ?? 0) * f)),
    minEl: minEl == null ? null : Math.round(minEl),
    maxEl: maxEl == null ? null : Math.round(maxEl),
  };
}
