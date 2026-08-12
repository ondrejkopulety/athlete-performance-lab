/** Klient nad /api/dashboard – jediný požadavek, který stránka dělá. */

export interface Today {
  date: string;
  readiness_score: number | null;
  pure_recovery_score: number | null;
  hrv_last_night: number | null;
  hrv_weekly_avg: number | null;
  hrv_cv_pct: number | null;
  rhr_day: number | null;
  rhr_baseline_14d: number | null;
  rhr_elevation_bpm: number | null;
  sleep_duration_min: number | null;
  sleep_need_min: number | null;
  sleep_performance_pct: number | null;
  sleep_score_day: number | null;
  avg_stress_day: number | null;
  whoop_strain: number | null;
  strain: number | null;
  trimp: number | null;
  ctl: number | null;
  atl: number | null;
  tsb: number | null;
  acwr: number | null;
  recovery_time_h: number | null;
  garmin_readiness_score: number | null;
  illness_warning: boolean | null;
  coach_advice: string | null;
}

export interface Biometric {
  date: string;
  value: number | null;
  reference: number | null;
}

/**
 * Na jak úplných datech čísla jízdy stojí.
 *
 * Rozhoduje `ok` (spočítané z pokrytí po doplnění mezer). `density` je
 * hustota vzorků a je jen informativní – Smart Recording zapisuje po pěti
 * sekundách a data tím neztrácí, takže na ní varování viset nesmí.
 */
export interface Coverage {
  ok: boolean;
  pct: number | null;
  density: number | null;
  gap: number | null;
  note: string | null;
}

export interface Ride {
  id: string;
  d: string;
  dur: number | null;
  km: number | null;
  avg: number | null;
  max: number | null;
  asc: number | null;
  trimp: number | null;
  kcal: number | null;
  z: number[];
  cov: Coverage | null;
}

export interface ActivityRow {
  id: string;
  d: string;
  z: number[];
  up: number | null;
  asc: number | null;
  grad: number | null;
  hrr: number | null;
  /** Nezáměrná Z3 v sekundách; null = bloky ještě spočítané nejsou (≠ 0). */
  z3u: number | null;
  cov: Coverage | null;
}

// ── Panely tepové křivky a souvislých bloků ───────────────────────────────

/** `hr === null` znamená, že období takhle dlouhé okno nemá. Nikdy ne 0. */
export interface CurvePoint {
  d: number;
  hr: number | null;
  activity_id: string | null;
  date: string | null;
  label: string | null;
}

export interface LastMaxEffort {
  date: string;
  activity_id: string | null;
  label: string | null;
  duration_s: number;
  days_ago: number;
  stale: boolean;
}

export interface CurvePeriod {
  label: string;
  from_date: string | null;
  to_date: string | null;
  rides_total: number;
  rides_excluded: number;
  points: CurvePoint[];
  last_max_effort: LastMaxEffort | null;
}

export interface HrCurvePayload {
  durations_s: number[];
  complete_only: boolean;
  coverage_warn_pct: number;
  period: CurvePeriod;
  reference: CurvePeriod | null;
}

export interface SegmentBucket {
  bucket: string;
  count: number;
  seconds: number;
}

export interface HrBlocksPayload {
  requested_bpm: number;
  threshold_bpm: number;
  zone: string;
  lthr_bpm: number;
  tolerance_s: number;
  complete_only: boolean;
  coverage_warn_pct: number;
  rides_total: number;
  rides_excluded: number;
  /** null = v období není ani jedna jízda (ne „nula sekund nad prahem"). */
  longest_block_s: number | null;
  longest_block: { activity_id: string | null; date: string | null; label: string | null } | null;
  previous_longest_block_s: number | null;
  trend_pct: number | null;
  hist: SegmentBucket[];
  totals: { segment_count: number; total_time_s: number; time_in_long_blocks_s: number };
}

export interface Threshold {
  lthr_bpm: number;
  hr_max_bpm: number;
  valid_from: string;
  note: string | null;
  days_ago: number;
  stale: boolean;
  stale_after_days: number;
  source: "user" | "settings";
  zone_thresholds: Record<string, number>;
}

/** [datum, ctl, atl, tsb, trimp, strain, readiness, acwr] */
export type DayTuple = [
  string,
  number,
  number,
  number,
  number,
  number | null,
  number | null,
  number | null,
];

export interface DashboardPayload {
  generated_at: string;
  today: Today | null;
  last_known: Record<"hrv" | "rhr" | "sleep", Biometric | null>;
  days: DayTuple[];
  rides: Ride[];
  activities: ActivityRow[];
}

/** Denní řádek v pohodlnějším tvaru, než jaký chodí po drátě. */
export interface Day {
  d: string;
  ctl: number;
  atl: number;
  tsb: number;
  trimp: number;
  strain: number | null;
  ready: number | null;
  acwr: number | null;
}

export function toDays(rows: DayTuple[]): Day[] {
  return rows.map((a) => ({
    d: a[0],
    ctl: a[1],
    atl: a[2],
    tsb: a[3],
    trimp: a[4],
    strain: a[5],
    ready: a[6],
    acwr: a[7],
  }));
}

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal, credentials: "same-origin" });
  if (!res.ok) {
    throw new Error(`API odpovědělo ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export async function fetchDashboard(signal?: AbortSignal): Promise<DashboardPayload> {
  return get<DashboardPayload>("/dashboard", signal);
}

/**
 * Panely křivky a bloků se dotahují zvlášť, na rozdíl od zbytku dashboardu.
 * Jsou to agregace přes období a přes filtr „jen úplná data" – tedy něco, co
 * se v prohlížeči z per-aktivitních řádků poskládat nedá, aniž by se sem
 * poslalo sedmnáct tisíc řádků bloků.
 */
export async function fetchCurve(
  params: { since: string; until: string; completeOnly: boolean; compare: "prev" | "year" },
  signal?: AbortSignal,
): Promise<HrCurvePayload> {
  const q = new URLSearchParams({
    since: params.since,
    until: params.until,
    complete_only: String(params.completeOnly),
    compare: params.compare,
  });
  return get<HrCurvePayload>(`/hr/curve?${q}`, signal);
}

export async function fetchBlocks(
  params: { since: string; until: string; completeOnly: boolean; tolerance: number; zone?: string },
  signal?: AbortSignal,
): Promise<HrBlocksPayload> {
  const q = new URLSearchParams({
    since: params.since,
    until: params.until,
    complete_only: String(params.completeOnly),
    tolerance: String(params.tolerance),
    ...(params.zone ? { zone: params.zone } : {}),
  });
  return get<HrBlocksPayload>(`/hr/blocks?${q}`, signal);
}

export async function fetchThreshold(signal?: AbortSignal): Promise<Threshold> {
  return get<Threshold>("/profile/threshold", signal);
}

export async function saveThreshold(body: {
  lthr_bpm: number;
  hr_max_bpm: number;
  valid_from?: string;
  note?: string;
}): Promise<Threshold> {
  const res = await fetch(`${BASE}/profile/threshold`, {
    method: "PUT",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`Práh se nepodařilo uložit (${res.status})`);
  }
  return (await res.json()) as Threshold;
}
