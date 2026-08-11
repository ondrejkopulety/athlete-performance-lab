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
}

export interface ActivityRow {
  id: string;
  d: string;
  z: number[];
  up: number | null;
  asc: number | null;
  grad: number | null;
  hrr: number | null;
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

export async function fetchDashboard(signal?: AbortSignal): Promise<DashboardPayload> {
  const res = await fetch(`${BASE}/dashboard`, { signal, credentials: "same-origin" });
  if (!res.ok) {
    throw new Error(`API odpovědělo ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as DashboardPayload;
}
