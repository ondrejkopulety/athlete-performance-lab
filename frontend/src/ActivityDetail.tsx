import { Fragment, useEffect, useMemo, useState } from "react";

import {
  fetchActivity,
  fetchActivityRecords,
  type ActivityDetail as ActivityDetailPayload,
  type RecordPoint,
} from "./api";
import { Shell } from "./components/Shell";
import { mono } from "./components/ui";
import { buildActivityBlocks } from "./derive/activityBlocks";
import {
  buildChart,
  buildHeader,
  buildLoadGauges,
  buildMap,
  buildStats,
  buildZoneTable,
} from "./derive/activityDetail";
import { buildElevation } from "./derive/elevation";
import { buildSplits } from "./derive/splits";
import { fmtShort } from "./format";
import { MAP_PALETTE, THEMES, type ThemeName } from "./theme";

const TABS = ["Přehled", "Grafy", "Stoupání"] as const;
type Tab = (typeof TABS)[number];
const ZONE_VARS = ["var(--z1)", "var(--z2)", "var(--z3)", "var(--z4)", "var(--z5)"];
const EYEBROW = mono(10, { letterSpacing: ".16em", textTransform: "uppercase", color: "var(--mut)" });
const CARD = { border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "24px 28px", display: "flex", flexDirection: "column", gap: 12 } as const;
const INFO = (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="var(--mut)" strokeWidth="1.7">
    <circle cx="12" cy="12" r="9" />
    <path d="M12 11v5M12 8h.01" strokeLinecap="round" />
  </svg>
);

export function ActivityDetail({ id, theme, onBack }: { id: string; theme: ThemeName; onBack: () => void }) {
  const T = THEMES[theme];
  const [activity, setActivity] = useState<ActivityDetailPayload | null>(null);
  const [records, setRecords] = useState<RecordPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [tab, setTab] = useState<Tab>("Přehled");
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const [metric, setMetric] = useState<"hr" | "spd" | "el">("hr");
  const [tol, setTol] = useState<Record<string, number>>({ z4: 0, z2: 0, z1: 0 });
  const [terrainOpen, setTerrainOpen] = useState(false);

  useEffect(() => {
    setActivity(null);
    setRecords(null);
    setError(null);
    setMounted(false);
    setTab("Přehled");
    const ctrl = new AbortController();
    Promise.all([fetchActivity(id, ctrl.signal), fetchActivityRecords(id, "10s", ctrl.signal)])
      .then(([a, r]) => {
        setActivity(a);
        setRecords(r);
        setTimeout(() => setMounted(true), 30);
      })
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      });
    return () => ctrl.abort();
  }, [id]);

  const header = useMemo(() => (activity ? buildHeader(activity) : null), [activity]);
  const stats = useMemo(() => (activity ? buildStats(activity) : []), [activity]);
  const gauges = useMemo(() => (activity ? buildLoadGauges(activity, T, mounted) : null), [activity, T, mounted]);
  const zoneTable = useMemo(() => (activity ? buildZoneTable(activity, T, mounted) : null), [activity, T, mounted]);
  const map = useMemo(() => buildMap(records ?? [], MAP_PALETTE[theme].zones), [records, theme]);
  const chart = useMemo(() => buildChart(records ?? [], activity?.duration_minutes ?? null), [records, activity]);
  const splits = useMemo(() => buildSplits(records ?? [], T), [records, T]);
  const elevation = useMemo(() => buildElevation(records ?? [], activity), [records, activity]);
  const blocks = useMemo(
    () => buildActivityBlocks(records ?? [], tol, (records ?? []).length > 10),
    [records, tol],
  );

  const chips = useMemo(() => {
    if (!activity) return [];
    const out: { icon: string; label: string }[] = [];
    if (activity.distance_km != null)
      out.push({ icon: "M12 21C12 21 4 14 4 8.5A4.5 4.5 0 0112 5a4.5 4.5 0 018 3.5C20 14 12 21 12 21z", label: `${activity.distance_km.toFixed(1).replace(".", ",")} km` });
    if (activity.duration_minutes != null)
      out.push({ icon: "M12 7v5l3 2M21 12a9 9 0 11-18 0 9 9 0 0118 0z", label: fmtShort(activity.duration_minutes) });
    if (header) out.push({ icon: "M8 2v4M16 2v4M3 10h18M5 4h14a2 2 0 012 2v14a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2z", label: header.dateLabel });
    return out;
  }, [activity, header]);

  return (
    <Shell screen="activity" maxWidth={880} gap={20}>
      {/* Header */}
      <header style={{ display: "flex", alignItems: "flex-start", gap: 14, padding: "2px 2px 6px" }}>
        <button
          type="button"
          onClick={onBack}
          aria-label="Zpět"
          className="hover-fg"
          style={{ flex: "0 0 auto", display: "flex", alignItems: "center", justifyContent: "center", width: 36, height: 36, marginTop: 12, borderRadius: "50%", background: "none", border: "none", color: "var(--fg)", cursor: "pointer" }}
        >
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M19 12H5M11 18l-6-6 6-6" />
          </svg>
        </button>
        <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>Aktivita</span>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>{header?.title ?? "…"}</h1>
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--mut)" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 20h9M16.5 3.5a2.1 2.1 0 013 3L7 19l-4 1 1-4z" />
            </svg>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14, paddingTop: 12, color: "var(--fg)" }}>
          <svg viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="18" cy="5" r="3" /><circle cx="6" cy="12" r="3" /><circle cx="18" cy="19" r="3" />
            <path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4" />
          </svg>
          <svg viewBox="0 0 24 24" width="19" height="19" fill="currentColor">
            <circle cx="12" cy="5" r="1.7" /><circle cx="12" cy="12" r="1.7" /><circle cx="12" cy="19" r="1.7" />
          </svg>
        </div>
      </header>

      {error && <section style={{ ...CARD }}><p style={NOTE}>Jízdu se nepodařilo načíst: {error}</p></section>}
      {!error && !activity && <section style={{ ...CARD }}><p style={NOTE}>Načítám jízdu…</p></section>}

      {activity && (
        <>
          {/* Chips */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", padding: "0 2px" }}>
            {chips.map((c) => (
              <div key={c.label} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", border: "1px solid var(--line2)", borderRadius: 999, background: "var(--card)" }}>
                <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="var(--mut)" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
                  <path d={c.icon} />
                </svg>
                <span style={mono(12, { color: "var(--fg)", whiteSpace: "nowrap" })}>{c.label}</span>
              </div>
            ))}
          </div>

          {/* Underline tabs */}
          <nav style={{ display: "flex", gap: 24, padding: "2px 2px 0", borderBottom: "1px solid var(--line)" }}>
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                onClick={() => setTab(t)}
                style={mono(11, {
                  appearance: "none",
                  background: "none",
                  border: "none",
                  padding: "0 0 11px",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  letterSpacing: ".12em",
                  textTransform: "uppercase",
                  transition: ".2s",
                  color: tab === t ? "var(--fg)" : "var(--mut)",
                  borderBottom: `2px solid ${tab === t ? "var(--fg)" : "transparent"}`,
                })}
              >
                {t}
              </button>
            ))}
          </nav>

          {tab === "Přehled" && gauges && zoneTable && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <RouteMap map={map} />
              <StatGrid stats={stats} />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 20, alignItems: "stretch" }}>
                <EffortCard gauges={gauges} />
                <ZoneCard zoneTable={zoneTable} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 20, alignItems: "stretch" }}>
                {blocks.map((b) => (
                  <BlockCard key={b.key} card={b} tolerance={tol[b.key] ?? 0} onPickTol={(v) => setTol((s) => ({ ...s, [b.key]: v }))} />
                ))}
              </div>
            </div>
          )}

          {tab === "Grafy" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <MetricChart chart={chart} metric={metric} onPickMetric={setMetric} hidden={hidden} onToggle={(k) => setHidden((h) => ({ ...h, [k]: !h[k] }))} />
              <SplitsCard splits={splits} />
            </div>
          )}

          {tab === "Stoupání" && <ClimbTab elevation={elevation} terrainOpen={terrainOpen} onToggleTerrain={() => setTerrainOpen((o) => !o)} />}
        </>
      )}
    </Shell>
  );
}

const NOTE = { margin: 0, fontSize: 12, lineHeight: 1.5, color: "var(--faint)", textWrap: "pretty" } as const;

// ── Mapa ────────────────────────────────────────────────────────────────
function RouteMap({ map }: { map: ReturnType<typeof buildMap> }) {
  return (
    <section style={{ position: "relative", border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", overflow: "hidden", height: 360 }}>
      <div style={{ position: "absolute", inset: 0, background: "var(--mapbg)" }} />
      {map.empty ? (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={mono(11, { color: "var(--maplabel)" })}>Trasa nemá GPS souřadnice</span>
        </div>
      ) : (
        <svg viewBox="0 0 860 300" preserveAspectRatio="xMidYMid meet" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
          {map.segments.map((s, i) => (
            <path key={i} d={s.d} fill="none" stroke={s.color} strokeWidth="4.5" strokeLinecap="round" strokeLinejoin="round" />
          ))}
        </svg>
      )}
      <div style={{ position: "absolute", top: 14, left: 14, padding: "12px 14px", borderRadius: 12, background: "var(--panel)", backdropFilter: "blur(8px)", pointerEvents: "none", display: "flex", flexDirection: "column", gap: 8 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 24 }}>
          <span style={mono(10, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--fg)" })}>Tep</span>
          <span style={mono(10, { color: "var(--mut)" })}>bpm</span>
        </div>
        {["Z1", "Z2", "Z3", "Z4", "Z5"].map((z, i) => (
          <div key={z} style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: ZONE_VARS[i] }} />
            <span style={mono(11, { color: "var(--fg2)" })}>{z}</span>
          </div>
        ))}
      </div>
      <div style={{ position: "absolute", top: 14, right: 14, display: "flex", alignItems: "center", justifyContent: "center", width: 34, height: 34, borderRadius: 10, background: "var(--panel)", backdropFilter: "blur(8px)", pointerEvents: "none" }}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--fg)" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
          <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
        </svg>
      </div>
    </section>
  );
}

// ── Stat grid ───────────────────────────────────────────────────────────
function StatGrid({ stats }: { stats: ReturnType<typeof buildStats> }) {
  return (
    <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 12 }}>
      {stats.map((s) => (
        <div key={s.label} style={{ border: "1px solid var(--line)", borderRadius: 18, background: "var(--card)", padding: 20, display: "flex", flexDirection: "column", gap: 6, minWidth: 0 }}>
          <span style={mono(9, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>{s.label}</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 6, minWidth: 0 }}>
            <span style={mono(26, { lineHeight: 1.05, letterSpacing: "-.03em", whiteSpace: "nowrap" })}>{s.value}</span>
            {s.unit && <span style={mono(10, { color: "var(--faint)" })}>{s.unit}</span>}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)", minWidth: 0 }}>
            <span style={mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>{s.subLabel}</span>
            <span style={mono(12, { color: "var(--fg2)", marginLeft: "auto", whiteSpace: "nowrap" })}>{s.subValue}</span>
          </div>
        </div>
      ))}
    </section>
  );
}

// ── Úsilí ───────────────────────────────────────────────────────────────
function EffortCard({ gauges }: { gauges: NonNullable<ReturnType<typeof buildLoadGauges>> }) {
  const ring = (g: { value: string; trackDash: string; valDash: string; color: string; label: string }) => (
    <div style={{ position: "relative", width: "100%", maxWidth: 150, aspectRatio: "1" }}>
      <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", transform: "rotate(135deg)" }}>
        <circle cx="50" cy="50" r="42" fill="none" style={{ stroke: "var(--track)" }} strokeWidth="9" strokeLinecap="round" strokeDasharray={g.trackDash} />
        <circle cx="50" cy="50" r="42" fill="none" stroke={g.color} strokeWidth="9" strokeLinecap="round" strokeDasharray={g.valDash} style={{ transition: "stroke-dasharray .9s cubic-bezier(.22,1,.36,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2 }}>
        <span style={mono(30, { lineHeight: 1, letterSpacing: "-.03em" })}>{g.value}</span>
        <span style={mono(9, { letterSpacing: ".16em", color: "var(--mut)" })}>{g.label}</span>
      </div>
    </div>
  );
  return (
    <section style={{ ...CARD }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={EYEBROW}>Úsilí</span>
        {INFO}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, justifyItems: "center" }}>
        {ring(gauges.trimp)}
        {ring(gauges.strain)}
      </div>
      <span style={mono(13, { color: "var(--warn)" })}>{gauges.verdict}</span>
    </section>
  );
}

// ── Tepové zóny ─────────────────────────────────────────────────────────
function ZoneCard({ zoneTable }: { zoneTable: NonNullable<ReturnType<typeof buildZoneTable>> }) {
  return (
    <section style={{ ...CARD, gap: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={EYEBROW}>Tepové zóny</span>
        {INFO}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "44px 62px 34px 1fr", gap: "8px 10px", alignItems: "center" }}>
        <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>Zóna</span>
        <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>Čas</span>
        <span /><span />
        {zoneTable.rows.map((z, i) => (
          <Fragment key={z.name}>
            <span style={mono(12, { color: ZONE_VARS[i] })}>{z.name}</span>
            <span style={mono(12, { color: "var(--fg)", whiteSpace: "nowrap" })}>{z.time}</span>
            <span style={mono(11, { color: "var(--mut)", textAlign: "right" })}>{z.pct} %</span>
            <span style={{ height: 7, borderRadius: 999, background: ZONE_VARS[i], width: z.w, minWidth: 7, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
          </Fragment>
        ))}
      </div>
      <div style={{ height: 1, background: "var(--line)" }} />
      <div style={{ display: "grid", gridTemplateColumns: "44px 62px 34px 1fr", gap: 10, alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "var(--mut)" }}>Celkem</span>
        <span style={mono(12)}>{zoneTable.totalTime}</span>
        <span style={mono(11, { color: "var(--mut)", textAlign: "right" })}>100%</span>
        <span />
      </div>
    </section>
  );
}

// ── Blokové karty ───────────────────────────────────────────────────────
function BlockCard({
  card,
  tolerance,
  onPickTol,
}: {
  card: ReturnType<typeof buildActivityBlocks>[number];
  tolerance: number;
  onPickTol: (v: number) => void;
}) {
  return (
    <section style={{ ...CARD }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={EYEBROW}>{card.title}</span>
          {INFO}
        </div>
        <div style={{ display: "flex", gap: 4, padding: 3, borderRadius: 999, background: "var(--track)" }}>
          {[0, 15].map((v) => (
            <button
              key={v}
              type="button"
              onClick={() => onPickTol(v)}
              style={mono(10, { appearance: "none", border: "none", cursor: "pointer", padding: "5px 12px", borderRadius: 999, letterSpacing: ".06em", background: tolerance === v ? "var(--card2)" : "transparent", color: tolerance === v ? card.color : "var(--mut)" })}
            >
              {v} s
            </button>
          ))}
        </div>
      </div>

      {card.isNormal && (
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <span style={mono(34, { lineHeight: 1, letterSpacing: "-.03em", color: "var(--fg)", fontVariantNumeric: "tabular-nums" })}>{card.longest}</span>
            <span style={mono(11, { color: card.color })}>{card.thresholdCaption}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {card.buckets.map((b) => (
              <div key={b.label} style={{ display: "grid", gridTemplateColumns: "56px 26px 1fr 48px", gap: 8, alignItems: "center" }}>
                <span style={mono(10, { color: "var(--mut)", whiteSpace: "nowrap" })}>{b.label}</span>
                <span style={mono(10, { color: "var(--faint)" })}>{b.count}</span>
                <span style={{ height: 6, borderRadius: 999, background: "var(--track)", overflow: "hidden", display: "block" }}>
                  <span style={{ display: "block", height: "100%", width: b.w, borderRadius: 999, background: card.color, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
                </span>
                <span style={mono(10, { color: "var(--fg2)", textAlign: "right", whiteSpace: "nowrap" })}>{b.min}</span>
              </div>
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 2, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)" }}>
            <span style={mono(11, { color: "var(--fg2)" })}>{card.countFooter}</span>
            <span style={mono(11, { color: "var(--mut)" })}>{card.inBlocksFooter}</span>
          </div>
        </div>
      )}
      {card.isEmpty && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "6px 0 2px" }}>
          <span style={mono(34, { lineHeight: 1, color: "var(--faint)" })}>—</span>
          <span style={{ fontSize: 12, lineHeight: 1.4, color: "var(--mut)", textWrap: "pretty" }}>{card.emptyMsg}</span>
        </div>
      )}
      {card.isNoData && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, padding: "10px 0" }}>
          <span style={{ fontSize: 12, lineHeight: 1.4, color: "var(--faint)", textWrap: "pretty" }}>
            Bloky nejsou k dispozici — tato aktivita nemá záznam tepu s dostatečnou frekvencí (vteřinová data).
          </span>
        </div>
      )}
    </section>
  );
}

// ── Grafy: metrický graf ────────────────────────────────────────────────
const SERIES: { k: "hr" | "spd" | "el"; label: string; color: string; unit: string }[] = [
  { k: "hr", label: "Tep", color: "var(--bad)", unit: "bpm" },
  { k: "spd", label: "Rychlost", color: "var(--blue)", unit: "km/h" },
  { k: "el", label: "Výška", color: "var(--ok)", unit: "m" },
];

function MetricChart({
  chart,
  metric,
  onPickMetric,
  hidden,
  onToggle,
}: {
  chart: ReturnType<typeof buildChart>;
  metric: "hr" | "spd" | "el";
  onPickMetric: (m: "hr" | "spd" | "el") => void;
  hidden: Record<string, boolean>;
  onToggle: (k: string) => void;
}) {
  const cur = SERIES.find((s) => s.k === metric)!;
  const series = metric === "hr" ? chart.hr : metric === "spd" ? chart.spd : chart.el;
  const heroVal =
    metric === "el"
      ? chart.maxEl == null ? "–" : String(chart.maxEl)
      : metric === "spd"
        ? chart.spdAvg == null ? "–" : chart.spdAvg.toFixed(1).replace(".", ",")
        : chart.hrAvg == null ? "–" : String(chart.hrAvg);

  return (
    <section style={{ border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "24px 20px 20px", display: "flex", flexDirection: "column", gap: 16 }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12, padding: "0 4px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={mono(10, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>{cur.label}</span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span style={{ fontFamily: "var(--font-sans),'DM Sans',sans-serif", fontSize: 38, lineHeight: 1, fontWeight: 500, letterSpacing: "-.03em", color: "var(--fg)", fontVariantNumeric: "tabular-nums" }}>{heroVal}</span>
            <span style={mono(11, { color: "var(--mut)" })}>{cur.unit}</span>
          </div>
        </div>
      </div>

      <div style={{ display: "flex", gap: 3, padding: 3, borderRadius: 999, background: "var(--track)", alignSelf: "flex-start" }}>
        {SERIES.map((s) => (
          <button
            key={s.k}
            type="button"
            onClick={() => onPickMetric(s.k)}
            style={mono(10, { appearance: "none", border: "none", cursor: "pointer", padding: "6px 13px", borderRadius: 999, letterSpacing: ".06em", background: metric === s.k ? "var(--card2)" : "transparent", color: metric === s.k ? s.color : "var(--mut)" })}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div style={{ position: "relative", padding: "0 4px" }}>
        {series == null ? (
          <div style={{ height: 196, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={mono(11, { color: "var(--faint)" })}>Jízda nemá dost vteřinových dat</span>
          </div>
        ) : (
          <svg viewBox="0 0 700 200" preserveAspectRatio="none" style={{ width: "100%", height: 196, display: "block" }}>
            <defs>
              <linearGradient id="mFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={cur.color} stopOpacity="0.22" />
                <stop offset="100%" stopColor={cur.color} stopOpacity="0" />
              </linearGradient>
            </defs>
            {chart.el && metric !== "el" && <path d={chart.el.area} fill="var(--track)" opacity={hidden.elbg ? 0 : 0.5} />}
            {chart.el && metric !== "el" && <path d={chart.el.line} fill="none" style={{ stroke: "var(--line2)" }} strokeWidth="1.2" vectorEffect="non-scaling-stroke" opacity={hidden.elbg ? 0 : 1} />}
            <path d={series.area} fill="url(#mFill)" />
            <path d={series.line} fill="none" stroke={cur.color} strokeWidth="1.6" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
          </svg>
        )}
        <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 8 }}>
          {chart.timeAxis.map((t, i) => (
            <span key={i} style={mono(10, { color: "var(--faint)" })}>{t}</span>
          ))}
        </div>
      </div>
      {chart.el && metric !== "el" && (
        <button type="button" onClick={() => onToggle("elbg")} style={mono(9, { alignSelf: "flex-start", padding: "4px 10px", borderRadius: 999, border: "1px solid var(--line2)", background: "var(--track)", color: hidden.elbg ? "var(--faint)" : "var(--fg2)", cursor: "pointer", letterSpacing: ".08em", textTransform: "uppercase" })}>
          Výškový podklad
        </button>
      )}
    </section>
  );
}

// ── Grafy: splity ───────────────────────────────────────────────────────
function SplitsCard({ splits }: { splits: ReturnType<typeof buildSplits> }) {
  return (
    <section style={{ ...CARD, gap: 4 }}>
      <span style={EYEBROW}>Průměrný tep po 5 km</span>
      {splits.length === 0 ? (
        <p style={NOTE}>Jízda nemá dost vteřinových dat se vzdáleností a tepem.</p>
      ) : (
        splits.map((s) => (
          <div key={s.range} style={{ display: "flex", flexDirection: "column", gap: 4, padding: "12px 2px", borderBottom: "1px solid var(--line)" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10 }}>
              <span style={mono(11, { color: "var(--mut)" })}>{s.range}</span>
              <span style={mono(13, { color: s.color })}>{s.hr} <span style={{ color: "var(--faint)", fontSize: 10 }}>bpm</span></span>
            </div>
            <div style={{ height: 5, borderRadius: 999, background: "var(--track)", overflow: "hidden" }}>
              <div style={{ height: "100%", width: s.w, background: s.color, borderRadius: 999 }} />
            </div>
            <span style={mono(10, { color: "var(--faint)" })}>{s.speed}</span>
          </div>
        ))
      )}
    </section>
  );
}

// ── Stoupání ────────────────────────────────────────────────────────────
function ClimbTab({
  elevation,
  terrainOpen,
  onToggleTerrain,
}: {
  elevation: ReturnType<typeof buildElevation>;
  terrainOpen: boolean;
  onToggleTerrain: () => void;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <section style={{ ...CARD }}>
        <span style={EYEBROW}>Výškový profil</span>
        {elevation.empty ? (
          <p style={NOTE}>Jízda nemá záznam výšky.</p>
        ) : (
          <>
            <div style={{ position: "relative", height: 200, paddingLeft: 34 }}>
              {elevation.elAxis.map((a) => (
                <span key={a.label} style={mono(9, { position: "absolute", left: 0, top: a.top, transform: "translateY(-50%)", color: "var(--faint)" })}>{a.label}</span>
              ))}
              <svg viewBox="0 0 700 200" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block" }}>
                <defs>
                  <linearGradient id="climbFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" style={{ stopColor: "var(--blue)" }} stopOpacity="0.22" />
                    <stop offset="100%" style={{ stopColor: "var(--blue)" }} stopOpacity="0" />
                  </linearGradient>
                </defs>
                <path d={elevation.area} fill="url(#climbFill)" />
                <path d={elevation.line} fill="none" style={{ stroke: "var(--blue)" }} strokeWidth="1.6" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
              </svg>
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: 34 }}>
              {elevation.distAxis.map((d, i) => (
                <span key={i} style={mono(9, { color: "var(--faint)" })}>{d}</span>
              ))}
            </div>
            <div style={{ display: "flex", gap: 24, flexWrap: "wrap", paddingTop: 6 }}>
              <ElevStat label="Převýšení" value={elevation.ascent} />
              <ElevStat label="Klesání" value={elevation.descent} />
              <ElevStat label="Prům. sklon" value={elevation.avgGrad} />
            </div>
            <div style={{ height: 1, background: "var(--line)", margin: "6px 0" }} />
            <button type="button" onClick={onToggleTerrain} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, appearance: "none", background: "none", border: "none", cursor: "pointer", padding: 0 }}>
              <span style={EYEBROW}>Terén</span>
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="var(--mut)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ transition: "transform .25s", transform: terrainOpen ? "rotate(180deg)" : "rotate(0deg)" }}>
                <path d="M6 9l6 6 6-6" />
              </svg>
            </button>
            {terrainOpen && (
              <p style={NOTE}>
                Rozdělení terénu (do kopce / z kopce / rovina) je odvozená vrstva nad vteřinovými daty,
                kterou pipeline zatím nepočítá. Připravujeme.
              </p>
            )}
          </>
        )}
      </section>

      <section style={{ ...CARD, gap: 8 }}>
        <span style={EYEBROW}>Vyhodnocené kopce</span>
        <p style={NOTE}>
          Detekce a kategorizace jednotlivých kopců je odvozená vrstva nad vteřinovými daty, kterou
          pipeline zatím nepočítá. Připravujeme.
        </p>
      </section>
    </div>
  );
}

function ElevStat({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>{label}</span>
      <span style={mono(14)}>{value}</span>
    </div>
  );
}
