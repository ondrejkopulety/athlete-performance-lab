import { Fragment, useEffect, useMemo, useState } from "react";

import {
  fetchActivity,
  fetchActivityRecords,
  type ActivityDetail as ActivityDetailPayload,
  type RecordPoint,
} from "./api";
import {
  buildChart,
  buildHeader,
  buildLoadGauges,
  buildMap,
  buildStats,
  buildZoneTable,
} from "./derive/activityDetail";
import { THEMES, type ThemeName } from "./theme";
import { CARD, mono, NOTE, SECTION_LABEL } from "./components/ui";

const TABS = ["Přehled", "Grafy", "Stoupání"] as const;
type Tab = (typeof TABS)[number];

const BACK_ICON = "M19 12H5M11 18l-6-6 6-6";

export function ActivityDetail({
  id,
  theme,
  onBack,
}: {
  id: string;
  theme: ThemeName;
  onBack: () => void;
}) {
  const T = THEMES[theme];

  const [activity, setActivity] = useState<ActivityDetailPayload | null>(null);
  const [records, setRecords] = useState<RecordPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [tab, setTab] = useState<Tab>("Přehled");
  const [hidden, setHidden] = useState<Record<string, boolean>>({});

  useEffect(() => {
    setActivity(null);
    setRecords(null);
    setError(null);
    setMounted(false);
    setTab("Přehled");

    const ctrl = new AbortController();
    Promise.all([
      fetchActivity(id, ctrl.signal),
      fetchActivityRecords(id, "10s", ctrl.signal),
    ])
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
  const gauges = useMemo(
    () => (activity ? buildLoadGauges(activity, T, mounted) : null),
    [activity, T, mounted],
  );
  const zoneTable = useMemo(
    () => (activity ? buildZoneTable(activity, T, mounted) : null),
    [activity, T, mounted],
  );
  const map = useMemo(() => buildMap(records ?? [], T), [records, T]);
  const chart = useMemo(
    () => buildChart(records ?? [], activity?.duration_minutes ?? null),
    [records, activity],
  );

  const toggle = (k: string) => setHidden((h) => ({ ...h, [k]: !h[k] }));

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "var(--bg)",
        color: "var(--fg)",
        fontFamily: "'DM Sans',system-ui,sans-serif",
        padding: "28px 20px 72px",
        transition: "background .35s ease,color .35s ease",
      }}
    >
      <div
        style={{
          maxWidth: 880,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <header style={{ display: "flex", alignItems: "flex-start", gap: 14 }}>
          <button
            type="button"
            onClick={onBack}
            aria-label="Zpět"
            className="hover-fg"
            style={{
              flex: "0 0 auto",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              height: 36,
              marginTop: 4,
              borderRadius: "50%",
              border: "1px solid var(--line2)",
              background: "var(--card)",
              color: "var(--fg)",
              cursor: "pointer",
            }}
          >
            <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d={BACK_ICON} />
            </svg>
          </button>
          <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 3 }}>
            <span style={{ ...SECTION_LABEL }}>Aktivita</span>
            <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>
              {header?.title ?? "…"}
            </h1>
            {header && <span style={mono(11.5, { color: "var(--mut)" })}>{header.dateLabel}</span>}
          </div>
        </header>

        {error && (
          <div style={{ ...CARD, padding: 20 }}>
            <p style={NOTE}>Jízdu se nepodařilo načíst: {error}</p>
          </div>
        )}

        {!error && !activity && (
          <div style={{ ...CARD, padding: 20 }}>
            <p style={NOTE}>Načítám jízdu…</p>
          </div>
        )}

        {activity && (
          <>
            <nav style={{ display: "flex", gap: 22, padding: "2px 2px 0", borderBottom: "1px solid var(--line)" }}>
              {TABS.map((t) => (
                <button
                  key={t}
                  type="button"
                  onClick={() => setTab(t)}
                  style={mono(11.5, {
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
                    borderBottom: `2px solid ${tab === t ? T.warn : "transparent"}`,
                  })}
                >
                  {t}
                </button>
              ))}
            </nav>

            {tab === "Přehled" && gauges && zoneTable && (
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <RouteMap map={map} theme={T} />
                <StatGrid stats={stats} />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: 10, alignItems: "stretch" }}>
                  <LoadCard gauges={gauges} />
                  <ZoneCard zoneTable={zoneTable} />
                </div>
              </div>
            )}

            {tab === "Grafy" && (
              <ChartCard chart={chart} hidden={hidden} onToggle={toggle} theme={T} />
            )}

            {tab === "Stoupání" && <ClimbPlaceholder />}
          </>
        )}
      </div>
    </div>
  );
}

// ── Mapa trasy ───────────────────────────────────────────────────────────

function RouteMap({
  map,
  theme,
}: {
  map: ReturnType<typeof buildMap>;
  theme: (typeof THEMES)["dark"];
}) {
  return (
    <section style={{ position: "relative", border: "1px solid var(--line)", borderRadius: 18, background: "var(--card)", overflow: "hidden", height: 300 }}>
      <div style={{ position: "absolute", inset: 0, background: "#101216" }} />
      {map.empty ? (
        <div style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <span style={mono(11, { color: "var(--faint)" })}>Trasa nemá GPS souřadnice</span>
        </div>
      ) : (
        <svg viewBox="0 0 860 300" preserveAspectRatio="xMidYMid meet" style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}>
          {map.segments.map((s, i) => (
            <path key={i} d={s.d} fill="none" stroke={s.color} strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
          ))}
        </svg>
      )}
      <div
        style={{
          position: "absolute",
          top: 14,
          left: 14,
          padding: "12px 14px",
          borderRadius: 12,
          background: "rgba(9,9,11,.82)",
          backdropFilter: "blur(8px)",
          pointerEvents: "none",
          display: "flex",
          flexDirection: "column",
          gap: 8,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", gap: 22 }}>
          <span style={mono(10.5, { letterSpacing: ".1em", textTransform: "uppercase", color: "#fafafa" })}>Tep</span>
          <span style={mono(10.5, { color: "#8b8b93" })}>bpm</span>
        </div>
        {["Z1", "Z2", "Z3", "Z4", "Z5"].map((z, i) => (
          <div key={z} style={{ display: "flex", alignItems: "center", gap: 9 }}>
            <span style={{ width: 9, height: 9, borderRadius: "50%", background: theme.zones[i] }} />
            <span style={mono(11, { color: "#e4e4e7" })}>{z}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

// ── Stat karty ───────────────────────────────────────────────────────────

function StatGrid({ stats }: { stats: ReturnType<typeof buildStats> }) {
  return (
    <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(164px,1fr))", gap: 10 }}>
      {stats.map((s) => (
        <div
          key={s.label}
          style={{
            border: "1px solid var(--line)",
            borderRadius: 16,
            background: "var(--card)",
            padding: "14px 15px",
            display: "flex",
            flexDirection: "column",
            gap: 6,
            minWidth: 0,
          }}
        >
          <span style={mono(9.5, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>
            {s.label}
          </span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 5, minWidth: 0 }}>
            <span style={mono(26, { lineHeight: 1.05, letterSpacing: "-.03em", whiteSpace: "nowrap" })}>{s.value}</span>
            {s.unit && <span style={mono(10.5, { color: "var(--faint)" })}>{s.unit}</span>}
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)", minWidth: 0 }}>
            <span style={mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" })}>
              {s.subLabel}
            </span>
            <span style={mono(12.5, { color: "var(--fg)", marginLeft: "auto", whiteSpace: "nowrap" })}>{s.subValue}</span>
          </div>
        </div>
      ))}
    </section>
  );
}

// ── Gaugy zátěže ─────────────────────────────────────────────────────────

function LoadCard({ gauges }: { gauges: ReturnType<typeof buildLoadGauges> }) {
  const ring = (g: typeof gauges.trimp) => (
    <div style={{ position: "relative", width: 150, height: 150, flex: "0 0 auto" }}>
      <svg viewBox="0 0 100 100" style={{ width: "100%", height: "100%", transform: "rotate(135deg)" }}>
        <circle cx="50" cy="50" r="42" fill="none" style={{ stroke: "var(--track)" }} strokeWidth="9" strokeLinecap="round" strokeDasharray={g.trackDash} />
        <circle cx="50" cy="50" r="42" fill="none" stroke={g.color} strokeWidth="9" strokeLinecap="round" strokeDasharray={g.valDash} style={{ transition: "stroke-dasharray .9s cubic-bezier(.22,1,.36,1)" }} />
      </svg>
      <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 2 }}>
        <span style={mono(32, { lineHeight: 1, letterSpacing: "-.03em" })}>{g.value}</span>
        <span style={mono(9.5, { letterSpacing: ".16em", color: "var(--mut)" })}>{g.label}</span>
      </div>
    </div>
  );

  return (
    <section style={{ ...CARD, padding: 14, gap: 12 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={SECTION_LABEL}>Jak těžká byla jízda</span>
      </div>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 18, flexWrap: "wrap" }}>
        {ring(gauges.trimp)}
        {ring(gauges.strain)}
      </div>
      <span style={mono(13, { color: "var(--warn)" })}>{gauges.verdict}</span>
      <p style={NOTE}>{gauges.strainNote}</p>
    </section>
  );
}

// ── Tepové zóny ──────────────────────────────────────────────────────────

function ZoneCard({ zoneTable }: { zoneTable: ReturnType<typeof buildZoneTable> }) {
  return (
    <section style={{ ...CARD, padding: 12, gap: 10 }}>
      <span style={SECTION_LABEL}>Tepové zóny</span>
      <div style={{ display: "grid", gridTemplateColumns: "76px 62px 34px 1fr", gap: "7px 10px", alignItems: "center" }}>
        <span style={mono(9.5, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>Zóna</span>
        <span style={mono(9.5, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>Čas</span>
        <span />
        <span />
        {zoneTable.rows.map((z) => (
          <Fragment key={z.name}>
            <span style={mono(12.5, { color: z.color, whiteSpace: "nowrap" })}>{z.name}</span>
            <span style={mono(12, { color: "var(--fg)", whiteSpace: "nowrap" })}>{z.time}</span>
            <span style={mono(11.5, { color: "var(--mut)", textAlign: "right" })}>{z.pct} %</span>
            <span style={{ height: 7, borderRadius: 99, background: z.color, width: z.w, minWidth: 7, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
          </Fragment>
        ))}
      </div>
      <div style={{ height: 1, background: "var(--line)" }} />
      <div style={{ display: "grid", gridTemplateColumns: "76px 62px 34px 1fr", gap: 10, alignItems: "center" }}>
        <span style={{ fontSize: 13, color: "var(--mut)" }}>Celkem</span>
        <span style={mono(12)}>{zoneTable.totalTime}</span>
        <span style={mono(11.5, { color: "var(--mut)", textAlign: "right" })}>100%</span>
        <span />
      </div>
    </section>
  );
}

// ── Graf tep/rychlost/výška ────────────────────────────────────────────

const SERIES = [
  { k: "hr", label: "Tep (bpm)", swatch: "#f43f5e" },
  { k: "spd", label: "Rychlost (km/h)", swatch: "#38bdf8" },
  { k: "el", label: "Výška", swatch: "#a3e635" },
] as const;

function ChartCard({
  chart,
  hidden,
  onToggle,
  theme,
}: {
  chart: ReturnType<typeof buildChart>;
  hidden: Record<string, boolean>;
  onToggle: (k: string) => void;
  theme: (typeof THEMES)["dark"];
}) {
  const opacity = (k: string) => (hidden[k] ? 0 : 1);

  return (
    <section style={{ ...CARD, padding: 12, gap: 10 }}>
      <span style={SECTION_LABEL}>Tep × rychlost × výška</span>
      <div style={{ display: "flex", gap: 9, flexWrap: "wrap" }}>
        {SERIES.map((s) => (
          <button
            key={s.k}
            type="button"
            onClick={() => onToggle(s.k)}
            style={mono(11.5, {
              display: "flex",
              alignItems: "center",
              gap: 7,
              padding: "5px 11px",
              borderRadius: 999,
              border: `1px solid ${hidden[s.k] ? "transparent" : "var(--line2)"}`,
              background: hidden[s.k] ? "transparent" : "var(--track)",
              color: hidden[s.k] ? "var(--faint)" : "var(--fg)",
              cursor: "pointer",
              transition: ".2s",
            })}
          >
            <span style={{ width: 14, height: 3, borderRadius: 2, background: hidden[s.k] ? "var(--faint)" : s.swatch }} />
            {s.label}
          </button>
        ))}
      </div>

      <div style={{ position: "relative", padding: "0 32px 0 28px" }}>
        <div style={{ position: "absolute", left: 0, top: 0, width: 26, height: 180 }}>
          <span style={mono(9.5, { position: "absolute", left: 0, top: -14, color: theme.bad })}>bpm</span>
          {chart.hrAxis.map((a) => (
            <span key={a.label} style={mono(10.5, { position: "absolute", left: 0, top: a.top, transform: "translateY(-50%)", color: theme.bad })}>
              {a.label}
            </span>
          ))}
        </div>
        <div style={{ position: "absolute", right: 0, top: 0, width: 32, height: 180 }}>
          <span style={mono(9.5, { position: "absolute", right: 0, top: -14, color: theme.blue })}>km/h</span>
          {chart.spdAxis.map((a) => (
            <span key={a.label} style={mono(10.5, { position: "absolute", right: 0, top: a.top, transform: "translateY(-50%)", color: theme.blue })}>
              {a.label}
            </span>
          ))}
        </div>

        {chart.hr == null && chart.spd == null && chart.el == null ? (
          <div style={{ height: 180, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <span style={mono(11, { color: "var(--faint)" })}>Jízda nemá dost vteřinových dat</span>
          </div>
        ) : (
          <svg viewBox="0 0 700 200" preserveAspectRatio="none" style={{ width: "100%", height: 180, display: "block" }}>
            <defs>
              <linearGradient id="hrFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity="0.36" />
                <stop offset="100%" stopColor="#f43f5e" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="spdFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.22" />
                <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
              </linearGradient>
              <linearGradient id="elFillDetail" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a3e635" stopOpacity="0.16" />
                <stop offset="100%" stopColor="#a3e635" stopOpacity="0" />
              </linearGradient>
            </defs>
            {chart.el && <path d={chart.el.area} fill="url(#elFillDetail)" opacity={opacity("el")} />}
            {chart.spd && <path d={chart.spd.area} fill="url(#spdFill)" opacity={opacity("spd")} />}
            {chart.hr && <path d={chart.hr.area} fill="url(#hrFill)" opacity={opacity("hr")} />}
            {chart.el && <path d={chart.el.line} fill="none" stroke="#a3e635" strokeWidth="1.4" strokeLinejoin="round" vectorEffect="non-scaling-stroke" opacity={opacity("el")} />}
            {chart.spd && <path d={chart.spd.line} fill="none" stroke="#38bdf8" strokeWidth="1.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" opacity={opacity("spd")} />}
            {chart.hr && <path d={chart.hr.line} fill="none" stroke="#f43f5e" strokeWidth="1.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" opacity={opacity("hr")} />}
          </svg>
        )}
        <div style={{ display: "flex", justifyContent: "space-between", paddingTop: 8 }}>
          {chart.timeAxis.map((t, i) => (
            <span key={i} style={mono(10.5, { color: "var(--faint)" })}>{t}</span>
          ))}
        </div>
      </div>
    </section>
  );
}

// ── Stoupání (zatím jen placeholder) ─────────────────────────────────────

function ClimbPlaceholder() {
  return (
    <section style={{ ...CARD, padding: 20, gap: 8 }}>
      <span style={SECTION_LABEL}>Stoupání</span>
      <p style={NOTE}>
        Detekce kopců a rozdělení terénu (do kopce / z kopce / rovina) je nová odvozená vrstva nad
        vteřinovými daty – čeká na schválené schéma migrace, ne na dopočet za běhu. Zatím tu proto
        nic není.
      </p>
    </section>
  );
}
