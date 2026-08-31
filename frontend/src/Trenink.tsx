import { useMemo, useState } from "react";

import { toDays, type DashboardPayload } from "./api";
import { BlockPanel } from "./components/BlockPanel";
import { RangeDropdown } from "./components/RangeDropdown";
import { Shell } from "./components/Shell";
import { StateScreen } from "./components/StateScreen";
import { mono } from "./components/ui";
import { useScrub } from "./components/useScrub";
import { buildClimb } from "./derive/climb";
import { buildHrBlocks } from "./derive/hrblocks";
import { buildHrr } from "./derive/hrr";
import { buildPolarization, buildZoneTime, inRange } from "./derive/quality";
import { rangeOptions, selectWindow, type Range } from "./derive/ranges";
import { THEMES, type ThemeName } from "./theme";
import { useBlockPanels } from "./useHrPanels";

/**
 * Trénink – 1:1 s Trénink.dc.html: jedna karta „Kvalita tréninku" s mřížkou
 * šesti bloků (polarizace, junk miles, tepová regenerace, VAM, souvislé
 * bloky Z4, souvislé bloky Z2) a sekcí „Čas v zónách".
 *
 * Čísla se pořád odvozují na klientu z `payload.activities` přes zvolené
 * období; jen souvislé bloky se dotahují ze serveru (`useBlockPanels`).
 */
export function Trenink({
  payload,
  theme,
  mounted,
  hidden,
  thresholdVersion,
  onOpenActivity,
}: {
  payload: DashboardPayload;
  theme: ThemeName;
  mounted: boolean;
  hidden?: boolean;
  thresholdVersion: number;
  onOpenActivity: (id: string) => void;
}) {
  const T = THEMES[theme];
  const days = useMemo(() => toDays(payload.days), [payload]);

  const [range, setRange] = useState<Range>(90);
  const [hrrSel, setHrrSel] = useState<number | null>(null);
  const [hrrScrub, setHrrScrub] = useState(false);

  const [z4Tol, setZ4Tol] = useState(0);
  const [z4Complete, setZ4Complete] = useState(true);
  const [z2Tol, setZ2Tol] = useState(0);
  const [z2Complete, setZ2Complete] = useState(true);

  const ranges = useMemo(() => rangeOptions(days), [days]);
  const window = useMemo(() => selectWindow(days, range), [days, range]);
  const { rows } = window;

  const blockReqs = useMemo(
    () => [
      { zone: "Z4", completeOnly: z4Complete, tolerance: z4Tol },
      { zone: "Z2", completeOnly: z2Complete, tolerance: z2Tol },
    ],
    [z4Complete, z4Tol, z2Complete, z2Tol],
  );
  const blocks = useBlockPanels(
    rows.length ? rows[0].d : null,
    rows.length ? rows[rows.length - 1].d : null,
    blockReqs,
    thresholdVersion,
  );

  const periodActivities = useMemo(
    () => (rows.length ? inRange(payload.activities, rows[0].d, rows[rows.length - 1].d) : []),
    [payload, rows],
  );

  const pol = useMemo(() => buildPolarization(periodActivities, T, mounted), [periodActivities, T, mounted]);
  const zoneTime = useMemo(() => buildZoneTime(periodActivities, T, mounted), [periodActivities, T, mounted]);
  const climb = useMemo(
    () => buildClimb(periodActivities, payload.activities, T, mounted),
    [periodActivities, payload, T, mounted],
  );
  const hrr = useMemo(() => buildHrr(periodActivities, T, hrrSel, hrrScrub), [periodActivities, T, hrrSel, hrrScrub]);
  const blkZ4 = useMemo(() => buildHrBlocks(blocks.byZone.Z4 ?? null, T, mounted), [blocks.byZone, T, mounted]);
  const blkZ2 = useMemo(() => buildHrBlocks(blocks.byZone.Z2 ?? null, T, mounted), [blocks.byZone, T, mounted]);

  const scrub = useScrub(hrrScrub, setHrrScrub, (f) => {
    if (hrr.count < 2) return;
    setHrrSel(Math.round(f * (hrr.count - 1)));
  });

  if (days.length === 0) {
    return (
      <StateScreen title="V databázi nejsou žádné denní metriky" detail="Spusť pipeline: python scripts/main.py" />
    );
  }

  const pct = (v: number) => v.toFixed(0);
  const eyebrow = mono(10, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" });
  const bigNum = (color: string) => mono(30, { lineHeight: 1, color });

  return (
    <Shell hidden={hidden}>
      <header style={{ display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 16, flexWrap: "wrap", padding: "2px 2px 6px" }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>TRÉNINK</span>
          <h1 style={{ margin: 0, fontSize: 26, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>Kvalita tréninku</h1>
        </div>
      </header>

      <section style={{ border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "24px 28px", display: "flex", flexDirection: "column", gap: 16 }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <span style={{ ...eyebrow, letterSpacing: ".16em" }}>Kvalita tréninku</span>
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span style={mono(11, { color: "var(--faint)" })}>průměr · {window.label}</span>
            <RangeDropdown options={ranges} value={range} onPick={setRange} theme={T} />
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))", gap: "32px 40px", alignItems: "start" }}>
          {/* Polarizace · Z1–Z2 */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
            <span style={eyebrow}>Polarizace · Z1–Z2</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={bigNum(pol.polColor)}>{pol.hasData ? pct(pol.low) : "–"}</span>
              <span style={mono(13, { color: "var(--faint)" })}>%</span>
              <span style={mono(10, { marginLeft: "auto", color: "var(--mut)" })}>cíl ≥ 75 %</span>
            </div>
            <div style={{ display: "flex", height: 6, borderRadius: 999, overflow: "hidden", background: "var(--track)", gap: 2 }}>
              <div style={{ width: pol.lowW, background: "var(--blue)" }} />
              <div style={{ width: pol.junkW, background: "var(--warn)" }} />
              <div style={{ width: pol.highW, background: "var(--bad)" }} />
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <Legend color="var(--blue)" text={`Z1–2 ${pct(pol.low)} %`} />
              <Legend color="var(--warn)" text={`Z3 ${pct(pol.junk)} %`} />
              <Legend color="var(--bad)" text={`Z4–5 ${pct(pol.high)} %`} />
            </div>
            <p style={NOTE}>Čas ve striktní Z1–Z2 — základ vytrvalosti.</p>
          </div>

          {/* Junk miles · Z3 */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
            <span style={eyebrow}>Junk miles · Z3</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={bigNum(pol.junkColor)}>{pol.hasData ? pct(pol.junk) : "–"}</span>
              <span style={mono(13, { color: "var(--faint)" })}>%</span>
              <span style={mono(10, { marginLeft: "auto", color: "var(--mut)" })}>cíl ≤ 15 %</span>
            </div>
            <div style={{ position: "relative", height: 6, borderRadius: 999, background: "var(--track)", overflow: "visible" }}>
              <div style={{ height: "100%", width: pol.junkW30, borderRadius: 999, background: pol.junkColor, transition: "width .8s cubic-bezier(.22,1,.36,1)" }} />
              <span style={{ position: "absolute", top: -4, left: "50%", width: 1.5, height: 15, background: "var(--faint)" }} />
            </div>
            <p style={NOTE}>Šedá zóna — moc těžké na regeneraci, moc lehké na rychlost.</p>
          </div>

          {/* Tepová regenerace · 60 s */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
            <span style={eyebrow}>Tepová regenerace · 60 s</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={bigNum(hrr.color)}>{hrr.last}</span>
              <span style={mono(13, { color: "var(--faint)" })}>{hrr.unit}</span>
              <span style={mono(10, { marginLeft: "auto", color: hrr.trendColor })}>{hrr.trend}</span>
            </div>
            <div style={{ position: "relative", width: "100%", height: 72, paddingLeft: 32 }}>
              <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}>{hrr.hi}</span>
              <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", bottom: -1 })}>{hrr.lo}</span>
              <svg viewBox="0 0 320 72" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}>
                <defs>
                  <linearGradient id="hrrFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" style={{ stopColor: "var(--ok)" }} stopOpacity="0.2" />
                    <stop offset="100%" style={{ stopColor: "var(--ok)" }} stopOpacity="0" />
                  </linearGradient>
                </defs>
                <rect x="0" y={hrr.goodY} width="320" height={hrr.goodH} style={{ fill: "var(--ok)", fillOpacity: 0.08 }} />
                <path d={hrr.area} fill="url(#hrrFill)" />
                <path d={hrr.line} fill="none" style={{ stroke: "var(--grey)" }} strokeWidth="1.5" strokeLinejoin="round" vectorEffect="non-scaling-stroke" />
                <path d={hrr.trendLine} fill="none" style={{ stroke: "var(--ok)" }} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
                <line x1={hrr.selX} y1="0" x2={hrr.selX} y2="72" style={{ stroke: "var(--grey)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" opacity={hrr.crossOpacity} />
                <circle cx={hrr.lastX} cy={hrr.lastY} r="3.5" style={{ fill: "var(--card)", stroke: "var(--ok)" }} strokeWidth="2" vectorEffect="non-scaling-stroke" />
              </svg>
              <div {...scrub} style={{ position: "absolute", inset: "0 0 0 30px", touchAction: "none", cursor: "crosshair" }}>
                <div style={mono(9, { position: "absolute", top: -2, left: hrr.selPct, transform: "translateX(-50%)", padding: "4px 8px", borderRadius: 999, background: "var(--track)", color: "var(--fg)", whiteSpace: "nowrap", pointerEvents: "none", opacity: hrr.crossOpacity, transition: "opacity .2s" })}>
                  {hrr.selLabel}
                </div>
              </div>
            </div>
            <div style={mono(9, { display: "flex", justifyContent: "space-between", paddingLeft: 32, color: "var(--mut)" })}>
              <span>{hrr.from}</span>
              <span>klouzavý průměr · dobré ≥ 50</span>
              <span>{hrr.to}</span>
            </div>
            <p style={NOTE}>Pokles tepu první minutu po zátěži.</p>
          </div>

          {/* Stoupání · VAM */}
          <div style={{ display: "flex", flexDirection: "column", gap: 12, padding: 2 }}>
            <span style={eyebrow}>Stoupání · VAM</span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
              <span style={bigNum(climb.vamColor)}>{climb.hasData ? climb.vam : "–"}</span>
              <span style={mono(13, { color: "var(--faint)" })}>m/h</span>
              <span style={mono(10, { marginLeft: "auto", color: climb.vamTrendColor })}>{climb.vamTrend}</span>
            </div>
            <div style={{ position: "relative", display: "flex", alignItems: "flex-end", gap: 2, height: 44, paddingLeft: 40 }}>
              <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: 0 })}>{climb.vamMax}</span>
              <span style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", bottom: -1 })}>0</span>
              {climb.bars.map((b, i) => (
                <div key={i} style={{ flex: 1, height: b.h, minHeight: 3, borderRadius: 3, background: b.color, transition: "height .6s cubic-bezier(.22,1,.36,1)" }} />
              ))}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, paddingTop: 2 }}>
              <Sub label="Do kopce" value={climb.uphill} />
              <Sub label="Prům. sklon" value={`${climb.grad} %`} />
            </div>
            <p style={NOTE}>Rychlost stoupání — odraz poměru výkon/váha.</p>
          </div>

          {/* Souvislé bloky · nad Z4 */}
          <BlockPanel
            title="Souvislé bloky · nad Z4"
            view={blkZ4}
            barColor="var(--orange)"
            tolerance={z4Tol}
            onPickTolerance={setZ4Tol}
            completeOnly={z4Complete}
            onToggleComplete={setZ4Complete}
            note="Nejdelší úseky nepřerušené jízdy nad prahem."
            onOpenActivity={onOpenActivity}
            activityId={blocks.byZone.Z4?.longest_block?.activity_id ?? null}
          />

          {/* Souvislé bloky · Z2 */}
          <BlockPanel
            title="Souvislé bloky · Z2"
            view={blkZ2}
            barColor="var(--blue)"
            tolerance={z2Tol}
            onPickTolerance={setZ2Tol}
            completeOnly={z2Complete}
            onToggleComplete={setZ2Complete}
            note="Nejdelší souvislé úseky ve vytrvalostní zóně 2 — čas na budování aerobní základny."
            onOpenActivity={onOpenActivity}
            activityId={blocks.byZone.Z2?.longest_block?.activity_id ?? null}
          />
        </div>

        <div style={{ height: 1, background: "var(--line)" }} />

        {/* Čas v zónách */}
        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <span style={eyebrow}>Čas v zónách</span>
            <span style={mono(11, { color: "var(--faint)" })}>{zoneTime.total} · {zoneTime.rides}</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {zoneTime.rows.map((z) => (
              <div key={z.name} style={{ display: "grid", gridTemplateColumns: "96px 1fr 74px 46px", gap: 12, alignItems: "center" }}>
                <span style={mono(10, { display: "flex", alignItems: "center", gap: 8, letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)", whiteSpace: "nowrap" })}>
                  <span style={{ width: 8, height: 8, borderRadius: 2, background: z.color }} />
                  {z.name}
                </span>
                <div style={{ height: 8, borderRadius: 999, background: "var(--track)", overflow: "hidden" }}>
                  <div style={{ height: "100%", width: z.w, borderRadius: 999, background: z.color, transition: "width .7s cubic-bezier(.22,1,.36,1)" }} />
                </div>
                <span style={mono(12, { color: "var(--fg)", textAlign: "right", whiteSpace: "nowrap" })}>{z.time}</span>
                <span style={mono(11, { color: "var(--faint)", textAlign: "right" })}>{z.pct} %</span>
              </div>
            ))}
          </div>
        </div>

        {blocks.error && (
          <p style={{ ...NOTE, color: T.bad }}>Panel bloků se nepodařilo načíst: {blocks.error}</p>
        )}
      </section>
    </Shell>
  );
}

const NOTE = { margin: 0, fontSize: 11, lineHeight: 1.5, color: "var(--faint)", textWrap: "pretty" } as const;

function Legend({ color, text }: { color: string; text: string }) {
  return (
    <span style={mono(9, { display: "flex", alignItems: "center", gap: 6, color: "var(--mut)" })}>
      <span style={{ width: 8, height: 8, borderRadius: 2, background: color }} />
      {text}
    </span>
  );
}

function Sub({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span style={mono(9, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>{label}</span>
      <span style={mono(14, { whiteSpace: "nowrap" })}>{value}</span>
    </div>
  );
}
