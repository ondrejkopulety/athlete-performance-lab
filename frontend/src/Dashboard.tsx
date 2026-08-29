import { useMemo, useRef, useState } from "react";

import { toDays, type DashboardPayload } from "./api";
import { DailyStatus } from "./components/DailyStatus";
import { Header } from "./components/Header";
import { HrBlocks } from "./components/HrBlocks";
import { HrCurve } from "./components/HrCurve";
import { LoadBalance } from "./components/LoadBalance";
import { Quality } from "./components/Quality";
import { Rides } from "./components/Rides";
import { StateScreen } from "./components/StateScreen";
import { buildClimb } from "./derive/climb";
import { buildGauges } from "./derive/gauges";
import { buildHrBlocks } from "./derive/hrblocks";
import { buildHrCurve } from "./derive/hrcurve";
import { buildHrr } from "./derive/hrr";
import { buildPmc } from "./derive/pmc";
import { buildPolarization, buildZoneTime, inRange } from "./derive/quality";
import { buildRides, ridesSummary } from "./derive/rides";
import { rangeOptions, selectWindow, type Range } from "./derive/ranges";
import { THEMES, type ThemeName } from "./theme";
import { useHrPanels } from "./useHrPanels";

/**
 * Celý pohled nad načtenými daty. Stav je většinou jen zobrazovací (období,
 * vybraný den, otevřená karta jízdy).
 *
 * Výjimkou jsou panely tepové křivky a souvislých bloků: jsou to agregace
 * přes období a přes filtr "jen úplná data", takže se na server doptávají
 * (viz useHrPanels).
 */
export function Dashboard({
  payload,
  theme,
  mounted,
  onToggleTheme,
  onOpenActivity,
}: {
  payload: DashboardPayload;
  theme: ThemeName;
  /** Až po prvním vykreslení se rozjedou animace kroužků a pruhů. */
  mounted: boolean;
  onToggleTheme: () => void;
  onOpenActivity: (id: string) => void;
}) {
  const T = THEMES[theme];
  const days = useMemo(() => toDays(payload.days), [payload]);

  const [range, setRange] = useState<Range>(30);
  const [sel, setSel] = useState<number>(days.length - 1);
  const [scrubbing, setScrubbing] = useState(false);
  const [hrrSel, setHrrSel] = useState<number | null>(null);
  const [hrrScrub, setHrrScrub] = useState(false);
  const [openRide, setOpenRide] = useState<string | null>(null);

  // Panely křivky a bloků mají vlastní období – přepínají se stejnými
  // taby, ale ptají se serveru, takže si drží i vlastní filtry.
  const [hrRange, setHrRange] = useState<Range>(90);
  const [completeOnly, setCompleteOnly] = useState(true);
  const [compare, setCompare] = useState<"prev" | "year">("prev");
  const [tolerance, setTolerance] = useState(15);
  const [thresholdVersion, setThresholdVersion] = useState(0);

  const window = useMemo(() => selectWindow(days, range), [days, range]);
  const ranges = useMemo(() => rangeOptions(days), [days]);
  const { rows, startIdx } = window;

  const hrWindow = useMemo(() => selectWindow(days, hrRange), [days, hrRange]);
  const hrPanels = useHrPanels(
    hrWindow.rows.length ? hrWindow.rows[0].d : null,
    hrWindow.rows.length ? hrWindow.rows[hrWindow.rows.length - 1].d : null,
    completeOnly,
    compare,
    tolerance,
    thresholdVersion,
  );

  const selIdxAbs = Math.min(
    Math.max(sel, startIdx),
    startIdx + Math.max(0, rows.length - 1),
  );

  // Aktuální okno drží ref, aby ho tahání po grafu vidělo bez re-renderu.
  const winRef = useRef({ start: startIdx, len: rows.length });
  winRef.current = { start: startIdx, len: rows.length };

  const pmc = useMemo(
    () => buildPmc(rows, startIdx, selIdxAbs, days, T),
    [rows, startIdx, selIdxAbs, days, T],
  );

  const periodActivities = useMemo(
    () => (rows.length ? inRange(payload.activities, rows[0].d, rows[rows.length - 1].d) : []),
    [payload, rows],
  );

  const polarization = useMemo(
    () => buildPolarization(periodActivities, T, mounted),
    [periodActivities, T, mounted],
  );
  const zoneTime = useMemo(
    () => buildZoneTime(periodActivities, T, mounted),
    [periodActivities, T, mounted],
  );
  const climb = useMemo(
    () => buildClimb(periodActivities, payload.activities, T, mounted),
    [periodActivities, payload, T, mounted],
  );
  const hrr = useMemo(
    () => buildHrr(periodActivities, T, hrrSel, hrrScrub),
    [periodActivities, T, hrrSel, hrrScrub],
  );
  const rideCards = useMemo(
    () => buildRides(payload.rides, T, theme === "light"),
    [payload, T, theme],
  );
  const gauges = useMemo(
    () => buildGauges(T, payload.today, payload.last_known, mounted),
    [payload, T, mounted],
  );
  const curveView = useMemo(
    () => buildHrCurve(hrPanels.curve, T),
    [hrPanels.curve, T],
  );
  const blocksView = useMemo(
    () => buildHrBlocks(hrPanels.blocks, T, mounted),
    [hrPanels.blocks, T, mounted],
  );

  if (days.length === 0) {
    return (
      <StateScreen
        title="V databázi nejsou žádné denní metriky"
        detail="Spusť pipeline: python scripts/main.py"
      />
    );
  }

  const today = payload.today;
  const readiness = today?.readiness_score ?? null;
  const readyColor =
    readiness == null ? T.mut : readiness >= 66 ? T.ok : readiness >= 40 ? T.warn : T.bad;

  const pickRange = (value: Range) => {
    setRange(value);
    setSel(days.length - 1);
  };

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
          maxWidth: 1120,
          margin: "0 auto",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <Header
          today={today}
          readiness={readiness}
          statusDot={readyColor}
          theme={theme}
          threshold={hrPanels.threshold}
          onThresholdSaved={(next) => {
            hrPanels.setThreshold(next);
            // Nový práh = jiný řádek mřížky, ne přepočet dat.
            setThresholdVersion((v) => v + 1);
          }}
          onToggleTheme={onToggleTheme}
        />

        <section
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(300px,1fr))",
            gap: 20,
            alignItems: "stretch",
          }}
        >
          <DailyStatus
            readiness={readiness}
            readyColor={readyColor}
            mounted={mounted}
            gauges={gauges}
            advice={today?.coach_advice ?? null}
          />

          <LoadBalance
            pmc={pmc}
            ranges={ranges}
            range={range}
            rangeLabel={window.label}
            theme={T}
            onPickRange={pickRange}
            onSelectDay={setSel}
            onScrub={(fraction) => {
              const w = winRef.current;
              if (w.len < 2) return;
              setSel(w.start + Math.round(fraction * (w.len - 1)));
            }}
            scrubbing={scrubbing}
            onScrubbingChange={(active) => {
              setScrubbing(active);
              // Po puštění se výběr vrací na poslední den okna, jako v designu.
              if (!active) {
                const w = winRef.current;
                setSel(w.len ? w.start + w.len - 1 : days.length - 1);
              }
            }}
          />
        </section>

        <Quality
          ranges={ranges}
          range={range}
          rangeLabel={window.label}
          onPickRange={pickRange}
          theme={T}
          polarization={polarization}
          zoneTime={zoneTime}
          climb={climb}
          hrr={hrr}
          onHrrScrub={(fraction) => {
            if (hrr.count < 2) return;
            setHrrSel(Math.round(fraction * (hrr.count - 1)));
          }}
          hrrScrubbing={hrrScrub}
          onHrrScrubbingChange={(active) => {
            setHrrScrub(active);
            if (!active) setHrrSel(null);
          }}
        />

        <HrCurve
          view={curveView}
          ranges={ranges}
          range={hrRange}
          rangeLabel={hrWindow.label}
          onPickRange={setHrRange}
          completeOnly={completeOnly}
          onToggleComplete={setCompleteOnly}
          compare={compare}
          onToggleCompare={setCompare}
          loading={hrPanels.loading}
          error={hrPanels.error}
          theme={T}
        />

        <HrBlocks
          view={blocksView}
          ranges={ranges}
          range={hrRange}
          rangeLabel={hrWindow.label}
          onPickRange={setHrRange}
          completeOnly={completeOnly}
          onToggleComplete={setCompleteOnly}
          tolerance={tolerance}
          onPickTolerance={setTolerance}
          loading={hrPanels.loading}
          error={hrPanels.error}
          theme={T}
        />

        <Rides
          rides={rideCards}
          summary={ridesSummary(payload.rides)}
          openId={openRide}
          onToggle={(id) => setOpenRide((cur) => (cur === id ? null : id))}
          onOpenDetail={onOpenActivity}
          theme={T}
        />
      </div>
    </div>
  );
}
