import { useEffect, useMemo, useState } from "react";

import { fetchMetricHistory, type MetricHistoryPoint } from "./api";
import { RangeDropdown } from "./components/RangeDropdown";
import { Shell } from "./components/Shell";
import { StateScreen } from "./components/StateScreen";
import { mono } from "./components/ui";
import { useScrub } from "./components/useScrub";
import { buildMetricHistory, metricConfig, type MetricWhich } from "./derive/metricHistory";
import type { Range, RangeOption } from "./derive/ranges";
import { THEMES, type ThemeName } from "./theme";

const RANGE_OPTS: RangeOption[] = [
  { label: "1M", full: "1 měsíc", value: 30 },
  { label: "3M", full: "3 měsíce", value: 90 },
  { label: "6M", full: "6 měsíců", value: 180 },
  { label: "1R", full: "1 rok", value: 365 },
  { label: "Vše", full: "Celá historie", value: "all" as Range },
];

export function MetricDetail({
  which,
  theme,
  onBack,
}: {
  which: MetricWhich;
  theme: ThemeName;
  onBack: () => void;
}) {
  const T = THEMES[theme];

  if (which === "spanek") {
    return (
      <Shell>
        <BackHeader eyebrow="" title="Spánek" onBack={onBack} info={false} />
        <section style={{ border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "40px 28px", minHeight: 320 }} />
      </Shell>
    );
  }

  return <MetricPage which={which} T={T} onBack={onBack} />;
}

function MetricPage({
  which,
  T,
  onBack,
}: {
  which: MetricWhich;
  T: (typeof THEMES)["dark"];
  onBack: () => void;
}) {
  const cfg = useMemo(() => metricConfig(which), [which]);
  const [points, setPoints] = useState<MetricHistoryPoint[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [range, setRange] = useState<Range>(90);
  const [hide, setHide] = useState<Record<string, boolean>>({});
  const [devWindow, setDevWindow] = useState<7 | 30>(7);
  const [selIdx, setSelIdx] = useState<number | null>(null);
  const [scrubbing, setScrubbing] = useState(false);

  useEffect(() => {
    setPoints(null);
    setError(null);
    const ctrl = new AbortController();
    fetchMetricHistory(cfg.apiKey, 1825, ctrl.signal)
      .then((p) => setPoints(p.points))
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      });
    return () => ctrl.abort();
  }, [cfg.apiKey]);

  const rangeDays = range === "all" ? 0 : (range as number);
  const view = useMemo(
    () =>
      points
        ? buildMetricHistory(points, cfg, T, rangeDays, devWindow, selIdx, scrubbing)
        : null,
    [points, cfg, T, rangeDays, devWindow, selIdx, scrubbing],
  );

  const scrub = useScrub(scrubbing, setScrubbing, (f) => {
    if (!view || view.count < 2) return;
    setSelIdx(Math.round(f * (view.count - 1)));
  });
  const resetRange = () => setSelIdx(null);

  if (error) return <StateScreen title={`${cfg.eyebrow} se nepodařilo načíst`} detail={error} />;

  const v = view;
  const on = (k: string) => (hide[k] ? "0" : "1");
  const toggleStyle = (active: boolean) =>
    mono(10, {
      display: "flex",
      alignItems: "center",
      gap: 7,
      padding: "5px 10px",
      borderRadius: 999,
      border: "1px solid var(--line2)",
      background: "var(--track)",
      color: active ? "var(--fg2)" : "var(--faint)",
      cursor: "pointer",
      opacity: active ? 1 : 0.5,
    });

  return (
    <Shell>
      <BackHeader eyebrow={cfg.eyebrow} title={cfg.title} onBack={onBack} info />

      {/* Souhrn */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(150px,1fr))", gap: 12 }}>
        <SumCell label={`Průměr ${cfg.eyebrow}`} value={v?.avg ?? "…"} unit={cfg.unit} footLabel="vs min. období" foot={v?.deltaLabel} footColor={v?.deltaColor} />
        <SumCell label="Nejvyšší" value={v?.maxVal ?? "…"} unit={cfg.unit} valueColor="var(--ok)" footLabel="datum" foot={v?.maxDate} />
        <SumCell label="Nejnižší" value={v?.minVal ?? "…"} unit={cfg.unit} valueColor="var(--bad)" footLabel="datum" foot={v?.minDate} />
        <div style={CELL}>
          <span style={CELL_LABEL}>Trend</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={mono(26, { lineHeight: 1.05, letterSpacing: "-.03em", color: v?.trendColor ?? "var(--mut)" })}>{v?.trendLabel ?? "…"}</span>
            <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke={v?.trendColor ?? "var(--mut)"} strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" style={{ transform: `rotate(${v?.trendRotate ?? "0deg"})`, transition: "transform .3s" }}>
              <path d="M5 12h14M13 5l7 7-7 7" />
            </svg>
          </div>
          <div style={CELL_FOOT}>
            <span style={CELL_FOOT_LABEL}>za</span>
            <span style={mono(12, { color: "var(--fg2)", marginLeft: "auto", whiteSpace: "nowrap" })}>posledních 7 dní</span>
          </div>
        </div>
      </section>

      {/* Graf v čase */}
      <section style={CARD}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10 }}>
          <span style={EYEBROW}>{cfg.eyebrow} v čase</span>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <button
              type="button"
              onClick={() => { /* brush mode na této obrazovce zatím jen vizuální */ }}
              title="Vybrat časový úsek v grafu"
              style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 26, height: 26, borderRadius: 999, border: "1px solid var(--line2)", background: "var(--track)", color: "var(--fg2)", cursor: "pointer", flex: "none" }}
            >
              <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 12h4M16 12h4M9 5v14M15 5v14" />
              </svg>
            </button>
            <RangeDropdown options={RANGE_OPTS} value={range} onPick={(r) => { setRange(r); setSelIdx(null); }} theme={T} />
          </div>
        </div>

        <div style={{ display: "flex", gap: 20, flexWrap: "wrap" }}>
          <button type="button" onClick={() => setHide((h) => ({ ...h, raw: !h.raw }))} style={toggleStyle(!hide.raw)}>
            <span style={{ width: 8, height: 8, borderRadius: "50%", background: "var(--blue2)" }} />
            {cfg.eyebrow} ({cfg.unit})
          </button>
          <button type="button" onClick={() => setHide((h) => ({ ...h, a7: !h.a7 }))} style={toggleStyle(!hide.a7)}>
            <span style={{ width: 11, height: 0, borderTop: "2px solid var(--fg2)" }} />
            7denní průměr
          </button>
          <button type="button" onClick={() => setHide((h) => ({ ...h, a30: !h.a30 }))} style={toggleStyle(!hide.a30)}>
            <span style={{ width: 11, height: 0, borderTop: "2px dashed var(--blue)" }} />
            30denní průměr
          </button>
        </div>

        <div style={{ position: "relative", width: "100%", height: 220, paddingLeft: 34 }}>
          {(v?.yTicks ?? []).map((t) => (
            <span key={t.top} style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: t.top, transform: "translateY(-50%)" })}>{t.label}</span>
          ))}
          <svg viewBox="0 0 700 220" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}>
            <defs>
              <linearGradient id="hrvFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" style={{ stopColor: "var(--blue2)" }} stopOpacity="0.28" />
                <stop offset="100%" style={{ stopColor: "var(--blue2)" }} stopOpacity="0" />
              </linearGradient>
            </defs>
            {(v?.gridLines ?? []).map((g) => (
              <line key={g.y} x1="0" y1={g.y} x2="700" y2={g.y} style={{ stroke: "var(--line)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" />
            ))}
            {v && <path d={v.areaPath} fill="url(#hrvFill)" opacity={on("raw")} />}
            {v && <path d={v.avg30Path} fill="none" style={{ stroke: "var(--blue)" }} strokeWidth="2.4" strokeDasharray="7 5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" opacity={on("a30")} />}
            {v && <path d={v.avg7Path} fill="none" style={{ stroke: "var(--fg2)" }} strokeWidth="2.4" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" opacity={on("a7")} />}
            {v && <path d={v.linePath} fill="none" style={{ stroke: "var(--blue2)" }} strokeWidth="2.2" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" opacity={on("raw")} />}
            {v && <circle cx={v.peak.x} cy={v.peak.y} r="4" style={{ fill: "var(--ok)" }} opacity={on("raw")} />}
            {v && <circle cx={v.trough.x} cy={v.trough.y} r="4" style={{ fill: "var(--blue2)" }} opacity={on("raw")} />}
            {v && <circle cx={v.scrub.x} cy={v.scrub.y} r="4.5" style={{ fill: "var(--card)", stroke: "var(--blue2)" }} strokeWidth="2" opacity={v.scrub.opacity} />}
          </svg>
          {v && !hide.raw && (
            <>
              <div style={{ position: "absolute", top: v.peak.labelTop, left: `calc(34px + ${v.peak.labelLeft})`, transform: "translate(-50%,-100%)", padding: "4px 10px", borderRadius: 8, background: "var(--ok)", color: "var(--card)", ...mono(12, { fontWeight: 700 }), whiteSpace: "nowrap", pointerEvents: "none" }}>{v.maxVal}</div>
              <div style={{ position: "absolute", top: v.trough.labelTop, left: `calc(34px + ${v.trough.labelLeft})`, transform: "translate(-50%,4px)", padding: "4px 10px", borderRadius: 8, background: "var(--blue2)", color: "var(--card)", ...mono(12, { fontWeight: 700 }), whiteSpace: "nowrap", pointerEvents: "none" }}>{v.minVal}</div>
            </>
          )}
          {v?.scrub.label && (
            <div style={mono(9, { position: "absolute", top: 2, left: `calc(34px + ${v.scrub.left})`, transform: "translateX(-50%)", padding: "4px 8px", borderRadius: 999, background: "var(--track)", color: "var(--fg)", whiteSpace: "nowrap", pointerEvents: "none", opacity: v.scrub.opacity, transition: "opacity .2s" })}>{v.scrub.label}</div>
          )}
          <div {...scrub} onDoubleClick={resetRange} style={{ position: "absolute", inset: "0 0 0 34px", touchAction: "none", cursor: "crosshair" }} />
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: 34 }}>
          {(v?.xTicks ?? []).map((t, i) => (
            <span key={i} style={mono(10, { color: "var(--faint)" })}>{t}</span>
          ))}
        </div>
      </section>

      {/* Odchylka od průměru */}
      <section style={CARD}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
          <span style={EYEBROW}>Odchylka od průměru</span>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ display: "flex", background: "var(--track)", borderRadius: 999, padding: 2 }}>
              {([7, 30] as const).map((w) => (
                <button
                  key={w}
                  type="button"
                  onClick={() => setDevWindow(w)}
                  style={mono(10, { appearance: "none", border: "none", cursor: "pointer", padding: "6px 12px", borderRadius: 999, letterSpacing: ".06em", background: devWindow === w ? "var(--card2)" : "transparent", color: devWindow === w ? "var(--fg)" : "var(--mut)" })}
                >
                  {w} dní
                </button>
              ))}
            </div>
            <span style={mono(12, { color: "var(--faint)", whiteSpace: "nowrap" })}>
              Dnes <span style={{ color: v?.todayDevColor ?? "var(--mut)" }}>{v?.todayDevLabel ?? "–"}</span>
            </span>
          </div>
        </div>
        <div style={{ position: "relative", width: "100%", height: 140, paddingLeft: 34 }}>
          {(v?.devYTicks ?? []).map((t) => (
            <span key={t.top} style={mono(9, { position: "absolute", left: 0, color: "var(--faint)", whiteSpace: "nowrap", top: t.top, transform: "translateY(-50%)" })}>{t.label}</span>
          ))}
          <svg viewBox="0 0 700 140" preserveAspectRatio="none" style={{ width: "100%", height: "100%", display: "block", overflow: "visible" }}>
            {(v?.devGridLines ?? []).map((g) => (
              <line key={g.y} x1="0" y1={g.y} x2="700" y2={g.y} style={{ stroke: "var(--line)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" />
            ))}
            {v && <line x1="0" y1={v.devZeroY} x2="700" y2={v.devZeroY} style={{ stroke: "var(--line2)" }} strokeWidth="1.4" vectorEffect="non-scaling-stroke" />}
            {(v?.devBars ?? []).map((b, i) => (
              <rect key={i} x={b.x} y={b.y} width={b.w} height={b.h} rx="1.5" style={{ fill: b.color }} opacity={b.op} />
            ))}
            {v && <line x1={v.scrub.x} y1="0" x2={v.scrub.x} y2="140" style={{ stroke: "var(--fg2)" }} strokeWidth="1" vectorEffect="non-scaling-stroke" opacity={v.scrub.opacity} />}
          </svg>
          {v?.devScrubLabel && (
            <div style={mono(9, { position: "absolute", top: 2, left: `calc(34px + ${v.scrub.left})`, transform: "translateX(-50%)", padding: "4px 8px", borderRadius: 999, background: "var(--track)", color: "var(--fg)", whiteSpace: "nowrap", pointerEvents: "none", opacity: v.scrub.opacity, transition: "opacity .2s" })}>{v.devScrubLabel}</div>
          )}
          <div {...scrub} onDoubleClick={resetRange} style={{ position: "absolute", inset: "0 0 0 34px", touchAction: "none", cursor: "crosshair" }} />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", paddingLeft: 34 }}>
          {(v?.xTicks ?? []).map((t, i) => (
            <span key={i} style={mono(10, { color: "var(--faint)" })}>{t}</span>
          ))}
        </div>
      </section>

      {/* Distribuce + rozpad podle dne */}
      <section style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit,minmax(280px,1fr))", gap: 20, alignItems: "stretch" }}>
        <div style={{ ...CARD, padding: "24px 28px" }}>
          <span style={EYEBROW}>Distribuce {cfg.eyebrow}</span>
          <div style={{ display: "flex", alignItems: "center", gap: 24, flexWrap: "wrap" }}>
            <svg viewBox="0 0 120 120" width="132" height="132" style={{ flex: "none", transform: "rotate(-90deg)" }}>
              {(v?.donut ?? []).map((d, i) => (
                <circle key={i} cx="60" cy="60" r="46" fill="none" stroke={d.color} strokeWidth="18" strokeDasharray={d.dash} strokeDashoffset={d.offset} />
              ))}
            </svg>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {(v?.donutLegend ?? []).map((l) => (
                <div key={l.label} style={{ display: "flex", alignItems: "center", gap: 10 }}>
                  <span style={{ width: 9, height: 9, borderRadius: "50%", background: l.color }} />
                  <span style={mono(12, { color: "var(--fg2)", minWidth: 76 })}>{l.label}</span>
                  <span style={mono(12, { color: "var(--mut)" })}>{l.pct} %</span>
                </div>
              ))}
            </div>
          </div>
          <span style={{ fontSize: 12, color: "var(--mut)" }}>Ideální rozsah: <span style={{ color: "var(--blue)" }}>{v?.idealText}</span></span>
        </div>

        <div style={{ ...CARD, padding: "24px 28px" }}>
          <span style={EYEBROW}>{cfg.eyebrow} podle dne</span>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            {(v?.weekdays ?? []).map((w) => (
              <div key={w.name} style={{ display: "grid", gridTemplateColumns: "88px 70px 1fr", gap: 12, alignItems: "center" }}>
                <span style={{ fontSize: 13, color: "var(--fg2)" }}>{w.name}</span>
                <span style={mono(13, { whiteSpace: "nowrap" })}>
                  <b>{w.value}</b> <span style={{ color: "var(--faint)", fontWeight: 400 }}>{cfg.unit}</span>
                </span>
                <div style={{ display: "flex", gap: 4 }}>
                  {w.squares.map((sq, i) => (
                    <span key={i} style={{ flex: 1, height: 12, borderRadius: 2, background: sq.color }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* O metrice */}
      <section style={{ ...CARD, padding: "24px 28px", gap: 8 }}>
        <span style={EYEBROW}>O {cfg.eyebrow}</span>
        <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6, color: "var(--faint)", textWrap: "pretty" }}>
          {cfg.about[0]}
          <br />
          {cfg.about[1]}
        </p>
      </section>
    </Shell>
  );
}

const CARD = { border: "1px solid var(--line)", borderRadius: 22, background: "var(--card)", padding: "24px 24px 20px", display: "flex", flexDirection: "column", gap: 14 } as const;
const EYEBROW = mono(10, { letterSpacing: ".16em", textTransform: "uppercase", color: "var(--mut)" });
const CELL = { border: "1px solid var(--line)", borderRadius: 18, background: "var(--card)", padding: 20, display: "flex", flexDirection: "column", gap: 6, minWidth: 0 } as const;
const CELL_LABEL = mono(9, { letterSpacing: ".12em", textTransform: "uppercase", color: "var(--mut)" });
const CELL_FOOT = { display: "flex", alignItems: "baseline", gap: 8, paddingTop: 8, marginTop: 2, borderTop: "1px solid var(--line)" } as const;
const CELL_FOOT_LABEL = mono(9, { letterSpacing: ".08em", textTransform: "uppercase", color: "var(--mut)" });

function SumCell({
  label,
  value,
  unit,
  valueColor,
  footLabel,
  foot,
  footColor,
}: {
  label: string;
  value: string;
  unit: string;
  valueColor?: string;
  footLabel: string;
  foot?: string;
  footColor?: string;
}) {
  return (
    <div style={CELL}>
      <span style={CELL_LABEL}>{label}</span>
      <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
        <span style={mono(26, { lineHeight: 1.05, letterSpacing: "-.03em", color: valueColor ?? "var(--fg)" })}>{value}</span>
        <span style={mono(10, { color: "var(--faint)" })}>{unit}</span>
      </div>
      <div style={CELL_FOOT}>
        <span style={CELL_FOOT_LABEL}>{footLabel}</span>
        <span style={mono(12, { color: footColor ?? "var(--fg2)", marginLeft: "auto", whiteSpace: "nowrap" })}>{foot ?? "…"}</span>
      </div>
    </div>
  );
}

function BackHeader({
  eyebrow,
  title,
  onBack,
  info,
}: {
  eyebrow: string;
  title: string;
  onBack: () => void;
  info: boolean;
}) {
  return (
    <header style={{ display: "flex", alignItems: "center", gap: 14, padding: "2px 2px 6px" }}>
      <button
        type="button"
        onClick={onBack}
        aria-label="Zpět"
        className="hover-fg"
        style={{ display: "flex", alignItems: "center", justifyContent: "center", width: 34, height: 34, borderRadius: "50%", border: "1px solid var(--line)", background: "none", color: "var(--fg2)", flex: "none", cursor: "pointer" }}
      >
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M15 18l-6-6 6-6" />
        </svg>
      </button>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {eyebrow && <span style={mono(11, { letterSpacing: ".14em", textTransform: "uppercase", color: "var(--mut)" })}>{eyebrow}</span>}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h1 style={{ margin: 0, fontSize: 22, lineHeight: 1.1, fontWeight: 500, letterSpacing: "-.02em" }}>{title}</h1>
          {info && (
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="var(--mut)" strokeWidth="1.7">
              <circle cx="12" cy="12" r="9" />
              <path d="M12 11v5M12 8h.01" strokeLinecap="round" />
            </svg>
          )}
        </div>
      </div>
    </header>
  );
}
