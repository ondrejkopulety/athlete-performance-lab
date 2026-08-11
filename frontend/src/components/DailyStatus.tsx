import type { Gauge } from "../derive/gauges";
import { CARD, mono, SECTION_LABEL, SMALL_LABEL } from "./ui";

const RING_C = 2 * Math.PI * 86;

export function DailyStatus({
  readiness,
  readyColor,
  mounted,
  gauges,
  advice,
}: {
  readiness: number | null;
  readyColor: string;
  mounted: boolean;
  gauges: Gauge[];
  advice: string | null;
}) {
  const verdict =
    readiness == null
      ? "Bez dat"
      : readiness >= 66
        ? "Připraven"
        : readiness >= 40
          ? "Lehký trénink"
          : "Nízká regenerace";

  const dash = `${(mounted && readiness != null ? (readiness / 100) * RING_C : 0).toFixed(1)} ${RING_C.toFixed(1)}`;

  return (
    <div
      style={{
        ...CARD,
        gridColumn: "span 1",
        padding: "28px 26px",
        alignItems: "center",
        gap: 18,
        minWidth: 0,
      }}
    >
      <div style={{ ...SECTION_LABEL, alignSelf: "flex-start" }}>Denní stav</div>

      <div
        style={{
          position: "relative",
          width: 212,
          height: 212,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg viewBox="0 0 200 200" style={{ width: "100%", height: "100%", transform: "rotate(-90deg)" }}>
          <circle cx="100" cy="100" r="86" fill="none" style={{ stroke: "var(--track)" }} strokeWidth="12" />
          <circle
            cx="100"
            cy="100"
            r="86"
            fill="none"
            stroke={readyColor}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={dash}
            style={{ transition: "stroke-dasharray .9s cubic-bezier(.22,1,.36,1)" }}
          />
        </svg>
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
          }}
        >
          <span
            style={mono(56, {
              fontWeight: 500,
              lineHeight: 1,
              letterSpacing: "-.03em",
              color: readyColor,
            })}
          >
            {readiness == null ? "–" : Math.round(mounted ? readiness : 0)}
          </span>
          <span
            style={mono(10, {
              letterSpacing: ".18em",
              textTransform: "uppercase",
              color: "var(--mut)",
            })}
          >
            {verdict}
          </span>
        </div>
      </div>

      <div
        style={{
          width: "100%",
          display: "grid",
          gridTemplateColumns: "repeat(2,1fr)",
          gap: "18px 10px",
        }}
      >
        {gauges.map((g) => (
          <div
            key={g.name}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 9,
              padding: "4px 2px",
            }}
          >
            <span style={{ ...SMALL_LABEL, fontSize: 9.5, color: g.nameColor }}>{g.name}</span>

            <div style={{ position: "relative", width: "100%", maxWidth: 74, aspectRatio: "1" }}>
              <svg
                viewBox="0 0 68 68"
                style={{ width: "100%", height: "100%", transform: "rotate(135deg)", overflow: "visible" }}
              >
                <circle
                  cx="34"
                  cy="34"
                  r="26"
                  fill="none"
                  style={{ stroke: "var(--track)" }}
                  strokeWidth="5"
                  strokeLinecap="round"
                  strokeDasharray={g.trackDash}
                />
                <circle
                  cx="34"
                  cy="34"
                  r="26"
                  fill="none"
                  stroke={g.bandColor}
                  strokeWidth="5"
                  strokeDasharray={g.bandDash}
                />
                <circle
                  cx="34"
                  cy="34"
                  r="26"
                  fill="none"
                  stroke={g.color}
                  strokeWidth="5"
                  strokeLinecap="round"
                  strokeDasharray={g.valDash}
                  style={{ transition: "stroke-dasharray .9s cubic-bezier(.22,1,.36,1)" }}
                />
              </svg>
              <svg
                viewBox="0 0 68 68"
                style={{ position: "absolute", inset: 0, width: "100%", height: "100%", overflow: "visible" }}
              >
                <line
                  x1={g.bx1}
                  y1={g.by1}
                  x2={g.bx2}
                  y2={g.by2}
                  style={{ stroke: "var(--faint)" }}
                  strokeWidth="1.5"
                />
              </svg>
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <span style={mono(16, { color: g.color })}>{g.value}</span>
              </div>
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 2,
                textAlign: "center",
              }}
            >
              <span style={{ fontSize: 10, color: g.deltaColor, lineHeight: 1.25, textWrap: "balance" }}>
                {g.delta}
              </span>
              <span style={mono(10, { color: "var(--mut)", lineHeight: 1.2 })}>ideál {g.ideal}</span>
            </div>
          </div>
        ))}
      </div>

      <p
        style={{
          margin: 0,
          width: "100%",
          maxWidth: 640,
          fontSize: 13,
          lineHeight: 1.55,
          color: "var(--mut)",
          textWrap: "pretty",
        }}
      >
        {advice ?? "Pipeline zatím nevygenerovala doporučení."}
      </p>
    </div>
  );
}
