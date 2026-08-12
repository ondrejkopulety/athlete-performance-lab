import { useEffect, useState } from "react";

import { saveThreshold, type Threshold } from "../api";
import type { Theme } from "../theme";
import { mono, NOTE } from "./ui";

/**
 * „LTHR 172 · nastaveno před N dny" – a po devadesáti dnech výrazně.
 *
 * Na tomhle jednom čísle visí prahy panelu bloků a tiše stárne; zóny v
 * settings.py jsou měřené z laktátového testu a mají přednost, takže změna
 * prahu **nepřepočítává žádná data** – mění se jen to, na který uložený
 * práh se panel bloků ptá.
 */
export function ThresholdBadge({
  threshold,
  onSaved,
  theme,
}: {
  threshold: Threshold | null;
  onSaved: (next: Threshold) => void;
  theme: Theme;
}) {
  const [open, setOpen] = useState(false);
  const [lthr, setLthr] = useState("");
  const [hrMax, setHrMax] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (threshold) {
      setLthr(String(threshold.lthr_bpm));
      setHrMax(String(threshold.hr_max_bpm));
    }
  }, [threshold]);

  if (!threshold) return null;

  const age =
    threshold.source === "settings"
      ? "z config/settings.py"
      : threshold.days_ago === 0
        ? "nastaveno dnes"
        : `nastaveno před ${threshold.days_ago} dny`;

  const submit = async () => {
    setSaving(true);
    setError(null);
    try {
      onSaved(
        await saveThreshold({ lthr_bpm: Number(lthr), hr_max_bpm: Number(hrMax) }),
      );
      setOpen(false);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="hover-fg"
        style={mono(10.5, {
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 12px",
          border: `1px solid ${threshold.stale ? theme.warn : "var(--line2)"}`,
          borderRadius: 999,
          background: "var(--card)",
          color: "var(--mut)",
          letterSpacing: ".06em",
          cursor: "pointer",
          whiteSpace: "nowrap",
        })}
      >
        <span style={{ color: "var(--fg2)" }}>LTHR {threshold.lthr_bpm}</span>
        <span style={{ color: threshold.stale ? theme.warn : "var(--mut)" }}>· {age}</span>
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            right: 0,
            top: "calc(100% + 8px)",
            zIndex: 20,
            width: 268,
            padding: 16,
            border: "1px solid var(--line2)",
            borderRadius: 16,
            background: "var(--card)",
            boxShadow: "0 18px 40px rgba(0,0,0,.28)",
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          <Field label="Prahový tep (LTHR)" value={lthr} onChange={setLthr} />
          <Field label="Maximální tep" value={hrMax} onChange={setHrMax} />

          <p style={NOTE}>
            Změna přepočítá jen zobrazení. Bloky jsou uložené na mřížce absolutních
            prahů ({Object.entries(threshold.zone_thresholds)
              .map(([z, t]) => `${z} ${t}`)
              .join(" · ")}
            ), takže se mění dotaz, ne data.
          </p>

          {error && <p style={{ ...NOTE, color: theme.bad }}>{error}</p>}

          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={submit}
              disabled={saving}
              style={mono(10.5, {
                flex: 1,
                padding: "7px 12px",
                borderRadius: 999,
                border: "none",
                background: theme.fg,
                color: theme.card,
                cursor: saving ? "wait" : "pointer",
              })}
            >
              {saving ? "Ukládám…" : "Uložit"}
            </button>
            <button
              type="button"
              onClick={() => setOpen(false)}
              style={mono(10.5, {
                padding: "7px 12px",
                borderRadius: 999,
                border: "1px solid var(--line2)",
                background: "transparent",
                color: "var(--mut)",
                cursor: "pointer",
              })}
            >
              Zpět
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
      <span style={mono(9.5, { letterSpacing: ".1em", textTransform: "uppercase", color: "var(--mut)" })}>
        {label}
      </span>
      <input
        type="number"
        inputMode="numeric"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={mono(14, {
          padding: "7px 10px",
          borderRadius: 10,
          border: "1px solid var(--line2)",
          background: "var(--track)",
          color: "var(--fg)",
        })}
      />
    </label>
  );
}
