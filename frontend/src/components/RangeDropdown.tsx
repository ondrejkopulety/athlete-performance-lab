import { useState } from "react";

import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { mono } from "./ui";

/**
 * pillDropdown pro výběr období – přesně dle designu 2.0 (button + chevron,
 * scrim + popup s dlouhými popisky). Nahrazuje `RangeTabs` na všech
 * obrazovkách.
 */
export function RangeDropdown({
  options,
  value,
  onPick,
  theme,
  align = "right",
}: {
  options: RangeOption[];
  value: Range;
  onPick: (value: Range) => void;
  theme: Theme;
  align?: "left" | "right";
}) {
  const [open, setOpen] = useState(false);
  const short = (options.find((o) => o.value === value) ?? options[0])?.label ?? "";

  return (
    <div style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        style={mono(10, {
          display: "flex",
          alignItems: "center",
          gap: 8,
          letterSpacing: ".06em",
          padding: "6px 10px 6px 12px",
          borderRadius: 999,
          border: "1px solid var(--line2)",
          background: "var(--track)",
          color: "var(--fg2)",
          cursor: "pointer",
          whiteSpace: "nowrap",
        })}
      >
        <span style={{ minWidth: 38, textAlign: "left" }}>{short}</span>
        <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="var(--mut)" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </button>

      {open && (
        <>
          <div onClick={() => setOpen(false)} style={{ position: "fixed", inset: 0, zIndex: 20 }} />
          <div
            style={{
              position: "absolute",
              top: "calc(100% + 7px)",
              [align]: 0,
              zIndex: 21,
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: 6,
              minWidth: 138,
              border: "1px solid var(--line2)",
              borderRadius: 14,
              background: "var(--card2)",
              boxShadow: "0 14px 34px rgba(0,0,0,.28)",
            }}
          >
            {options.map((o) => {
              const on = o.value === value;
              return (
                <button
                  key={String(o.value)}
                  type="button"
                  onClick={() => {
                    onPick(o.value);
                    setOpen(false);
                  }}
                  style={mono(11, {
                    letterSpacing: ".04em",
                    textAlign: "left",
                    padding: "8px 10px",
                    borderRadius: 9,
                    border: "none",
                    cursor: "pointer",
                    whiteSpace: "nowrap",
                    transition: "background .15s,color .15s",
                    background: on ? theme.fg : "transparent",
                    color: on ? theme.card : theme.fg2,
                  })}
                >
                  {o.full}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
