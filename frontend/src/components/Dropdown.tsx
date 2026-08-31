import type { CSSProperties, ReactNode } from "react";

import { mono } from "./ui";

export interface DropdownItem {
  key: string;
  label: string;
  active: boolean;
  onPick: () => void;
}

/**
 * Malé rozbalovací menu (výběr období, srovnání s rokem). Neviditelný overlay
 * přes celou obrazovku zavírá menu kliknutím mimo něj – stejný trik jako
 * v designu.
 */
export function Dropdown({
  open,
  onToggle,
  onClose,
  buttonLabel,
  buttonExtra,
  items,
  align = "right",
  minWidth = 138,
  buttonStyle,
}: {
  open: boolean;
  onToggle: () => void;
  onClose: () => void;
  buttonLabel: ReactNode;
  buttonExtra?: ReactNode;
  items: DropdownItem[];
  align?: "left" | "right";
  minWidth?: number;
  buttonStyle?: CSSProperties;
}) {
  return (
    <div style={{ position: "relative" }}>
      <button
        type="button"
        onClick={onToggle}
        style={mono(10, {
          display: "flex",
          alignItems: "center",
          gap: 8,
          padding: "6px 10px 6px 12px",
          borderRadius: 999,
          border: "1px solid var(--line2)",
          background: "var(--track)",
          color: "var(--fg2)",
          cursor: "pointer",
          whiteSpace: "nowrap",
          ...buttonStyle,
        })}
      >
        {buttonLabel}
        {buttonExtra}
      </button>
      {open && (
        <>
          <div
            onClick={onClose}
            style={{ position: "fixed", inset: 0, zIndex: 20 }}
          />
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
              minWidth,
              border: "1px solid var(--line2)",
              borderRadius: 14,
              background: "var(--card2)",
              boxShadow: "0 14px 34px rgba(0,0,0,.28)",
            }}
          >
            {items.map((it) => (
              <button
                key={it.key}
                type="button"
                onClick={it.onPick}
                style={mono(11, {
                  letterSpacing: ".04em",
                  textAlign: "left",
                  padding: "8px 10px",
                  borderRadius: 9,
                  border: "none",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                  transition: "background .15s,color .15s",
                  background: it.active ? "var(--fg)" : "transparent",
                  color: it.active ? "var(--card)" : "var(--fg2)",
                })}
              >
                {it.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
