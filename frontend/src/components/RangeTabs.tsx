import type { Range, RangeOption } from "../derive/ranges";
import type { Theme } from "../theme";
import { mono } from "./ui";

export function RangeTabs({
  options,
  value,
  onPick,
  theme,
}: {
  options: RangeOption[];
  value: Range;
  onPick: (value: Range) => void;
  theme: Theme;
}) {
  return (
    <div
      style={{
        display: "flex",
        gap: 2,
        padding: 2,
        borderRadius: 999,
        background: "var(--track)",
        flexWrap: "wrap",
      }}
    >
      {options.map((o) => (
        <button
          key={String(o.value)}
          type="button"
          onClick={() => onPick(o.value)}
          style={mono(10.5, {
            letterSpacing: ".06em",
            padding: "6px 13px",
            borderRadius: 999,
            border: "none",
            cursor: "pointer",
            transition: "background .2s,color .2s",
            whiteSpace: "nowrap",
            background: value === o.value ? theme.fg : "transparent",
            color: value === o.value ? theme.card : theme.mut,
          })}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
