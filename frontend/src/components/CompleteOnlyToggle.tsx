import type { Theme } from "../theme";
import { mono } from "./ui";

/**
 * Přepínač „jen úplná data", výchozí stav zapnuto.
 *
 * Metrika souvislého bloku je u jízd s nízkým pokrytím systematicky
 * podhodnocená – blok ukončí pauza, ne fyziologie. Míchat takové jízdy do
 * maxima znamená srovnávat výkon s kvalitou záznamu.
 */
export function CompleteOnlyToggle({
  value,
  onChange,
  theme,
}: {
  value: boolean;
  onChange: (value: boolean) => void;
  theme: Theme;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      title="Vyloučí jízdy, u kterých tep chybí na větší části záznamu"
      style={mono(10.5, {
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "5px 12px",
        borderRadius: 999,
        border: `1px solid ${value ? theme.lineOn : theme.line}`,
        background: value ? "var(--track)" : "transparent",
        color: value ? "var(--fg2)" : theme.mut,
        cursor: "pointer",
        whiteSpace: "nowrap",
      })}
    >
      <span style={{ color: value ? theme.ok : theme.grey }}>{value ? "✓" : "○"}</span>
      jen úplná data
    </button>
  );
}
