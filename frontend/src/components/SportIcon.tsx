import { SPORT_ICON, type SportKey } from "../derive/sportIcons";

export function SportIcon({
  sport,
  size = 18,
  color = "currentColor",
}: {
  sport: SportKey;
  size?: number;
  color?: string;
}) {
  const ic = SPORT_ICON[sport];
  return (
    <svg
      viewBox={ic.vb}
      width={size}
      height={size}
      fill={ic.fill ? color : "none"}
      stroke={ic.fill ? "none" : color}
      strokeWidth={ic.fill ? undefined : 1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={ic.d} />
    </svg>
  );
}
