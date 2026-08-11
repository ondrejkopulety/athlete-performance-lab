import { useMemo, type PointerEvent as ReactPointerEvent } from "react";

/**
 * Tahání prstem/myší po grafu. Myš stačí jen přejet (jako v designu),
 * dotyk musí držet – proto pointer capture.
 */
export function useScrub(
  active: boolean,
  setActive: (value: boolean) => void,
  onFraction: (fraction: number) => void,
) {
  return useMemo(() => {
    const at = (e: ReactPointerEvent<HTMLDivElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      if (r.width === 0) return;
      onFraction(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)));
    };

    return {
      onPointerDown: (e: ReactPointerEvent<HTMLDivElement>) => {
        try {
          e.currentTarget.setPointerCapture(e.pointerId);
        } catch {
          /* Safari občas odmítne – tahání pak jede bez capture */
        }
        setActive(true);
        at(e);
      },
      onPointerMove: (e: ReactPointerEvent<HTMLDivElement>) => {
        if (active || e.pointerType === "mouse") at(e);
      },
      onPointerUp: (e: ReactPointerEvent<HTMLDivElement>) => {
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          /* viz výše */
        }
        setActive(false);
      },
      onPointerCancel: () => setActive(false),
      onPointerLeave: () => setActive(false),
    };
  }, [active, setActive, onFraction]);
}
