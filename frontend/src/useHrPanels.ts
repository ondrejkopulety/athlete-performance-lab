/**
 * Dotahování dat pro panely tepové křivky a souvislých bloků.
 *
 * Zbytek dashboardu se načte jedním požadavkem a období se přepíná v
 * prohlížeči. Tyhle dva panely ne: jsou to agregace přes období a přes
 * filtr „jen úplná data" nad tabulkami, které mají desítky tisíc řádků.
 * Přepnutí období, tolerance nebo filtru proto znamená nový dotaz.
 *
 * Během načítání zůstávají viditelná stará data (jen ztlumená) – prázdný
 * panel na půl vteřiny vypadá jako "žádná data", což je jiné tvrzení.
 */

import { useEffect, useState } from "react";

import {
  fetchBlocks,
  fetchCurve,
  fetchThreshold,
  type HrBlocksPayload,
  type HrCurvePayload,
  type Threshold,
} from "./api";

export interface HrPanelsState {
  curve: HrCurvePayload | null;
  blocks: HrBlocksPayload | null;
  threshold: Threshold | null;
  loading: boolean;
  error: string | null;
}

export function useHrPanels(
  since: string | null,
  until: string | null,
  completeOnly: boolean,
  compare: "prev" | "year",
  tolerance: number,
  thresholdVersion: number,
): HrPanelsState & { setThreshold: (next: Threshold) => void } {
  const [curve, setCurve] = useState<HrCurvePayload | null>(null);
  const [blocks, setBlocks] = useState<HrBlocksPayload | null>(null);
  const [threshold, setThreshold] = useState<Threshold | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchThreshold(ctrl.signal)
      .then(setThreshold)
      .catch(() => {
        /* práh je doplněk, jeho výpadek nesmí shodit panely */
      });
    return () => ctrl.abort();
  }, []);

  useEffect(() => {
    if (!since || !until) return;
    const ctrl = new AbortController();
    setLoading(true);

    Promise.all([
      fetchCurve({ since, until, completeOnly, compare }, ctrl.signal),
      fetchBlocks({ since, until, completeOnly, tolerance }, ctrl.signal),
    ])
      .then(([c, b]) => {
        setCurve(c);
        setBlocks(b);
        setError(null);
      })
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      })
      .finally(() => setLoading(false));

    return () => ctrl.abort();
    // thresholdVersion je tu schválně: po změně LTHR se panel bloků musí
    // zeptat na jiný práh. Přepočítává se dotaz, ne data.
  }, [since, until, completeOnly, compare, tolerance, thresholdVersion]);

  return { curve, blocks, threshold, loading, error, setThreshold };
}
