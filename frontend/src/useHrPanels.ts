/**
 * Dotahování dat pro panely souvislých bloků na Tréninku.
 *
 * Zbytek Tréninku se odvozuje na klientu z `payload.activities`. Bloky ne:
 * jsou to agregace přes období a přes filtr „jen úplná data" nad tabulkou
 * s desítkami tisíc řádků. Každá zóna (Z4, Z2) má vlastní toleranci
 * přemostění a vlastní filtr, takže se dotahují nezávisle.
 *
 * Během načítání zůstávají viditelná stará data (jen ztlumená) – prázdný
 * panel na půl vteřiny vypadá jako „žádná data", což je jiné tvrzení.
 */

import { useEffect, useState } from "react";

import { fetchBlocks, fetchThreshold, type HrBlocksPayload, type Threshold } from "./api";

export interface BlockRequest {
  zone: string;
  completeOnly: boolean;
  tolerance: number;
}

export interface BlockPanelsState {
  byZone: Record<string, HrBlocksPayload | null>;
  threshold: Threshold | null;
  loading: boolean;
  error: string | null;
}

export function useBlockPanels(
  since: string | null,
  until: string | null,
  requests: BlockRequest[],
  thresholdVersion: number,
): BlockPanelsState {
  const [byZone, setByZone] = useState<Record<string, HrBlocksPayload | null>>({});
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

  // Serializace requestů do stabilního klíče pro dep array.
  const key = JSON.stringify(requests);

  useEffect(() => {
    if (!since || !until) return;
    const reqs: BlockRequest[] = JSON.parse(key);
    const ctrl = new AbortController();
    setLoading(true);

    Promise.all(
      reqs.map((r) =>
        fetchBlocks(
          { since, until, completeOnly: r.completeOnly, tolerance: r.tolerance, zone: r.zone },
          ctrl.signal,
        ).then((b) => [r.zone, b] as const),
      ),
    )
      .then((pairs) => {
        setByZone(Object.fromEntries(pairs));
        setError(null);
      })
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      })
      .finally(() => setLoading(false));

    return () => ctrl.abort();
    // thresholdVersion schválně: po změně LTHR se panel musí zeptat na jiný práh.
  }, [since, until, key, thresholdVersion]);

  return { byZone, threshold, loading, error };
}
