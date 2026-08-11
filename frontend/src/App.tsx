import { useCallback, useEffect, useState } from "react";

import { fetchDashboard, type DashboardPayload } from "./api";
import { StateScreen } from "./components/StateScreen";
import { Dashboard } from "./Dashboard";
import { applyTheme, readStoredTheme, storeTheme, type ThemeName } from "./theme";

/** Načtení dat a motiv; všechno ostatní řeší Dashboard. */
export default function App() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [theme, setTheme] = useState<ThemeName>(() => readStoredTheme());

  useEffect(() => {
    applyTheme(theme);
  }, [theme]);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchDashboard(ctrl.signal)
      .then((payload) => {
        setData(payload);
        setTimeout(() => setMounted(true), 30);
      })
      .catch((err: unknown) => {
        if ((err as Error).name === "AbortError") return;
        setError((err as Error).message);
      });
    return () => ctrl.abort();
  }, []);

  const toggleTheme = useCallback(() => {
    setTheme((t) => {
      const next: ThemeName = t === "dark" ? "light" : "dark";
      storeTheme(next);
      return next;
    });
  }, []);

  if (error) {
    return <StateScreen title="Data se nepodařilo načíst" detail={error} />;
  }
  if (!data) {
    return <StateScreen title="Načítám data…" />;
  }

  return (
    <Dashboard payload={data} theme={theme} mounted={mounted} onToggleTheme={toggleTheme} />
  );
}
