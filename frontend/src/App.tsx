import { useCallback, useEffect, useState } from "react";

import { ActivityDetail } from "./ActivityDetail";
import { Aktivity } from "./Aktivity";
import { fetchDaily, fetchDashboard, type DailyRow, type DashboardPayload } from "./api";
import { BottomNav } from "./components/BottomNav";
import { StateScreen } from "./components/StateScreen";
import { Dashboard } from "./Dashboard";
import { MetricDetail } from "./MetricDetail";
import { Profil } from "./Profil";
import { Stats } from "./Stats";
import { applyTheme, readStoredTheme, storeTheme, type ThemeName } from "./theme";
import { Trenink } from "./Trenink";
import { useRoute } from "./useRoute";

/**
 * Načtení dat (`/api/dashboard` + `/api/daily`), motiv a routing. Přehled a
 * Trénink zůstávají mountované (`display:none`), aby držely zvolené období i
 * po návratu z jiné obrazovky – stejný důvod jako u detailu jízdy.
 */
export default function App() {
  const [data, setData] = useState<DashboardPayload | null>(null);
  const [daily, setDaily] = useState<DailyRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);
  const [theme, setTheme] = useState<ThemeName>(() => readStoredTheme());
  const { route, navigate, openActivity, back } = useRoute();

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
    fetchDaily({ from: "2021-01-01" }, ctrl.signal)
      .then(setDaily)
      .catch(() => {
        /* stepper na Přehledu si poradí i bez plné řady */
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
    <>
      <Dashboard
        payload={data}
        dailySeries={daily}
        theme={theme}
        mounted={mounted}
        hidden={route.name !== "dashboard"}
        onOpenMetric={navigate}
      />

      <Trenink
        payload={data}
        theme={theme}
        mounted={mounted}
        hidden={route.name !== "trenink"}
        thresholdVersion={0}
        onOpenActivity={openActivity}
      />

      <Aktivity theme={theme} hidden={route.name !== "aktivity"} onOpenActivity={openActivity} />

      {route.name === "stats" && <Stats payload={data} theme={theme} />}
      {route.name === "profil" && <Profil theme={theme} onToggleTheme={toggleTheme} />}
      {route.name === "metric" && <MetricDetail which={route.which} theme={theme} onBack={back} />}
      {route.name === "activity" && <ActivityDetail id={route.id} theme={theme} onBack={back} />}

      <BottomNav route={route} onNavigate={navigate} />
    </>
  );
}
