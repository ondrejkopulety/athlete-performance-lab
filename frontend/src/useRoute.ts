import { useEffect, useState } from "react";

/**
 * Vlastní mini-router nad History API. Aplikace má pět hlavních obrazovek
 * (Přehled `/`, Aktivity `/aktivity`, Trénink `/trenink`, Stats `/stats`,
 * Profil `/profil`), tři drilldowny metrik (`/hrv`, `/rhr`, `/spanek`) a
 * detail jízdy (`/activity/{id}`). react-router by pro tohle byla zbytečná
 * závislost, kterou frontend nemá.
 */
export type MetricKind = "hrv" | "rhr" | "spanek";

export type Route =
  | { name: "dashboard" }
  | { name: "aktivity" }
  | { name: "trenink" }
  | { name: "stats" }
  | { name: "profil" }
  | { name: "metric"; which: MetricKind }
  | { name: "activity"; id: string };

function parse(pathname: string): Route {
  const m = pathname.match(/^\/activity\/([^/]+)\/?$/);
  if (m) return { name: "activity", id: decodeURIComponent(m[1]) };
  if (/^\/aktivity\/?$/.test(pathname)) return { name: "aktivity" };
  if (/^\/trenink\/?$/.test(pathname)) return { name: "trenink" };
  if (/^\/stats\/?$/.test(pathname)) return { name: "stats" };
  if (/^\/profil\/?$/.test(pathname)) return { name: "profil" };
  if (/^\/hrv\/?$/.test(pathname)) return { name: "metric", which: "hrv" };
  if (/^\/rhr\/?$/.test(pathname)) return { name: "metric", which: "rhr" };
  if (/^\/spanek\/?$/.test(pathname)) return { name: "metric", which: "spanek" };
  return { name: "dashboard" };
}

export function useRoute(): {
  route: Route;
  navigate: (path: string) => void;
  openActivity: (id: string) => void;
  back: () => void;
} {
  const [route, setRoute] = useState<Route>(() => parse(window.location.pathname));

  useEffect(() => {
    const onPop = () => setRoute(parse(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const navigate = (path: string) => {
    if (path === window.location.pathname) return;
    window.history.pushState(null, "", path);
    setRoute(parse(path));
  };

  const openActivity = (id: string) => {
    const path = `/activity/${encodeURIComponent(id)}`;
    window.history.pushState(null, "", path);
    setRoute({ name: "activity", id });
  };

  const back = () => {
    // Zpět v historii, když nějaká je; jinak spadni na Přehled.
    if (window.history.length > 1) {
      window.history.back();
      // popstate posluchač dorovná route; pro jistotu ještě naplánuj fallback.
      setTimeout(() => setRoute(parse(window.location.pathname)), 0);
    } else {
      window.history.pushState(null, "", "/");
      setRoute({ name: "dashboard" });
    }
  };

  return { route, navigate, openActivity, back };
}
