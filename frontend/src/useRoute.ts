import { useEffect, useState } from "react";

/**
 * Vlastní mini-router nad History API. Jediná další cesta je detail
 * aktivity (`/activity/{id}`) – react-router by pro jednu trasu byl
 * zbytečná závislost, kterou dnes frontend nemá.
 */
export type Route = { name: "dashboard" } | { name: "activity"; id: string };

function parse(pathname: string): Route {
  const m = pathname.match(/^\/activity\/([^/]+)\/?$/);
  return m ? { name: "activity", id: decodeURIComponent(m[1]) } : { name: "dashboard" };
}

export function useRoute(): {
  route: Route;
  openActivity: (id: string) => void;
  back: () => void;
} {
  const [route, setRoute] = useState<Route>(() => parse(window.location.pathname));

  useEffect(() => {
    const onPop = () => setRoute(parse(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const openActivity = (id: string) => {
    window.history.pushState(null, "", `/activity/${encodeURIComponent(id)}`);
    setRoute({ name: "activity", id });
  };

  const back = () => {
    window.history.pushState(null, "", "/");
    setRoute({ name: "dashboard" });
  };

  return { route, openActivity, back };
}
