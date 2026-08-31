import type { Route } from "../useRoute";

/**
 * Spodní navigace – povinná na každé obrazovce (viz Dashboard 2.0/readme.md).
 * 5 položek, fixní, s rozmazaným podkladem. Aktivní stav podle aktuální routy.
 */

type NavKey = "dashboard" | "aktivity" | "trenink" | "stats" | "profil";

const ITEMS: { key: NavKey; label: string; path: string; icon: string }[] = [
  { key: "dashboard", label: "Přehled", path: "/", icon: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z" },
  { key: "aktivity", label: "Aktivity", path: "/aktivity", icon: "M3 12h4l2-7 4 14 2-7h6" },
  { key: "trenink", label: "Trénink", path: "/trenink", icon: "M6 20V13M12 20V6M18 20v-9" },
  { key: "stats", label: "Stats", path: "/stats", icon: "M3 17l5-5 4 4 8-9" },
  { key: "profil", label: "Profil", path: "/profil", icon: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 20c0-4 4-6 8-6s8 2 8 6" },
];

/** Které routy patří pod kterou položku nav baru. */
function activeKey(route: Route): NavKey | null {
  switch (route.name) {
    case "dashboard":
    case "metric":
      return "dashboard";
    case "aktivity":
    case "activity":
      return "aktivity";
    case "trenink":
      return "trenink";
    case "stats":
      return "stats";
    case "profil":
      return "profil";
    default:
      return null;
  }
}

export function BottomNav({
  route,
  onNavigate,
}: {
  route: Route;
  onNavigate: (path: string) => void;
}) {
  const active = activeKey(route);

  return (
    <nav
      style={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: 60,
        display: "flex",
        justifyContent: "center",
        padding: "8px 14px calc(8px + env(safe-area-inset-bottom))",
        background: "color-mix(in oklab, var(--bg) 90%, transparent)",
        backdropFilter: "blur(16px)",
        WebkitBackdropFilter: "blur(16px)",
        borderTop: "1px solid var(--line)",
      }}
    >
      <div style={{ display: "flex", gap: 4, width: "100%", maxWidth: 420 }}>
        {ITEMS.map((it) => {
          const on = it.key === active;
          return (
            <a
              key={it.key}
              href={it.path}
              onClick={(e) => {
                e.preventDefault();
                onNavigate(it.path);
              }}
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 4,
                padding: "8px 4px",
                borderRadius: 14,
                textDecoration: "none",
                background: on ? "var(--track)" : "transparent",
                color: on ? "var(--fg)" : "var(--mut)",
                transition: "background .2s, color .2s",
              }}
            >
              <svg
                viewBox="0 0 24 24"
                width="20"
                height="20"
                fill="none"
                stroke={on ? "var(--fg)" : "var(--mut)"}
                strokeWidth="1.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d={it.icon} />
              </svg>
              <span
                style={{
                  fontFamily: "'JetBrains Mono',monospace",
                  fontSize: 9,
                  letterSpacing: ".08em",
                  textTransform: "uppercase",
                }}
              >
                {it.label}
              </span>
            </a>
          );
        })}
      </div>
    </nav>
  );
}
