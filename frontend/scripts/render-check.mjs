/**
 * Vykreslí dashboard na serveru nad reálnou odpovědí API a zkontroluje,
 * že v HTML nezůstalo NaN/undefined a že projdou všechna období.
 *
 * Náhrada za "kouknu se do prohlížeče" – ručně nikdo neproklikává všechny
 * rozsahy v obou motivech.
 *
 *   node scripts/render-check.mjs [http://localhost:8000/api/dashboard]
 */

import { build } from "esbuild";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const API = process.argv[2] ?? "http://localhost:8000/api/dashboard";

const ENTRY = `
import { createElement } from "react";
import { renderToString } from "react-dom/server";

import { Dashboard } from "../src/Dashboard";
import { toDays } from "../src/api";
import { buildClimb } from "../src/derive/climb";
import { buildGauges } from "../src/derive/gauges";
import { buildHrr } from "../src/derive/hrr";
import { buildPmc } from "../src/derive/pmc";
import { buildPolarization, buildZoneTime, inRange } from "../src/derive/quality";
import { rangeOptions, selectWindow } from "../src/derive/ranges";
import { THEMES } from "../src/theme";

export function render(payload, theme, mounted) {
  return renderToString(
    createElement(Dashboard, {
      payload,
      theme,
      mounted,
      onToggleTheme: () => {},
      onOpenActivity: () => {},
    }),
  );
}

/** Projde všechna období a vrátí, co se v nich spočítalo. */
export function ranges(payload, theme) {
  const T = THEMES[theme];
  const days = toDays(payload.days);
  return rangeOptions(days).map((option) => {
    const win = selectWindow(days, option.value);
    const rows = win.rows;
    const acts = rows.length ? inRange(payload.activities, rows[0].d, rows[rows.length - 1].d) : [];
    const pmc = buildPmc(rows, win.startIdx, win.startIdx + rows.length - 1, days, T);
    return {
      label: option.label,
      days: rows.length,
      activities: acts.length,
      values: [
        pmc.ctlLine, pmc.atlLine, pmc.selX, pmc.sel.ctl, pmc.sel.tsb, pmc.sel.acwr,
        JSON.stringify(pmc.bars.slice(0, 3)),
        JSON.stringify(buildPolarization(acts, T, true)),
        JSON.stringify(buildZoneTime(acts, T, true)),
        JSON.stringify(buildClimb(acts, payload.activities, T, true)),
        JSON.stringify(buildHrr(acts, T, null, false)),
        JSON.stringify(buildGauges(T, payload.today, payload.last_known, true)),
      ].join(" "),
    };
  });
}
`;

const res = await fetch(API);
if (!res.ok) {
  console.error(`API ${API} odpovědělo ${res.status}. Běží uvicorn?`);
  process.exit(1);
}
const payload = await res.json();

// Bundle musí ležet uvnitř projektu, jinak si node nenajde react.
const scriptsDir = fileURLToPath(new URL(".", import.meta.url));
const dir = await mkdtemp(join(scriptsDir, "..", "node_modules", ".render-check-"));
const outfile = join(dir, "entry.mjs");

await build({
  stdin: {
    contents: ENTRY,
    resolveDir: scriptsDir,
    loader: "tsx",
  },
  bundle: true,
  format: "esm",
  platform: "node",
  outfile,
  external: ["react", "react-dom", "react-dom/server"],
  // Mimo Vite `import.meta.env` neexistuje; API klient se tady stejně nevolá.
  define: { "import.meta.env.VITE_API_BASE": '"/api"' },
  logLevel: "error",
});

const mod = await import(pathToFileURL(outfile).href);
await rm(dir, { recursive: true, force: true });

const BAD = ["NaN", "undefined", "Infinity"];
const problems = [];

for (const theme of ["dark", "light"]) {
  for (const mounted of [false, true]) {
    const html = mod.render(payload, theme, mounted);
    const where = `${theme}/mounted=${mounted}`;
    BAD.forEach((n) => html.includes(n) && problems.push(`${where}: HTML obsahuje ${n}`));
    ["Dobré ráno", "Bilance zátěže", "Kvalita tréninku", "Poslední jízdy"].forEach(
      (s) => html.includes(s) || problems.push(`${where}: chybí sekce „${s}"`),
    );
  }

  for (const r of mod.ranges(payload, theme)) {
    BAD.forEach(
      (n) => r.values.includes(n) && problems.push(`${theme}/${r.label}: v datech je ${n}`),
    );
    console.log(`  ${theme.padEnd(5)} ${r.label.padEnd(6)} ${String(r.days).padStart(5)} dní  ${String(r.activities).padStart(4)} jízd`);
  }
}

console.log(
  `\nDní: ${payload.days.length}, jízd: ${payload.rides.length}, aktivit: ${payload.activities.length}`,
);

if (problems.length) {
  console.error("\nProblémy:");
  problems.forEach((p) => console.error(` - ${p}`));
  process.exit(1);
}
console.log("Vše vykresleno bez NaN/undefined, všechna období projdou.");
