# Readiness Dashboard

Webový dashboard nad `/api/dashboard`. React + Vite + TypeScript, žádný
další framework.

Vznikl přepisem návrhu z Claude Design (`../Fitness Dashboard design/Readiness
Dashboard.dc.html`). Ten soubor zůstává v repu jako referenční originál —
barvy, rozměry, prahy a texty se braly odtud. Data v něm jsou zapečená
napevno; tady se stahují z API.

## Vývoj

```bash
npm install
npm run dev          # http://localhost:5173, /api se proxuje na :8000
```

Backend musí běžet zvlášť:

```bash
cd .. && .venv/bin/uvicorn src.api.app:app --reload
```

Jiné API než `localhost:8000`: `VITE_API_TARGET=http://jiny-host:8000 npm run dev`.

## Kontrola

```bash
npm run build            # typová kontrola + produkční build
node scripts/render-check.mjs
```

`render-check.mjs` vykreslí dashboard na serveru nad **reálnou odpovědí API**
ve všech obdobích a obou motivech a zkontroluje, že v HTML nezůstalo
`NaN`/`undefined`. Chytá to, co typy nechytí: dělení nulou v prázdném období,
chybějící biometrii, jediný den v okně.

## Struktura

```
src/
  api.ts            typy a fetch /api/dashboard
  theme.ts          barvy pro JS (CSS proměnné jsou ve styles.css)
  format.ts         české formátování času, dat a desetinné čárky
  derive/           výpočty z designu, oddělené od JSX
    ranges.ts       přepínač období (7D … Vše)
    pmc.ts          graf kondice/únavy a sloupce zátěže
    gauges.ts       čtyři ukazatele – HRV, klidový tep, spánek, strain
    quality.ts      polarizace a čas v zónách
    climb.ts        VAM
    hrr.ts          tepová regenerace
    rides.ts        karty jízd
  components/       JSX se styly opsanými z návrhu
  Dashboard.tsx     pohled nad načtenými daty (veškerý zobrazovací stav)
  App.tsx           načtení dat a přepínač motivu
```

Rozdělení na `App` (data) a `Dashboard` (pohled) není kosmetika — díky němu
jde pohled vykreslit v Node bez prohlížeče, což dělá `render-check.mjs`.

## Odchylky od návrhu

Návrh byl statický snímek, pár věcí v něm bylo napevno:

| V návrhu | Tady |
|---|---|
| Dnešek `2026-08-07` | poslední den z API |
| HRV 53, tep 51, spánek 6:30 | `today`, při chybějícím měření poslední naměřená hodnota i s datem |
| Ideální pásma 62–78 ms a 42–48 tepů | z vlastního baseline (týdenní průměr HRV, 14denní bazál tepu) |
| Strain jako `včerejší TRIMP / 22` | `whoop_strain` z databáze |
| Doporučení jako pevný text | `coach_advice` |
| Rok 2026 v přepínači období | odvozeno z dat |
| Bez ošetření prázdného období | guardy a stav „málo dat" |
