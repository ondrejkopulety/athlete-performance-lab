"""
cli.py – dávkové výpočty nad vteřinovými daty
==============================================

Dva podpříkazy, dva různé zdroje dat:

``rr`` – extrakce a posouzení R-R intervalů **z FIT souborů**::

    python -m src.physio.cli rr data/fit                 # celá složka
    python -m src.physio.cli rr data/fit/activity_X.fit  # jeden soubor
    python -m src.physio.cli rr data/fit --force         # ignoruj cache
    python -m src.physio.cli rr data/fit --messages      # výpis typů zpráv
    python -m src.physio.cli rr data/fit --write-db      # zapiš posudek do DB

Aktivita, pro kterou už existuje ``{activity_id}_rr.csv``, se znovu
neparsuje – FIT soubory jsou velké a jejich čtení je nejdražší část.

``hr`` – tepová křivka a souvislé bloky nad prahem **z tabulky records**::

    python -m src.physio.cli hr                     # vše, co ještě nemá výsledky
    python -m src.physio.cli hr --force             # přepočítej všechno
    python -m src.physio.cli hr --since 2026-01-01  # jen novější aktivity
    python -m src.physio.cli hr --activity 22807256593
    python -m src.physio.cli hr --dry-run           # spočítej, nezapisuj

R-R musí z FIT, protože ``hrv`` zprávy v databázi nejsou. Tep tam ale je,
a to už po sloučení fragmentů a kanonizaci sportu – takže ``hr`` čte
``records`` a vidí tentýž stream jako zbytek pipeline. FIT navíc pokrývá
jen třetinu aktivit v databázi.

Podpříkaz ``rr`` se doplní i tehdy, když se vynechá: ``python -m
src.physio.cli data/fit`` funguje dál jako dřív.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.settings import RR_DIR
from src.physio.quality import assess_rr_authenticity
from src.physio.rr_clean import clean_rr
from src.physio.rr_extract import extract_rr, read_rr_csv, write_rr_csv

log = logging.getLogger("physio.cli")


@dataclass
class ActivityRrReport:
    """Výsledek posouzení jedné aktivity."""

    activity_id: str
    fit_path: str
    has_hrv_messages: bool
    beat_count: int
    artifact_pct: float | None
    reliable: bool | None
    verdict: str
    reason: str
    zero_diff_pct: float | None = None
    unique_values: int | None = None
    lattice_coverage: float | None = None
    rmssd_ms: float | None = None
    rr_csv: str | None = None
    from_cache: bool = False
    error: str | None = None

    @property
    def dfa_quality(self) -> str:
        """
        Hodnota pro ``activity_metrics.dfa_quality``.

        ``no_rr``        – FIT neobsahuje ``hrv`` zprávy
        ``synthetic_rr`` – ``hrv`` zprávy jsou, ale nenesou variabilitu mezi tepy
        ``unreliable``   – skutečné R-R, ale přes 10 % artefaktů
        ``rr_ok``        – skutečné R-R, připravené pro DFA
        ``failed``       – soubor se nepodařilo přečíst
        """
        if self.error:
            return "failed"
        if not self.has_hrv_messages:
            return "no_rr"
        if self.verdict == "synthetic":
            return "synthetic_rr"
        if self.verdict == "unknown":
            return "no_rr"
        return "rr_ok" if self.reliable else "unreliable"


def _find_fit_files(target: Path) -> list[Path]:
    """Cesta k souboru → jednoprvkový seznam; cesta ke složce → všechny .fit."""
    if target.is_file():
        return [target]
    if not target.is_dir():
        raise FileNotFoundError(f"Cesta neexistuje: {target}")
    files = sorted(p for p in target.iterdir() if p.suffix.lower() == ".fit" and p.is_file())
    return files


def process_one(fit_path: str, out_dir: str, force: bool = False) -> ActivityRrReport:
    """
    Zpracuje jeden FIT soubor: extrakce → čištění → posudek.

    Args:
        fit_path: Cesta k .fit souboru.
        out_dir: Adresář pro ``{activity_id}_rr.csv``.
        force: Přeparsovat i když cache existuje.

    Returns:
        ActivityRrReport – nikdy nevyhazuje výjimku, chybu nese v ``error``.
    """
    path = Path(fit_path)

    # Cache řeší run_batch ještě před spuštěním workeru – sem se dostane
    # jen soubor, který se opravdu musí přečíst.
    extraction = extract_rr(path)

    if extraction.error:
        return ActivityRrReport(
            activity_id=extraction.activity_id,
            fit_path=str(path),
            has_hrv_messages=False,
            beat_count=0,
            artifact_pct=None,
            reliable=None,
            verdict="unknown",
            reason=f"FIT nelze přečíst: {extraction.error}",
            error=extraction.error,
        )

    if not extraction.has_hrv_messages:
        return ActivityRrReport(
            activity_id=extraction.activity_id,
            fit_path=str(path),
            has_hrv_messages=False,
            beat_count=0,
            artifact_pct=None,
            reliable=None,
            verdict="unknown",
            reason="soubor neobsahuje 'hrv' zprávy – R-R intervaly v něm nejsou",
        )

    cleaned = clean_rr(extraction.rr_seconds)
    authenticity = assess_rr_authenticity(cleaned.rr_seconds)
    csv_path = write_rr_csv(extraction, out_dir)

    return ActivityRrReport(
        activity_id=extraction.activity_id,
        fit_path=str(path),
        has_hrv_messages=True,
        beat_count=extraction.beat_count,
        artifact_pct=cleaned.artifact_pct,
        reliable=cleaned.reliable,
        verdict=authenticity.verdict,
        reason=authenticity.reason,
        zero_diff_pct=authenticity.zero_diff_pct,
        unique_values=authenticity.unique_values,
        lattice_coverage=authenticity.lattice_coverage,
        rmssd_ms=authenticity.rmssd_ms,
        rr_csv=str(csv_path),
        from_cache=False,
    )


def _cached_report(fit_path: Path, cache: Path) -> ActivityRrReport | None:
    """Sestaví posudek z uloženého ``_rr.csv`` bez otevírání FIT souboru."""
    try:
        rr = read_rr_csv(cache)
    except (OSError, ValueError, KeyError) as exc:
        log.debug("Cache %s nejde přečíst (%s) – parsuji FIT znovu.", cache, exc)
        return None
    if rr.size == 0:
        return None

    cleaned = clean_rr(rr)
    authenticity = assess_rr_authenticity(cleaned.rr_seconds)
    return ActivityRrReport(
        activity_id=cache.name.removesuffix("_rr.csv"),
        fit_path=str(fit_path),
        has_hrv_messages=True,
        beat_count=int(rr.size),
        artifact_pct=cleaned.artifact_pct,
        reliable=cleaned.reliable,
        verdict=authenticity.verdict,
        reason=authenticity.reason,
        zero_diff_pct=authenticity.zero_diff_pct,
        unique_values=authenticity.unique_values,
        lattice_coverage=authenticity.lattice_coverage,
        rmssd_ms=authenticity.rmssd_ms,
        rr_csv=str(cache),
        from_cache=True,
    )


def _progress(done: int, total: int, width: int = 34) -> None:
    """
    Jednoduchý progress bar na stderr – bez externí závislosti.

    Mimo terminál (roura, log) se překreslování přes ``\\r`` neprojeví a
    každý krok by zůstal na výstupu jako samostatný řádek, takže se v tom
    případě hlásí jen desetinové zlomky.
    """
    if not sys.stderr.isatty():
        if total and (done == total or done % max(1, total // 10) == 0):
            print(f"  {done}/{total}", file=sys.stderr, flush=True)
        return

    filled = int(width * done / total) if total else width
    bar = "█" * filled + "·" * (width - filled)
    print(f"\r  [{bar}] {done}/{total}", end="", file=sys.stderr, flush=True)
    if done == total:
        print(file=sys.stderr)


def run_batch(
    target: Path,
    out_dir: Path,
    force: bool = False,
    workers: int = 0,
) -> list[ActivityRrReport]:
    """
    Zpracuje soubor nebo celou složku paralelně.

    Args:
        target: Cesta k .fit souboru nebo ke složce s nimi.
        out_dir: Kam ukládat ``_rr.csv``.
        force: Ignorovat cache.
        workers: Počet procesů; 0 = počet jader.

    Returns:
        Posudky v pořadí, v jakém byly soubory nalezeny.
    """
    files = _find_fit_files(target)
    if not files:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, ActivityRrReport] = {}
    todo: list[Path] = []

    # Cache: hledá se podle ID odvozeného z názvu souboru. Když sedí,
    # FIT se vůbec neotevírá.
    if not force:
        by_id = {p.name.removesuffix("_rr.csv"): p for p in out_dir.glob("*_rr.csv")}
        for f in files:
            from src.physio.rr_extract import extract_activity_id

            cache = by_id.get(extract_activity_id(f))
            report = _cached_report(f, cache) if cache else None
            if report is not None:
                reports[str(f)] = report
            else:
                todo.append(f)
    else:
        todo = list(files)

    if reports:
        print(f"  {len(reports)} aktivit načteno z cache (--force je přeparsuje).",
              file=sys.stderr)

    if todo:
        max_workers = workers or (os.cpu_count() or 4)
        done = 0
        _progress(done, len(todo))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(process_one, str(f), str(out_dir), force): f for f in todo
            }
            for fut in as_completed(futures):
                f = futures[fut]
                try:
                    reports[str(f)] = fut.result()
                except Exception as exc:  # noqa: BLE001 – jeden soubor nesmí shodit dávku
                    log.error("[%s] Zpracování selhalo: %s", f.name, exc)
                    reports[str(f)] = ActivityRrReport(
                        activity_id=f.stem, fit_path=str(f), has_hrv_messages=False,
                        beat_count=0, artifact_pct=None, reliable=None,
                        verdict="unknown", reason=str(exc), error=str(exc),
                    )
                done += 1
                _progress(done, len(todo))

    return [reports[str(f)] for f in files if str(f) in reports]


def print_summary(reports: list[ActivityRrReport]) -> None:
    """Souhrnný report na stdout."""
    total = len(reports)
    with_rr = [r for r in reports if r.beat_count > 0]
    by_quality: dict[str, int] = {}
    for r in reports:
        by_quality[r.dfa_quality] = by_quality.get(r.dfa_quality, 0) + 1

    print()
    print("═" * 68)
    print(f"  Zpracováno FIT souborů: {total}")
    print(f"  Obsahuje R-R intervaly: {len(with_rr)}  ({len(with_rr) / total * 100:.0f} %)"
          if total else "  Obsahuje R-R intervaly: 0")
    print("═" * 68)

    labels = {
        "rr_ok": "skutečné R-R, použitelné pro DFA",
        "unreliable": "skutečné R-R, ale přes 10 % artefaktů",
        "synthetic_rr": "hrv zprávy jsou, ale nenesou variabilitu mezi tepy",
        "no_rr": "žádné hrv zprávy ve FIT",
        "failed": "soubor nelze přečíst",
    }
    for key in ("rr_ok", "unreliable", "synthetic_rr", "no_rr", "failed"):
        if key in by_quality:
            print(f"  {key:14s} {by_quality[key]:4d}   {labels[key]}")

    if with_rr:
        beats = [r.beat_count for r in with_rr]
        arts = [r.artifact_pct for r in with_rr if r.artifact_pct is not None]
        print()
        print(f"  Tepů na aktivitu:  {min(beats)} – {max(beats)}"
              f"  (medián {sorted(beats)[len(beats) // 2]})")
        if arts:
            print(f"  Artefaktů:         {min(arts) * 100:.2f} – {max(arts) * 100:.2f} %")

    usable = by_quality.get("rr_ok", 0)
    if usable == 0 and with_rr:
        print()
        print("  ⚠ Žádná aktivita nemá R-R použitelné pro DFA-alpha1.")
        example = next(r for r in with_rr if r.verdict == "synthetic")
        print(f"    Důvod (např. {example.activity_id}): {example.reason}")
        print("    DFA-alpha1 měří fluktuace mezi tepy. Nad touto řadou vychází")
        print("    α1 ≈ 1,8 (Brownovský signál) a prahy z ní nevypadnou.")
    print()


def print_hr_summary(result, written: dict[str, int] | None) -> None:
    """Souhrn předvýpočtu tepové křivky a bloků na stdout."""
    from config.settings import HR_COVERAGE_WARN_PCT

    print()
    print("═" * 68)
    print(f"  Zpracováno aktivit:     {len(result.processed)}")
    if result.cached:
        print(f"  Z cache (aktuální):     {len(result.cached)}   (--force je přepočítá)")
    if result.skipped_no_hr:
        print(f"  Bez tepu, přeskočeno:   {len(result.skipped_no_hr)}")
    print("═" * 68)

    if result.processed:
        print(f"  Sekundová mřížka:       {result.grid_seconds / 3600:.1f} h")
        # Podíl děr je diagnostika přeindexování: kdyby byl nulový, znamenalo
        # by to, že se autopauza do dat vůbec nepropsala, a klouzavá okna by
        # pokrývala víc reálného času, než tvrdí.
        print(f"  Z toho díry (pauzy):    {result.gap_pct:.1f} %")
        print(f"  Řádky křivky:           {len(result.curve_rows)}")
        print(f"  Řádky bloků:            {len(result.block_rows)}")

    low = result.low_coverage(HR_COVERAGE_WARN_PCT)
    if low:
        print()
        print(f"  ⚠ {len(low)} aktivit pod {HR_COVERAGE_WARN_PCT:.0f} % pokrytí "
              "– čísla stojí na části dat")
        print(f"    {'aktivita':<14} {'pokrytí':>8} {'vzorky':>8} {'rozsah':>9}"
              f" {'nejdelší díra':>14} {'oken křivky':>12}")
        for cov in low[:15]:
            print(f"    {cov.activity_id:<14} {cov.coverage_pct:7.1f} %"
                  f" {cov.sample_density_pct:7.1f} % {cov.span_s / 60:8.1f}m"
                  f" {cov.longest_gap_s / 60:13.1f}m {cov.curve_windows:12d}")
        if len(low) > 15:
            print(f"    … a dalších {len(low) - 15}")
        print("    Kratší okna křivky můžou vyjít i tady – vzniknou jen tam, kde je")
        print("    okno plně pokryté. Bloky nad prahem díra vždy ukončí.")

    # Nízká hustota vzorků při plném pokrytí není porucha, ale Smart
    # Recording. Vypisuje se odděleně, ať se nemíchá s výpadky dat.
    sparse = [
        c for c in result.coverage
        if c.coverage_pct >= HR_COVERAGE_WARN_PCT and c.sample_density_pct < 50.0
    ]
    if sparse:
        print()
        print(f"  {len(sparse)} aktivit má řídký zápis (Smart Recording, medián vzorku 5 s),")
        print("  ale po doplnění mezer plné pokrytí – okna z nich vycházejí normálně.")

    if written is not None:
        print()
        print(f"  Zapsáno do activity_hr_curve:  {written['curve']} řádků")
        print(f"  Zapsáno do activity_hr_blocks: {written['blocks']} řádků")
    print()


def run_hr(args: argparse.Namespace) -> int:
    """Podpříkaz ``hr`` – předvýpočet tepové křivky a souvislých bloků."""
    from config.settings import HR_BLOCKS_VERSION, HR_CURVE_VERSION
    from src.db import repository as repo
    from src.db.models import Activity, ActivityHrBlocks, ActivityHrCurve
    from src.db.session import session_scope
    from src.physio.hr_batch import run_batch as run_hr_batch
    from src.physio.persist import write_hr_rows

    with session_scope() as session:
        from sqlalchemy import select

        stmt = select(Activity.activity_id).order_by(Activity.date)
        if args.activity:
            stmt = stmt.where(Activity.activity_id.in_(args.activity))
        if args.since:
            stmt = stmt.where(Activity.date >= args.since)
        if args.until:
            stmt = stmt.where(Activity.date <= args.until)
        if args.sport:
            stmt = stmt.where(Activity.sport.ilike(f"%{args.sport}%"))
        candidates = [row[0] for row in session.execute(stmt).all()]

        if not candidates:
            print("Žádné aktivity neodpovídají filtru.", file=sys.stderr)
            return 1

        cached: list[str] = []
        todo = candidates
        if not args.force:
            # Hotovost se pozná podle bloků: ty vzniknou pro celou mřížku
            # prahů vždy, když je z čeho počítat, takže "má aktuální bloky"
            # znamená "výpočet nad touhle aktivitou proběhl".
            #
            # Křivka se posuzuje zvlášť, ale jen u aktivit, které nějaké
            # řádky křivky mají. Aktivita kratší než nejkratší okno je
            # legitimně nemá – a kdyby se čekalo i na ně, počítala by se
            # taková aktivita při každém běhu znovu.
            done = repo.hr_computed_ids(session, ActivityHrBlocks, HR_BLOCKS_VERSION)
            stale_curve = repo.hr_computed_ids(session, ActivityHrCurve, 0) - (
                repo.hr_computed_ids(session, ActivityHrCurve, HR_CURVE_VERSION)
            )
            done -= stale_curve
            cached = [a for a in candidates if a in done]
            todo = [a for a in candidates if a not in done]

        if cached:
            print(f"  {len(cached)} aktivit má aktuální výsledky (--force je přepočítá).",
                  file=sys.stderr)
        if not todo:
            print("  Není co počítat.", file=sys.stderr)
            return 0

        print(f"  Čtu vteřinová data ({len(todo)} aktivit)…", file=sys.stderr)
        series = repo.read_hr_series(session, todo)

        result = run_hr_batch(series, todo, cached=cached, progress=_progress)

        written = None
        if not args.dry_run:
            written = write_hr_rows(
                session,
                result.curve_rows,
                result.block_rows,
                [c.to_row(HR_CURVE_VERSION) for c in result.coverage],
            )

    if args.json:
        print(json.dumps(
            {
                "processed": result.processed,
                "cached": result.cached,
                "skipped_no_hr": result.skipped_no_hr,
                "curve_rows": len(result.curve_rows),
                "block_rows": len(result.block_rows),
                "coverage": [
                    {
                        "activity_id": c.activity_id,
                        "coverage_pct": round(c.coverage_pct, 1),
                        "sample_density_pct": round(c.sample_density_pct, 1),
                        "span_s": c.span_s,
                        "measured_s": c.measured_s,
                        "usable_s": c.usable_s,
                        "longest_gap_s": c.longest_gap_s,
                        "curve_windows": c.curve_windows,
                    }
                    for c in sorted(result.coverage, key=lambda c: c.coverage_pct)
                ],
                "written": written,
            },
            ensure_ascii=False, indent=2,
        ))
    else:
        print_hr_summary(result, written)
    return 0


def run_rr(args: argparse.Namespace) -> int:
    """Podpříkaz ``rr`` – extrakce a posouzení R-R intervalů z FIT souborů."""
    if args.messages:
        from src.physio.rr_extract import scan_fit_messages

        files = _find_fit_files(args.path)
        for f in files[:1] if len(files) > 1 else files:
            print(f"\n{f.name}")
            counts = scan_fit_messages(f)
            for name, n in counts.items():
                mark = "  ← R-R intervaly" if name == "hrv" else ""
                print(f"  {name:28s} {n:7d}{mark}")
            if "hrv" not in counts:
                print("\n  hrv zprávy v souboru NEJSOU – R-R intervaly z něj nelze získat.")
        return 0

    reports = run_batch(args.path, args.out, force=args.force, workers=args.workers)
    if not reports:
        print("Nenalezeny žádné .fit soubory.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([asdict(r) | {"dfa_quality": r.dfa_quality} for r in reports],
                         ensure_ascii=False, indent=2))
    else:
        print_summary(reports)

    if args.write_db:
        from src.physio.persist import write_reports_to_db

        n = write_reports_to_db(reports)
        print(f"  Zapsáno do activity_metrics: {n} řádků.")

    return 0


SUBCOMMANDS = ("rr", "hr")


def _normalize_argv(argv: list[str]) -> list[str]:
    """
    Doplní podpříkaz ``rr``, když chybí.

    ``python -m src.physio.cli data/fit`` fungovalo dřív, než ``hr`` vzniklo,
    a je ve skriptech i v README. Rozbít to kvůli druhému podpříkazu by byla
    zbytečná daň.
    """
    for i, token in enumerate(argv):
        if token in SUBCOMMANDS:
            return argv
        if not token.startswith("-"):
            return ["rr", *argv]
        if token in ("-h", "--help"):
            return argv
    return argv if argv else ["--help"]


def main(argv: list[str] | None = None) -> int:
    """Vstupní bod CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src.physio.cli",
        description="Výpočty nad vteřinovými daty: R-R intervaly z FIT, "
                    "tepová křivka a souvislé bloky z databáze.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="podrobné logování")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rr = subparsers.add_parser(
        "rr", help="extrakce R-R intervalů z FIT souborů a posouzení jejich použitelnosti"
    )
    rr.add_argument("path", type=Path,
                    help="cesta k .fit souboru nebo ke složce s .fit soubory")
    rr.add_argument("--out", type=Path, default=RR_DIR,
                    help=f"kam ukládat {{activity_id}}_rr.csv (výchozí: {RR_DIR})")
    rr.add_argument("--force", action="store_true",
                    help="přeparsovat i aktivity, které už mají _rr.csv")
    rr.add_argument("--workers", type=int, default=0,
                    help="počet paralelních procesů (výchozí: počet jader)")
    rr.add_argument("--messages", action="store_true",
                    help="vypsat typy zpráv v souboru (jen pro jeden soubor)")
    rr.add_argument("--write-db", action="store_true",
                    help="zapsat dfa_quality a diagnostiku do activity_metrics")
    rr.add_argument("--json", action="store_true",
                    help="místo reportu vypsat výsledky jako JSON")
    rr.add_argument("-v", "--verbose", action="store_true", help="podrobné logování")

    hr = subparsers.add_parser(
        "hr", help="tepová křivka a souvislé bloky nad prahem z tabulky records"
    )
    hr.add_argument("--force", action="store_true",
                    help="přepočítat i aktivity, které už mají aktuální výsledky")
    hr.add_argument("--since", type=date.fromisoformat, metavar="RRRR-MM-DD",
                    help="jen aktivity od tohoto data")
    hr.add_argument("--until", type=date.fromisoformat, metavar="RRRR-MM-DD",
                    help="jen aktivity do tohoto data")
    hr.add_argument("--sport", help="jen aktivity, jejichž sport obsahuje tento řetězec")
    hr.add_argument("--activity", nargs="+", metavar="ID",
                    help="jen vyjmenované aktivity")
    hr.add_argument("--dry-run", action="store_true",
                    help="spočítat a vypsat souhrn, ale nezapisovat do databáze")
    hr.add_argument("--json", action="store_true",
                    help="místo reportu vypsat výsledky jako JSON")
    hr.add_argument("-v", "--verbose", action="store_true", help="podrobné logování")

    args = parser.parse_args(_normalize_argv(list(argv) if argv is not None else sys.argv[1:]))

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    return run_hr(args) if args.command == "hr" else run_rr(args)


if __name__ == "__main__":
    raise SystemExit(main())
