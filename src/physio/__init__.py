"""
physio – R-R intervaly z FIT souborů
=====================================

Jediný zdroj pravdy pro extrakci a posouzení R-R intervalů. Vznikl proto,
že extrakce existovala v repu dvakrát (``fit_parser.extract_rr_from_fit``
jako mrtvý kód a ``loader.extract_rr_intervals_ms`` jako živá cesta do DB)
a ani jedna kopie neuměla říct, jestli jsou naměřená data k něčemu.

Moduly:
  • rr_extract – FIT → R-R intervaly + diagnostika typů zpráv
  • rr_clean   – filtr artefaktů klouzavým mediánem
  • quality    – rozliší skutečné beat-to-beat R-R od dopočítané křivky
  • cli        – dávkové zpracování, cache do ``{activity_id}_rr.csv``

Výpočet DFA-alpha1 a odhad prahů tu **záměrně nejsou**. Diagnostika nad
současnými daty ukázala, že ``hrv`` zprávy v těchto FIT souborech nenesou
variabilitu mezi tepy (viz quality.py), takže by DFA neměla z čeho počítat.
Až budou k dispozici soubory se skutečným R-R, přibude sem modul dfa.py.
"""

from __future__ import annotations

from src.physio.quality import RrAuthenticity, assess_rr_authenticity
from src.physio.rr_clean import CleanedRr, clean_rr
from src.physio.rr_extract import RrExtraction, extract_rr, scan_fit_messages, write_rr_csv

__all__ = [
    "CleanedRr",
    "RrAuthenticity",
    "RrExtraction",
    "assess_rr_authenticity",
    "clean_rr",
    "extract_rr",
    "scan_fit_messages",
    "write_rr_csv",
]
