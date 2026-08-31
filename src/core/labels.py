"""
labels.py  –  krátké lidské popisky pro UI / API panely
=======================================================

Formátovací primitivum (žádné DB ani analytické závislosti), aby ho mohla
volat jak repository vrstva při skládání panelových řádků, tak analytika,
aniž by kvůli tomu vznikala závislost mezi vrstvami.
"""

from __future__ import annotations


def activity_label(sport: str | None, km: float | None, minutes: float | None) -> str:
    """
    Popisek jízdy do panelu – "odkud ten bod je" (např. ``cycling · 42 km · 1:35``).

    Nepoužívá ``activities.activity_name``: ten je v celé databázi NULL
    (Garmin ho v exportu neposílá), takže by z popisku zbylo prázdno.
    """
    parts = [(sport or "aktivita").split("/")[0]]
    if km:
        parts.append(f"{km:.0f} km")
    if minutes:
        hours, mins = divmod(int(minutes), 60)
        parts.append(f"{hours}:{mins:02d}" if hours else f"{mins} min")
    return " · ".join(parts)
