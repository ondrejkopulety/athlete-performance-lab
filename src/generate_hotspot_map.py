#!/usr/bin/env python3
"""
generate_hotspot_map.py – Standalone Botanical Hotspot Visualizer
=================================================================
Načte výsledky z detect_botanical_hotspots.py a detect_botanical_stops.py,
propojí je a vygeneruje interaktivní Folium mapu s detailními popup okny.

Vstup:
  • data/processed/botanical_hotspots_ranked.csv
  • data/processed/green_stops_report.csv

Výstup:
  • reports/hotspot_map.html

Spuštění:
  python src/generate_hotspot_map.py
"""

from __future__ import annotations

import math
import os
import re
import sys
from typing import Optional

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    import folium
except ImportError:
    print(
        "\n╔══════════════════════════════════════════════════════════╗\n"
        "║  Chybí knihovna folium – nainstaluj prosím:              ║\n"
        "║                                                          ║\n"
        "║  pip install folium                                      ║\n"
        "║                                                          ║\n"
        "╚══════════════════════════════════════════════════════════╝\n"
    )
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Chybí knihovna pandas. Nainstaluj: pip install pandas")
    sys.exit(1)

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HOTSPOTS_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "botanical_hotspots_ranked.csv")
STOPS_CSV = os.path.join(_PROJECT_ROOT, "data", "processed", "green_stops_report.csv")
OUTPUT_HTML = os.path.join(_PROJECT_ROOT, "reports", "hotspot_map.html")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two WGS84 points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _extract_coords_from_gmaps(link: str) -> Optional[tuple[float, float]]:
    """Extract (lat, lon) from a Google Maps URL like ...query=49.123,16.456."""
    if not isinstance(link, str) or link == "N/A":
        return None
    m = re.search(r"query=([-\d.]+),([-\d.]+)", link)
    if m:
        try:
            return float(m.group(1)), float(m.group(2))
        except ValueError:
            return None
    return None


def _confidence_color(avg_conf: float) -> str:
    """Return marker color based on average confidence %."""
    if avg_conf > 50.0:
        return "#e74c3c"   # red
    elif avg_conf >= 20.0:
        return "#f39c12"   # orange
    else:
        return "#27ae60"   # green


def _confidence_label(avg_conf: float) -> str:
    if avg_conf > 50.0:
        return "Vysoká"
    elif avg_conf >= 20.0:
        return "Střední"
    else:
        return "Nízká"


def _marker_radius(visit_count: int) -> float:
    """Circle radius scaled by visit count (min 6, max 30)."""
    return max(6, min(30, 4 + visit_count * 1.5))


# ─────────────────────────────────────────────────────────────────────────────
# MATCH STOPS TO HOTSPOTS
# ─────────────────────────────────────────────────────────────────────────────

MATCH_RADIUS_M = 400  # max distance to associate a stop with a hotspot centroid


def match_stops_to_hotspots(
    hotspots_df: pd.DataFrame, stops_df: pd.DataFrame
) -> dict[int, pd.DataFrame]:
    """
    Pro každý hotspot (identifikovaný indexem v hotspots_df) najde odpovídající
    zastávky z stops_df na základě GPS proximity (≤ MATCH_RADIUS_M).

    Returns: {hotspot_idx: DataFrame of matching stops}
    """
    result: dict[int, pd.DataFrame] = {}

    # Pre-parse stop coordinates
    stop_coords: list[Optional[tuple[float, float]]] = []
    for _, row in stops_df.iterrows():
        stop_coords.append(_extract_coords_from_gmaps(row.get("Google Maps Link", "")))

    for h_idx, h_row in hotspots_df.iterrows():
        h_lat = h_row["Centroid Lat"]
        h_lon = h_row["Centroid Lon"]
        if pd.isna(h_lat) or pd.isna(h_lon):
            continue

        matching_indices: list[int] = []
        for s_idx, coords in enumerate(stop_coords):
            if coords is None:
                continue
            dist = _haversine_m(h_lat, h_lon, coords[0], coords[1])
            if dist <= MATCH_RADIUS_M:
                matching_indices.append(s_idx)

        if matching_indices:
            result[h_idx] = stops_df.iloc[matching_indices]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# BUILD POPUP HTML
# ─────────────────────────────────────────────────────────────────────────────

def _build_popup_html(h_row: pd.Series, matched_stops: Optional[pd.DataFrame]) -> str:
    """Build rich HTML popup content for a hotspot marker."""
    location = h_row.get("Location", "Neznámá lokalita")
    visit_count = int(h_row.get("Visit Count", 0))
    avg_conf = float(h_row.get("Avg Confidence (%)", 0))
    max_conf = float(h_row.get("Max Confidence (%)", 0))
    avg_dur = float(h_row.get("Avg Duration (min)", 0))
    gmaps_url = h_row.get("Google Maps", "")
    category = h_row.get("Spot Category", "")

    # ── Header ───────────────────────────────────────────────────────────────
    conf_color = _confidence_color(avg_conf)
    html_parts = [
        '<div style="font-family:Arial,sans-serif;max-width:360px;font-size:13px;line-height:1.5;">',
        f'<h3 style="margin:0 0 6px;color:{conf_color};">{category}</h3>',
        f'<p style="margin:0 0 8px;color:#555;font-size:12px;">{location}</p>',
        '<hr style="border:0;border-top:1px solid #ddd;margin:6px 0;">',
    ]

    # ── Statistics ───────────────────────────────────────────────────────────
    html_parts.append('<table style="width:100%;border-collapse:collapse;font-size:12px;">')
    stats = [
        ("Počet návštěv", f"{visit_count}"),
        ("Průměrná délka pauzy", f"{avg_dur:.0f} min"),
        ("Prům. Confidence", f"{avg_conf:.1f} %"),
        ("Max Confidence", f"{max_conf:.1f} %"),
    ]
    for label, val in stats:
        html_parts.append(
            f'<tr><td style="padding:2px 8px 2px 0;color:#666;">{label}:</td>'
            f'<td style="padding:2px 0;font-weight:600;">{val}</td></tr>'
        )
    html_parts.append("</table>")

    # ── Activity list ────────────────────────────────────────────────────────
    if matched_stops is not None and not matched_stops.empty:
        html_parts.append('<hr style="border:0;border-top:1px solid #ddd;margin:6px 0;">')
        html_parts.append(
            '<p style="margin:4px 0;font-weight:600;font-size:12px;">'
            f"Zastávky ({len(matched_stops)}):</p>"
        )
        html_parts.append(
            '<div style="max-height:180px;overflow-y:auto;font-size:11px;">'
        )

        for _, s_row in matched_stops.iterrows():
            s_date = s_row.get("Date", "?")
            s_act_id = s_row.get("Activity ID", "?")
            s_time = s_row.get("Stop Time", "")
            s_dur = s_row.get("Duration (min)", "?")
            s_conf = s_row.get("Confidence Score (%)", "?")
            html_parts.append(
                f'<div style="padding:2px 0;border-bottom:1px dotted #eee;">'
                f"<b>{s_date}</b> {s_time} &nbsp;|&nbsp; "
                f"ID: {s_act_id} &nbsp;|&nbsp; "
                f"{s_dur} min &nbsp;|&nbsp; "
                f"Conf: {s_conf}%"
                f"</div>"
            )

        html_parts.append("</div>")

    # ── Historie návštěv – kompletní seznam datumů z Full Date List ──────
    full_dates_str = h_row.get("Full Date List", "")
    if isinstance(full_dates_str, str) and full_dates_str.strip():
        all_dates = [d.strip() for d in full_dates_str.split(",") if d.strip()]
        if all_dates:
            html_parts.append('<hr style="border:0;border-top:1px solid #ddd;margin:6px 0;">')
            html_parts.append(
                '<p style="margin:4px 0;font-weight:600;font-size:12px;">'
                f"Historie návštěv ({len(all_dates)}):</p>"
            )
            html_parts.append(
                '<div style="max-height:120px;overflow-y:auto;font-size:11px;">'
            )
            dates_items = "".join(f"<div style='padding:1px 0;'>{d}</div>" for d in all_dates)
            html_parts.append(dates_items)
            html_parts.append("</div>")
    elif matched_stops is None or matched_stops.empty:
        # Fallback – use Dates column from hotspots CSV
        dates_str = h_row.get("Dates", "")
        if isinstance(dates_str, str) and dates_str.strip():
            html_parts.append('<hr style="border:0;border-top:1px solid #ddd;margin:6px 0;">')
            html_parts.append(
                '<p style="margin:4px 0;font-size:11px;color:#666;">'
                f"Data: {dates_str}</p>"
            )

    # ── Google Maps link ─────────────────────────────────────────────────────
    if isinstance(gmaps_url, str) and gmaps_url.startswith("http"):
        html_parts.append(
            f'<p style="margin:8px 0 2px;"><a href="{gmaps_url}" target="_blank" '
            f'style="color:#1a73e8;text-decoration:none;">📍 Otevřít v Google Maps</a></p>'
        )

    html_parts.append("</div>")
    return "\n".join(html_parts)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATE MAP
# ─────────────────────────────────────────────────────────────────────────────

def generate_map(hotspots_df: pd.DataFrame, stops_df: pd.DataFrame) -> folium.Map:
    """Create a Folium map with hotspot markers."""

    # Center map on mean coordinates
    center_lat = hotspots_df["Centroid Lat"].mean()
    center_lon = hotspots_df["Centroid Lon"].mean()
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    # Match stops to hotspots
    matched = match_stops_to_hotspots(hotspots_df, stops_df)

    # Add markers
    for h_idx, h_row in hotspots_df.iterrows():
        lat = h_row.get("Centroid Lat")
        lon = h_row.get("Centroid Lon")
        if pd.isna(lat) or pd.isna(lon):
            continue

        avg_conf = float(h_row.get("Avg Confidence (%)", 0))
        visit_count = int(h_row.get("Visit Count", 0))

        color = _confidence_color(avg_conf)
        radius = _marker_radius(visit_count)

        popup_html = _build_popup_html(h_row, matched.get(h_idx))
        popup = folium.Popup(popup_html, max_width=400)

        tooltip_text = (
            f"{h_row.get('Spot Category', '')} | "
            f"{visit_count}× návštěva | "
            f"Conf: {avg_conf:.0f}%"
        )

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=2,
            popup=popup,
            tooltip=tooltip_text,
        ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: white; padding: 12px 16px; border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-family: Arial, sans-serif;
        font-size: 13px; line-height: 1.8;">
        <b>🌿 Botanical Hotspots</b><br>
        <span style="color:#e74c3c;">●</span> Vysoká pravděpodobnost (&gt;50%)<br>
        <span style="color:#f39c12;">●</span> Střední (20–50%)<br>
        <span style="color:#27ae60;">●</span> Nízká (&lt;20%)<br>
        <span style="font-size:11px;color:#888;">Velikost = počet návštěv</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Validate inputs ──────────────────────────────────────────────────────
    for label, path in [("Hotspoty", HOTSPOTS_CSV), ("Zastávky", STOPS_CSV)]:
        if not os.path.isfile(path):
            print(f"❌ Soubor nenalezen: {path}")
            print(f"   Nejprve spusť příslušný detekční skript.")
            sys.exit(1)

    # ── Load data ────────────────────────────────────────────────────────────
    hotspots_df = pd.read_csv(HOTSPOTS_CSV)
    stops_df = pd.read_csv(STOPS_CSV)

    print(f"📂 Načteno {len(hotspots_df)} hotspotů a {len(stops_df)} zastávek.")

    if hotspots_df.empty:
        print("⚠  Hotspoty CSV je prázdné – není co vizualizovat.")
        sys.exit(0)

    # ── Generate map ─────────────────────────────────────────────────────────
    m = generate_map(hotspots_df, stops_df)

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    m.save(OUTPUT_HTML)
    print(f"✅ Mapa uložena: {OUTPUT_HTML}")
    print(f"   Otevři v prohlížeči: file://{OUTPUT_HTML}")


if __name__ == "__main__":
    main()
