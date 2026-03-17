"""
Map Overlay Module
==================
Creates interactive Folium maps with real lat/lon coordinates for
detected threats, trajectories, and anomaly hotspots.
Designed for DRDO panel presentation.
"""

import folium
from folium import plugins
import numpy as np
from typing import Dict, List, Tuple, Optional

# Risk level colors
RISK_COLORS = {
    "LOW": "#33cc33",
    "MEDIUM": "#ffcc00",
    "HIGH": "#ff8800",
    "CRITICAL": "#ff3333",
}

RISK_ICONS = {
    "LOW": "info-sign",
    "MEDIUM": "warning-sign",
    "HIGH": "exclamation-sign",
    "CRITICAL": "remove-sign",
}

CLASS_ICONS = {
    "vehicle": "car",
    "aircraft": "plane",
    "ship": "ship",
    "armored_vehicle": "cog",
    "personnel": "user",
    "helicopter": "plane",
    "radar_installation": "signal",
    "missile_launcher": "fire",
}


def create_threat_map(
    risk_results: Dict,
    trajectories: Dict,
    hotspots: List[Dict],
    geo_info: Dict,
    image_size: Tuple[int, int] = (1024, 1024),
) -> folium.Map:
    """
    Create an interactive Folium map showing:
      - Threat markers at real lat/lon positions
      - Trajectory polylines color-coded by risk
      - Hotspot circles for anomaly zones
      - Mini-map and fullscreen control
    """
    from utils.real_data import pixel_to_latlon, REGION

    center_lat = geo_info.get("center_lat", REGION["center_lat"])
    center_lon = geo_info.get("center_lon", REGION["center_lon"])

    # Base map with satellite-style tiles
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
    )

    # ── Tile layers ──
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # ── Feature groups ──
    fg_threats = folium.FeatureGroup(name="Threat Markers")
    fg_trajectories = folium.FeatureGroup(name="Trajectories")
    fg_hotspots = folium.FeatureGroup(name="Anomaly Hotspots")
    fg_aoi = folium.FeatureGroup(name="Area of Interest")

    # ── Area of Interest rectangle ──
    bbox = geo_info.get("bbox", {
        "min_lat": REGION["min_lat"], "max_lat": REGION["max_lat"],
        "min_lon": REGION["min_lon"], "max_lon": REGION["max_lon"],
    })
    folium.Rectangle(
        bounds=[[bbox["min_lat"], bbox["min_lon"]], [bbox["max_lat"], bbox["max_lon"]]],
        color="#00d4ff",
        weight=2,
        fill=True,
        fill_color="#00d4ff",
        fill_opacity=0.05,
        tooltip=f"AOI: {geo_info.get('region', 'Surveillance Zone')}",
    ).add_to(fg_aoi)

    # ── Trajectories ──
    for tid, tdata in trajectories.items():
        traj = tdata.get("trajectory", [])
        if len(traj) < 2:
            continue

        risk_level = risk_results.get(tid, {}).get("risk_level", "LOW")
        color = RISK_COLORS[risk_level]

        # Convert pixel coords to lat/lon
        latlons = []
        for pt in traj:
            lat, lon = pixel_to_latlon(pt["cx"], pt["cy"])
            latlons.append([lat, lon])

        if len(latlons) >= 2:
            folium.PolyLine(
                locations=latlons,
                color=color,
                weight=2.5,
                opacity=0.8,
                tooltip=f"Track {tid} | {tdata.get('class_name','?')} | {risk_level}",
            ).add_to(fg_trajectories)

            # Start point (circle)
            folium.CircleMarker(
                location=latlons[0],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.5,
                tooltip=f"T{tid} start",
            ).add_to(fg_trajectories)

    # ── Threat markers ──
    for tid, data in risk_results.items():
        risk_level = data.get("risk_level", "LOW")
        class_name = data.get("class_name", "unknown")
        risk_score = data.get("risk_score", 0.0)
        traj_length = data.get("trajectory_length", 0)

        # Get last known position
        traj = trajectories.get(tid, {}).get("trajectory", [])
        if not traj:
            continue

        lat, lon = pixel_to_latlon(traj[-1]["cx"], traj[-1]["cy"])

        # Popup HTML
        bd = data.get("anomaly_breakdown", {})
        popup_html = f"""
        <div style="font-family:monospace; min-width:220px;">
          <h4 style="margin:0;color:{RISK_COLORS[risk_level]};">
            ⚠ Track {tid} — {class_name.upper()}
          </h4>
          <hr style="border-color:#444;margin:4px 0;">
          <b>Risk Score:</b> {risk_score:.4f}<br>
          <b>Risk Level:</b> <span style="color:{RISK_COLORS[risk_level]};font-weight:bold;">{risk_level}</span><br>
          <b>Position:</b> {lat:.4f}°N, {lon:.4f}°E<br>
          <b>Trajectory:</b> {traj_length} frames<br>
          <hr style="border-color:#444;margin:4px 0;">
          <b>Anomaly Breakdown:</b><br>
          &nbsp; Speed: {bd.get('speed', 0):.3f}<br>
          &nbsp; Direction: {bd.get('direction', 0):.3f}<br>
          &nbsp; Loitering: {bd.get('loitering', 0):.3f}<br>
          &nbsp; Convergence: {bd.get('convergence', 0):.3f}<br>
          &nbsp; Transformer: {bd.get('transformer', 0):.3f}<br>
        </div>
        """

        icon_name = CLASS_ICONS.get(class_name, "info-sign")
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"T{tid} | {class_name} | {risk_level} ({risk_score:.3f})",
            icon=folium.Icon(
                color=_folium_color(risk_level),
                icon=icon_name,
                prefix="glyphicon",
            ),
        ).add_to(fg_threats)

    # ── Hotspot circles ──
    for i, hs in enumerate(hotspots):
        lat, lon = pixel_to_latlon(hs["center_x"], hs["center_y"])
        intensity = hs.get("intensity", 0.5)
        radius_m = int(3000 * intensity)  # scaled circle size

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color="#ff3333",
            weight=1.5,
            fill=True,
            fill_color="#ff3333",
            fill_opacity=0.15 * intensity,
            tooltip=f"Hotspot #{i+1} | Intensity: {intensity:.3f}",
        ).add_to(fg_hotspots)

        # Label for top 3
        if i < 3:
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;color:#ff3333;font-weight:bold;background:rgba(0,0,0,0.6);padding:2px 5px;border-radius:3px;">H{i+1}</div>',
                    icon_size=(30, 20),
                ),
                tooltip=f"Hotspot #{i+1}",
            ).add_to(fg_hotspots)

    # Add all feature groups
    fg_aoi.add_to(m)
    fg_hotspots.add_to(m)
    fg_trajectories.add_to(m)
    fg_threats.add_to(m)

    # ── Plugins ──
    plugins.MiniMap(toggle_display=True).add_to(m)
    plugins.Fullscreen(position="topright").add_to(m)
    plugins.MousePosition(
        position="bottomleft",
        separator=" | Lon: ",
        prefix="Lat: ",
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # ── Legend ──
    legend_html = """
    <div style="position:fixed; bottom:30px; right:10px; z-index:1000;
                background:rgba(10,15,26,0.9); color:#a0b8d0;
                padding:12px 16px; border-radius:8px;
                border:1px solid #1e3a5f; font-family:monospace; font-size:12px;">
      <b style="color:#00d4ff;">RISK LEVELS</b><br>
      <span style="color:#33cc33;">●</span> LOW &nbsp;&nbsp;
      <span style="color:#ffcc00;">●</span> MEDIUM<br>
      <span style="color:#ff8800;">●</span> HIGH &nbsp;
      <span style="color:#ff3333;">●</span> CRITICAL<br>
      <hr style="border-color:#1e3a5f;margin:6px 0;">
      <span style="color:#ff3333;">○</span> Anomaly Hotspot<br>
      <span style="color:#00d4ff;">—</span> Trajectory
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def _folium_color(risk_level: str) -> str:
    """Map risk level to Folium marker color name."""
    mapping = {
        "LOW": "green",
        "MEDIUM": "orange",
        "HIGH": "red",
        "CRITICAL": "darkred",
    }
    return mapping.get(risk_level, "blue")
