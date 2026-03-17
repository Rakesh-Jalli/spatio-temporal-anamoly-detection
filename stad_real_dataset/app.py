"""
STAD - Spatio-Temporal Anomaly Detection
=========================================
Focused on temporal anomaly detection using real satellite imagery.
Pipeline: Data -> Detection -> Tracking -> Temporal Transformer -> Anomaly Scoring
"""

import streamlit as st
import numpy as np
import cv2
import time
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime

from utils.config import get_config
from utils.preprocessing import enhance_optical
from utils.visualization import (
    draw_detections, draw_trajectories, draw_anomaly_heatmap,
    create_dashboard, RISK_COLORS,
)
from utils.real_data import (
    pixel_to_latlon, set_active_region,
    load_highres_scenes,
)
from utils.scene_manager import load_cached_scenes
from utils.map_overlay import create_threat_map
from utils.change_detection import compute_change_maps, compute_temporal_stats
from models.detector import YOLOv8OBBDetector
from models.tracker import ByteTracker
from models.transformer import TemporalAnalyzer
from models.anomaly import RiskScorer, SpatialAnomalyMapper

try:
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


# Page Config
st.set_page_config(
    page_title="STAD - Temporal Anomaly Detection",
    page_icon="\U0001f6f0\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');
    .main .block-container { padding-top: 0.8rem; max-width: 100%; }

    .cmd-header {
        background: linear-gradient(135deg, #040810 0%, #0c1624 40%, #0a1018 100%);
        border: 1px solid #1a3050;
        border-left: 4px solid #00d4ff;
        border-radius: 6px;
        padding: 18px 28px;
        margin-bottom: 16px;
    }
    .cmd-header h1 {
        font-family: 'Rajdhani', sans-serif;
        color: #00d4ff;
        font-size: 1.9em;
        margin: 0;
        letter-spacing: 4px;
        text-transform: uppercase;
    }
    .cmd-header .sub {
        font-family: 'JetBrains Mono', monospace;
        color: #4a7090;
        font-size: 0.78em;
        margin: 4px 0 0 0;
    }

    .kpi-card {
        background: linear-gradient(145deg, #0c1420, #10202e);
        border: 1px solid #1a3050;
        border-radius: 6px;
        padding: 14px 16px;
        text-align: center;
    }
    .kpi-card .val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8em; font-weight: 700; color: #00d4ff;
    }
    .kpi-card .lbl {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.78em; color: #4a7090;
        text-transform: uppercase; letter-spacing: 1px;
    }

    .risk-critical { color: #ff3333; font-weight: bold; }
    .risk-high     { color: #ff8800; font-weight: bold; }
    .risk-medium   { color: #ffcc00; font-weight: bold; }
    .risk-low      { color: #33cc33; }

    .status-live {
        display: inline-block; width: 8px; height: 8px;
        background: #00ff88; border-radius: 50%;
        animation: blink 1.5s infinite; margin-right: 6px;
    }
    @keyframes blink {
        0%, 100% { box-shadow: 0 0 0 0 rgba(0,255,136,0.4); }
        50% { box-shadow: 0 0 0 6px rgba(0,255,136,0); }
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #060c14 0%, #0c1824 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00d4ff; font-family: 'Rajdhani', sans-serif;
    }
    div[data-testid="stExpander"] { border: 1px solid #1a3050; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Regions with high-res data
# ============================================================

REGIONS = {
    "San Diego Intl Airport — 190 planes detected (0.6m NAIP)": {
        "min_lon": -117.205, "max_lon": -117.180,
        "min_lat": 32.728, "max_lat": 32.740,
        "center_lat": 32.734, "center_lon": -117.1925,
        "name": "San Diego Intl Airport — 190 planes detected (0.6m NAIP)",
    },
    "Port of Long Beach — 64 ships, 96 vehicles (0.6m NAIP)": {
        "min_lon": -118.230, "max_lon": -118.205,
        "min_lat": 33.742, "max_lat": 33.758,
        "center_lat": 33.750, "center_lon": -118.2175,
        "name": "Port of Long Beach — 64 ships, 96 vehicles (0.6m NAIP)",
    },
    "San Diego Naval Base — USS carriers & destroyers (0.6m NAIP)": {
        "min_lon": -117.128, "max_lon": -117.112,
        "min_lat": 32.683, "max_lat": 32.696,
        "center_lat": 32.6895, "center_lon": -117.120,
        "name": "San Diego Naval Base — USS carriers & destroyers (0.6m NAIP)",
    },
}


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="cmd-header">
    <h1>STAD &mdash; Spatio-Temporal Anomaly Detection</h1>
    <p class="sub">
        <span class="status-live"></span>
        Temporal Anomaly Detection &nbsp;&bull;&nbsp;
        NAIP 0.6m Aerial Imagery &nbsp;&bull;&nbsp;
        Detection &rarr; Tracking &rarr; Transformer &rarr; Anomaly Scoring
    </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")

    region_choice = st.selectbox("Region", list(REGIONS.keys()), index=0)
    selected_region = {"name": region_choice, **REGIONS[region_choice]}

    st.info(
        f"**{selected_region['name']}**\n\n"
        f"Lat: {selected_region['min_lat']:.2f} - {selected_region['max_lat']:.2f} N\n\n"
        f"Lon: {selected_region['min_lon']:.2f} - {selected_region['max_lon']:.2f} E"
    )

    st.markdown("---")
    num_frames = st.slider("Temporal Frames", 5, 20, 16, 1)
    num_objects = st.slider("Target Count", 4, 20, 12)

    with st.expander("Detection Settings"):
        conf_threshold = st.slider("Confidence", 0.05, 0.9, 0.15, 0.05)
        iou_threshold = st.slider("IoU", 0.1, 0.9, 0.45, 0.05)

    with st.expander("Anomaly Thresholds"):
        risk_low = st.slider("Low Cutoff", 0.1, 0.5, 0.3, 0.05)
        risk_med = st.slider("Medium Cutoff", 0.3, 0.8, 0.6, 0.05)
        risk_high = st.slider("High Cutoff", 0.5, 0.95, 0.85, 0.05)

    st.markdown("---")
    run_pipeline = st.button("RUN TEMPORAL ANOMALY DETECTION", use_container_width=True, type="primary")


# ============================================================
# Helpers
# ============================================================

def cv2_to_pil(img):
    if img is None:
        return Image.new("RGB", (256, 256), (20, 20, 30))
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def risk_hex(level):
    return {"LOW": "#33cc33", "MEDIUM": "#ffcc00", "HIGH": "#ff8800", "CRITICAL": "#ff3333"}.get(level, "#888")


def _plotly_dark(fig, height=350):
    fig.update_layout(
        height=height,
        plot_bgcolor="#060c14", paper_bgcolor="#060c14",
        font=dict(color="#8aa0b8", family="JetBrains Mono, monospace", size=11),
        margin=dict(l=40, r=20, t=30, b=40),
    )
    return fig


# ============================================================
# MAIN — landing / pipeline
# ============================================================

if not run_pipeline:
    # ── Landing page ──
    st.markdown("### How Temporal Anomaly Detection Works")
    st.markdown("""
This system analyses **multi-temporal aerial imagery** to detect anomalous
behaviour patterns over time. Each frame is a real NAIP 0.6m aerial
acquisition from a different year.

**Pipeline Stages:**

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Input | Real NAIP 0.6m imagery | Multi-year aerial acquisitions |
| 2. Detection | YOLOv8-OBB | Detect objects with oriented bounding boxes |
| 3. Tracking | ByteTrack | Link detections across temporal frames |
| 4. Temporal Analysis | Transformer Encoder | Model motion sequences with self-attention |
| 5. Anomaly Scoring | 5-indicator fusion | Speed + Direction + Loitering + Convergence + Transformer |
| 6. Spatial Mapping | Gaussian heatmap | Identify anomaly hotspot regions |
""")

    st.markdown("---")
    st.markdown("### Input Data Preview")
    st.caption("NAIP 0.6m aerial imagery — planes, ships and vehicles clearly visible")

    preview_scenes = load_highres_scenes(region=selected_region, max_scenes=8)
    if preview_scenes:
        st.success(f"Loaded {len(preview_scenes)} temporal scenes for **{selected_region['name']}** "
                   f"({preview_scenes[0]['date']} to {preview_scenes[-1]['date']})")

        st.markdown("**NAIP 0.6m Aerial — Multi-Year Acquisitions:**")
        n_show = min(len(preview_scenes), 8)
        cols = st.columns(n_show)
        for i in range(n_show):
            with cols[i]:
                st.caption(preview_scenes[i]["date"])
                opt = preview_scenes[i]["optical"]
                if len(opt.shape) == 3:
                    opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)
                st.image(opt, use_container_width=True)
    else:
        st.warning("No high-res data for this region. Select a region with downloaded data.")

    st.info("Click **RUN TEMPORAL ANOMALY DETECTION** in the sidebar to start the pipeline.")

else:
    # ============================================================
    # EXECUTE PIPELINE
    # ============================================================
    config = get_config()
    config.num_demo_frames = num_frames
    config.image_size = (2048, 2048)
    config.detector.confidence_threshold = conf_threshold
    config.detector.iou_threshold = iou_threshold
    config.detector.input_size = (640, 640)
    config.anomaly.risk_threshold_low = risk_low
    config.anomaly.risk_threshold_medium = risk_med
    config.anomaly.risk_threshold_high = risk_high

    progress = st.progress(0, text="Initializing...")

    # ── Models ──
    detector = YOLOv8OBBDetector(config.detector)
    tracker = ByteTracker(config.tracker)
    temporal_analyzer = TemporalAnalyzer(config.transformer, device="cpu")
    risk_scorer = RiskScorer(config.anomaly)
    anomaly_mapper = SpatialAnomalyMapper(
        image_size=config.image_size,
        grid_size=config.anomaly.spatial_grid_size,
    )
    progress.progress(10, text="YOLOv8-OBB + ByteTrack loaded")

    # ── Data ──
    set_active_region(selected_region)

    progress.progress(15, text="Loading NAIP 0.6m imagery...")

    temporal_scenes = load_highres_scenes(
        region=selected_region,
        image_size=config.image_size,
        max_scenes=num_frames,
    )
    if not temporal_scenes:
        temporal_scenes = load_cached_scenes(
            region=selected_region,
            image_size=config.image_size,
            max_scenes=num_frames,
        )
    if not temporal_scenes:
        st.error("No satellite data available for this region.")
        st.stop()

    geo_info = temporal_scenes[-1].get("geo_info", {
        "region": selected_region.get("name"),
        "center_lat": selected_region.get("center_lat"),
        "center_lon": selected_region.get("center_lon"),
        "bbox": {
            "min_lat": selected_region.get("min_lat"),
            "max_lat": selected_region.get("max_lat"),
            "min_lon": selected_region.get("min_lon"),
            "max_lon": selected_region.get("max_lon"),
        },
    })

    # Use optical images directly (NAIP is optical-only)
    optical_frames = [scene["optical"] for scene in temporal_scenes]
    actual_frames = len(optical_frames)

    change_maps_list = compute_change_maps(temporal_scenes)
    temporal_stats = compute_temporal_stats(temporal_scenes, change_maps_list)

    st.success(f"Loaded **{actual_frames}** NAIP 0.6m scenes "
               f"({temporal_scenes[0]['date']} to {temporal_scenes[-1]['date']}, "
               f"{temporal_stats.get('total_days', 0)} days)")
    progress.progress(25, text=f"{actual_frames} frames ready")

    # ── Real YOLOv8-OBB Detection + Tracking per frame ──
    detection_counts, track_counts, frame_times = [], [], []
    all_frame_detections = []  # Store detections per frame for confusion matrix

    st.markdown("### Stage 1-3: Real YOLOv8-OBB Detection & Tracking")

    col_input, col_det = st.columns(2)
    with col_input:
        st.markdown("**Input: NAIP 0.6m Aerial**")
        live_frame = st.empty()
    with col_det:
        st.markdown("**Output: YOLOv8-OBB Detections (Tiled)**")
        live_det = st.empty()

    for idx in range(actual_frames):
        t0 = time.time()
        frame = optical_frames[idx]

        # Run real YOLOv8-OBB with sliding-window tiling on full-res image
        detections = detector.detect(frame)
        det_dicts = [d.to_dict() for d in detections]
        detection_counts.append(len(detections))
        all_frame_detections.append(detections)

        active_tracks = tracker.update(det_dicts)
        track_counts.append(len(active_tracks))
        frame_times.append(time.time() - t0)

        # Visualize on the optical frame
        frame_rgb = frame.copy()
        if len(frame_rgb.shape) == 2:
            frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_GRAY2BGR)
        det_vis = draw_detections(frame_rgb.copy(), detections, show_labels=True)

        if idx % 2 == 0 or idx == actual_frames - 1:
            live_frame.image(cv2_to_pil(frame_rgb), use_container_width=True)
            live_det.image(cv2_to_pil(det_vis), use_container_width=True)

        if idx == actual_frames - 1:
            last_fused = frame_rgb
            last_detections = detections
            last_det_vis = det_vis

        pct = 25 + int(55 * (idx + 1) / actual_frames)
        progress.progress(pct, text=f"Frame {idx+1}/{actual_frames} — {len(detections)} det, {len(active_tracks)} tracks")

    # ── Temporal analysis ──
    progress.progress(82, text="Stage 4: Transformer temporal analysis...")
    trajectories = tracker.get_all_trajectories()
    transformer_scores = temporal_analyzer.analyze(trajectories)

    progress.progress(88, text="Stage 5: Anomaly risk scoring...")
    current_track_dicts = [t.to_dict() for t in tracker.active_tracks]
    risk_results = risk_scorer.compute_risk_scores(
        trajectories, transformer_scores, current_track_dicts,
    )

    progress.progress(94, text="Stage 6: Spatial anomaly heatmap...")
    heatmap = anomaly_mapper.generate_heatmap(risk_results, trajectories)
    hotspots = anomaly_mapper.get_hotspots(heatmap, threshold=0.5)

    traj_vis = draw_trajectories(last_fused.copy(), trajectories, risk_results)
    heatmap_vis = draw_anomaly_heatmap(heatmap, config.image_size, last_fused, alpha=0.6)

    progress.progress(100, text="Pipeline complete")

    # ============================================================
    # RESULTS
    # ============================================================
    st.markdown("---")

    # ── KPIs ──
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for _, data in risk_results.items():
        risk_counts[data["risk_level"]] += 1

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f'<div class="kpi-card"><div class="val">{actual_frames}</div>'
                f'<div class="lbl">Temporal Frames</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="val">{len(trajectories)}</div>'
                f'<div class="lbl">Tracked Objects</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="val">{np.mean(frame_times)*1000:.0f}ms</div>'
                f'<div class="lbl">Avg Frame Time</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="val" style="color:#33cc33">{risk_counts["LOW"]}</div>'
                f'<div class="lbl">Low Risk</div></div>', unsafe_allow_html=True)
    k5.markdown(f'<div class="kpi-card"><div class="val" style="color:#ffcc00">{risk_counts["MEDIUM"]}</div>'
                f'<div class="lbl">Medium Risk</div></div>', unsafe_allow_html=True)
    k6.markdown(f'<div class="kpi-card"><div class="val" style="color:#ff3333">{risk_counts["HIGH"] + risk_counts["CRITICAL"]}</div>'
                f'<div class="lbl">High / Critical</div></div>', unsafe_allow_html=True)

    # ── Input Temporal Stack ──
    st.markdown("---")
    st.markdown("### Input: Multi-Temporal NAIP 0.6m Aerial Imagery")
    st.caption(f"Region: **{selected_region['name']}** — "
               f"{temporal_scenes[0]['date']} to {temporal_scenes[-1]['date']} — "
               f"{temporal_stats.get('total_days', 0)} days coverage")

    n_show = min(actual_frames, 8)
    st.markdown("**NAIP 0.6m Aerial Imagery:**")
    opt_cols = st.columns(n_show)
    for i in range(n_show):
        with opt_cols[i]:
            st.caption(temporal_scenes[i]["date"])
            opt = enhance_optical(temporal_scenes[i]["optical"])
            if len(opt.shape) == 3:
                opt = cv2.cvtColor(opt, cv2.COLOR_BGR2RGB)
            st.image(opt, use_container_width=True)

    # ── Change Detection ──
    if change_maps_list:
        st.markdown("---")
        st.markdown("### Temporal Change Detection")
        st.caption("Pixel-level change between consecutive acquisitions — highlights new activity")
        n_chg = min(len(change_maps_list), 8)
        chg_cols = st.columns(n_chg)
        for k in range(n_chg):
            chg = change_maps_list[k]
            with chg_cols[k]:
                pct_val = chg["change_pct"]
                color = "#ff4444" if pct_val > 10 else ("#ffaa00" if pct_val > 5 else "#33cc33")
                st.caption(f"{chg['date_from'][-5:]} -> {chg['date_to'][-5:]}")
                st.markdown(f"<span style='color:{color};font-weight:bold'>Change {pct_val:.1f}%</span>",
                            unsafe_allow_html=True)
                st.image(cv2.cvtColor(chg["change_rgb"], cv2.COLOR_BGR2RGB),
                         use_container_width=True)

    # ── Detection & Tracking Output ──
    st.markdown("---")
    st.markdown("### Detection & Tracking Results")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Last Frame — NAIP 0.6m Input**")
        st.image(cv2_to_pil(last_fused), use_container_width=True)
    with c2:
        st.markdown("**Last Frame — YOLOv8-OBB Detections (Tiled)**")
        st.image(cv2_to_pil(last_det_vis), use_container_width=True)

    det_data = [{
        "Class": d.class_name,
        "Confidence": f"{d.confidence:.3f}",
        "Center": f"({d.cx:.0f}, {d.cy:.0f})",
        "Size": f"{d.width:.0f}x{d.height:.0f}",
        "Angle": f"{d.angle:.1f} deg",
    } for d in last_detections]
    if det_data:
        st.dataframe(pd.DataFrame(det_data), use_container_width=True, hide_index=True)

    # ── Zoom Insets: Cropped views of detected objects ──
    if last_detections:
        st.markdown("#### Object Close-Up Views")
        st.caption("Zoomed-in crops around each detected object for clear visibility")
        n_det = min(len(last_detections), 8)
        zoom_cols = st.columns(min(n_det, 4))
        for di in range(n_det):
            det = last_detections[di]
            col = zoom_cols[di % min(n_det, 4)]
            # Crop a 256×256 region around detection center
            h_img, w_img = last_det_vis.shape[:2]
            crop_r = 128
            x1 = max(0, int(det.cx) - crop_r)
            y1 = max(0, int(det.cy) - crop_r)
            x2 = min(w_img, int(det.cx) + crop_r)
            y2 = min(h_img, int(det.cy) + crop_r)
            crop = last_det_vis[y1:y2, x1:x2]
            if crop.size > 0:
                # Upscale crop for visibility
                crop_big = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
                with col:
                    st.image(cv2_to_pil(crop_big), use_container_width=True,
                             caption=f"{det.class_name} ({det.confidence:.2f})")

    # ── Temporal Anomaly Results ──
    st.markdown("---")
    st.markdown("### Temporal Anomaly Detection Results")
    st.caption("Anomaly scores from 5 indicators: Speed, Direction Change, "
               "Loitering, Convergence, and Transformer attention scores")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Trajectory Trails (Risk-Colored)**")
        st.image(cv2_to_pil(traj_vis), use_container_width=True)
    with c2:
        st.markdown("**Spatial Anomaly Heatmap**")
        st.image(cv2_to_pil(heatmap_vis), use_container_width=True)

    # Threat table
    st.markdown("#### Anomaly Scores per Tracked Object")
    threat_rows = []
    for tid, data in sorted(risk_results.items(),
                             key=lambda x: x[1]["risk_score"], reverse=True):
        bd = data["anomaly_breakdown"]
        threat_rows.append({
            "Track": tid,
            "Class": data["class_name"],
            "Risk Score": f"{data['risk_score']:.4f}",
            "Level": data["risk_level"],
            "Speed": f"{bd['speed']:.3f}",
            "Direction": f"{bd['direction']:.3f}",
            "Loitering": f"{bd['loitering']:.3f}",
            "Convergence": f"{bd['convergence']:.3f}",
            "Transformer": f"{bd['transformer']:.3f}",
        })
    if threat_rows:
        st.dataframe(pd.DataFrame(threat_rows), use_container_width=True, hide_index=True)

    # ── Analytics Charts ──
    st.markdown("---")
    st.markdown("### Temporal Analytics")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Risk Distribution**")
        fig_risk = px.bar(
            x=list(risk_counts.keys()), y=list(risk_counts.values()),
            color=list(risk_counts.keys()),
            color_discrete_map={"LOW": "#33cc33", "MEDIUM": "#ffcc00",
                                "HIGH": "#ff8800", "CRITICAL": "#ff3333"},
        )
        fig_risk = _plotly_dark(fig_risk, 300)
        fig_risk.update_layout(showlegend=False, xaxis_title="Risk Level", yaxis_title="Count")
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        st.markdown("**Detections & Tracks Over Time**")
        fig_det = go.Figure()
        fig_det.add_trace(go.Scatter(
            x=list(range(1, actual_frames+1)), y=detection_counts,
            mode="lines+markers", name="Detections",
            line=dict(color="#00d4ff", width=2),
        ))
        fig_det.add_trace(go.Scatter(
            x=list(range(1, actual_frames+1)), y=track_counts,
            mode="lines+markers", name="Tracks",
            line=dict(color="#00ff88", width=2),
        ))
        fig_det = _plotly_dark(fig_det, 300)
        fig_det.update_layout(xaxis_title="Frame", yaxis_title="Count")
        st.plotly_chart(fig_det, use_container_width=True)

    with c3:
        st.markdown("**Anomaly Radar — Top 3 Threats**")
        top3 = sorted(risk_results.items(),
                      key=lambda x: x[1]["risk_score"], reverse=True)[:3]
        fig_radar = go.Figure()
        cats = ["Speed", "Direction", "Loitering", "Convergence", "Transformer"]
        for tid, data in top3:
            bd = data["anomaly_breakdown"]
            vals = [bd["speed"], bd["direction"], bd["loitering"],
                    bd["convergence"], bd["transformer"]]
            vals.append(vals[0])
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=cats + [cats[0]], fill="toself",
                name=f"T{tid} [{data['risk_level']}]",
                opacity=0.6,
            ))
        fig_radar = _plotly_dark(fig_radar, 300)
        fig_radar.update_layout(
            polar=dict(bgcolor="#060c14",
                       radialaxis=dict(visible=True, range=[0, 1], color="#3a5a7a"),
                       angularaxis=dict(color="#8aa0b8")),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Detection Confusion / Distribution Matrix ──
    st.markdown("---")
    st.markdown("### Detection Class Matrix & Confidence Analysis")

    # Build class × frame matrix
    from models.detector import DOTA_CLASSES
    detected_classes = sorted({d.class_name for frame_dets in all_frame_detections for d in frame_dets})
    if detected_classes:
        # --- Class × Frame heatmap ---
        matrix_data = []
        for cls in detected_classes:
            row = []
            for fidx, frame_dets in enumerate(all_frame_detections):
                count = sum(1 for d in frame_dets if d.class_name == cls)
                row.append(count)
            matrix_data.append(row)

        cm1, cm2 = st.columns(2)
        with cm1:
            st.markdown("**Class × Frame Detection Matrix**")
            frame_labels = [f"F{i+1}" for i in range(len(all_frame_detections))]
            fig_matrix = go.Figure(data=go.Heatmap(
                z=matrix_data,
                x=frame_labels,
                y=detected_classes,
                colorscale="YlOrRd",
                text=matrix_data,
                texttemplate="%{text}",
                textfont={"size": 11, "color": "white"},
                hovertemplate="Class: %{y}<br>Frame: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_matrix = _plotly_dark(fig_matrix, max(300, len(detected_classes) * 30 + 100))
            fig_matrix.update_layout(xaxis_title="Temporal Frame", yaxis_title="Object Class")
            st.plotly_chart(fig_matrix, use_container_width=True)

        # --- Confidence distribution per class (box plot) ---
        with cm2:
            st.markdown("**Per-Class Confidence Distribution**")
            conf_data = []
            for frame_dets in all_frame_detections:
                for d in frame_dets:
                    conf_data.append({"Class": d.class_name, "Confidence": d.confidence})
            if conf_data:
                df_conf = pd.DataFrame(conf_data)
                fig_conf = px.box(
                    df_conf, x="Class", y="Confidence",
                    color="Class", points="outliers",
                )
                fig_conf = _plotly_dark(fig_conf, max(300, len(detected_classes) * 30 + 100))
                fig_conf.update_layout(showlegend=False, xaxis_tickangle=-45,
                                       xaxis_title="Object Class", yaxis_title="Confidence")
                st.plotly_chart(fig_conf, use_container_width=True)

        # --- Class co-occurrence / confusion-style matrix ---
        st.markdown("**Inter-Class Spatial Co-occurrence Matrix**")
        st.caption("Normalized co-occurrence: how often two classes appear in the same spatial region (128px proximity)")
        n_cls = len(detected_classes)
        cls_idx = {c: i for i, c in enumerate(detected_classes)}
        cooccur = np.zeros((n_cls, n_cls), dtype=np.float64)
        for frame_dets in all_frame_detections:
            for i, d1 in enumerate(frame_dets):
                for d2 in frame_dets[i+1:]:
                    dist = np.sqrt((d1.cx - d2.cx)**2 + (d1.cy - d2.cy)**2)
                    if dist < 128:
                        ci, cj = cls_idx[d1.class_name], cls_idx[d2.class_name]
                        cooccur[ci][cj] += 1
                        cooccur[cj][ci] += 1
        # Normalize rows
        row_sums = cooccur.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cooccur_norm = cooccur / row_sums

        fig_cooccur = go.Figure(data=go.Heatmap(
            z=np.round(cooccur_norm, 2).tolist(),
            x=detected_classes,
            y=detected_classes,
            colorscale="Viridis",
            text=np.round(cooccur_norm, 2).tolist(),
            texttemplate="%{text}",
            textfont={"size": 10, "color": "white"},
            hovertemplate="%{y} ↔ %{x}: %{z:.2f}<extra></extra>",
        ))
        fig_cooccur = _plotly_dark(fig_cooccur, max(400, n_cls * 35 + 100))
        fig_cooccur.update_layout(xaxis_title="Class", yaxis_title="Class",
                                   xaxis_tickangle=-45)
        st.plotly_chart(fig_cooccur, use_container_width=True)

    # ── Trajectory Plot ──
    st.markdown("---")
    st.markdown("### Object Trajectories")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig_traj = go.Figure()
        for tid, tdata in trajectories.items():
            traj = tdata["trajectory"]
            xs = [pt["cx"] for pt in traj]
            ys = [pt["cy"] for pt in traj]
            rl = risk_results.get(tid, {}).get("risk_level", "LOW")
            fig_traj.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=f"T{tid} ({tdata['class_name']}) [{rl}]",
                line=dict(color=risk_hex(rl), width=2),
                marker=dict(size=3),
            ))
        fig_traj = _plotly_dark(fig_traj, 500)
        fig_traj.update_layout(yaxis=dict(autorange="reversed"),
                               xaxis_title="X (pixels)", yaxis_title="Y (pixels)",
                               title="All Tracked Object Paths")
        st.plotly_chart(fig_traj, use_container_width=True)

    with c2:
        if hotspots:
            st.markdown("**Anomaly Hotspots**")
            st.dataframe(pd.DataFrame([{
                "Rank": i+1,
                "Location": f"({hs['center_x']:.0f}, {hs['center_y']:.0f})",
                "Intensity": f"{hs['intensity']:.4f}",
            } for i, hs in enumerate(hotspots[:10])]),
                use_container_width=True, hide_index=True)

        st.markdown("**Heatmap (Interactive)**")
        fig_heat = px.imshow(heatmap, color_continuous_scale="YlOrRd",
                             labels=dict(color="Anomaly"), aspect="equal")
        fig_heat = _plotly_dark(fig_heat, 300)
        st.plotly_chart(fig_heat, use_container_width=True)

    # ── Geo-Intelligence Map ──
    if FOLIUM_AVAILABLE:
        st.markdown("---")
        st.markdown("### Geo-Intelligence Map")
        try:
            threat_map = create_threat_map(
                risk_results, trajectories, geo_info, hotspots
            )
            st_folium(threat_map, width=None, height=500, returned_objects=[])
        except Exception as e:
            st.warning(f"Map rendering failed: {e}")

    # ── Report ──
    st.markdown("---")
    st.markdown("### Export Report")

    report = {
        "system": "STAD - Spatio-Temporal Anomaly Detection",
        "timestamp": datetime.now().isoformat(),
        "region": selected_region["name"],
        "pipeline": {
            "frames_analyzed": actual_frames,
            "date_range": f"{temporal_scenes[0]['date']} to {temporal_scenes[-1]['date']}",
            "total_days": temporal_stats.get("total_days", 0),
        },
        "results": {
            "total_tracks": len(risk_results),
            "risk_distribution": risk_counts,
            "hotspots_detected": len(hotspots),
        },
        "threats": [
            {
                "track_id": tid,
                "class": data["class_name"],
                "risk_score": round(data["risk_score"], 4),
                "risk_level": data["risk_level"],
                "anomaly_breakdown": {k: round(v, 4) for k, v in data["anomaly_breakdown"].items()},
            }
            for tid, data in sorted(risk_results.items(),
                                     key=lambda x: x[1]["risk_score"], reverse=True)
        ],
    }

    # ── Generate PDF Report ──
    def generate_pdf_report(report_data, det_vis_img, heatmap_img, traj_img):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm, inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
            PageBreak,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        import io
        import tempfile

        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=A4,
            leftMargin=20*mm, rightMargin=20*mm,
            topMargin=20*mm, bottomMargin=20*mm,
        )

        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(
            name="Title2", parent=styles["Title"],
            fontSize=22, textColor=HexColor("#003366"),
            spaceAfter=6,
        ))
        styles.add(ParagraphStyle(
            name="SubTitle", parent=styles["Normal"],
            fontSize=11, textColor=HexColor("#446688"),
            spaceAfter=14,
        ))
        styles.add(ParagraphStyle(
            name="SectionHead", parent=styles["Heading2"],
            fontSize=14, textColor=HexColor("#004488"),
            spaceBefore=16, spaceAfter=8,
        ))
        styles.add(ParagraphStyle(
            name="BodyText2", parent=styles["Normal"],
            fontSize=10, leading=14,
        ))

        elements = []

        # Title
        elements.append(Paragraph("STAD — Spatio-Temporal Anomaly Detection", styles["Title2"]))
        elements.append(Paragraph(
            f"Region: {report_data['region']}<br/>"
            f"Generated: {report_data['timestamp'][:19]}",
            styles["SubTitle"],
        ))
        elements.append(Spacer(1, 8*mm))

        # Pipeline Summary
        elements.append(Paragraph("Pipeline Summary", styles["SectionHead"]))
        pipe = report_data["pipeline"]
        summary_data = [
            ["Parameter", "Value"],
            ["Frames Analyzed", str(pipe["frames_analyzed"])],
            ["Date Range", pipe["date_range"]],
            ["Total Days Coverage", str(pipe["total_days"])],
            ["Detector", "YOLOv8-OBB (Multi-Scale Tiled)"],
            ["Tracker", "ByteTrack"],
            ["Temporal Model", "Transformer Encoder (4-layer)"],
            ["Total Tracked Objects", str(report_data["results"]["total_tracks"])],
            ["Hotspots Detected", str(report_data["results"]["hotspots_detected"])],
        ]
        t = Table(summary_data, colWidths=[55*mm, 105*mm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 1), (-1, -1), HexColor("#f0f4f8")),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#aabbcc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 6*mm))

        # Risk Distribution
        elements.append(Paragraph("Risk Distribution", styles["SectionHead"]))
        rd = report_data["results"]["risk_distribution"]
        risk_data = [
            ["Risk Level", "Count"],
            ["LOW", str(rd.get("LOW", 0))],
            ["MEDIUM", str(rd.get("MEDIUM", 0))],
            ["HIGH", str(rd.get("HIGH", 0))],
            ["CRITICAL", str(rd.get("CRITICAL", 0))],
        ]
        rt = Table(risk_data, colWidths=[55*mm, 55*mm])
        risk_colors = {"LOW": "#33cc33", "MEDIUM": "#cc9900", "HIGH": "#ff6600", "CRITICAL": "#cc0000"}
        rt_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#aabbcc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, level in enumerate(["LOW", "MEDIUM", "HIGH", "CRITICAL"], start=1):
            rt_style.append(("TEXTCOLOR", (0, i), (0, i), HexColor(risk_colors[level])))
            rt_style.append(("FONTNAME", (0, i), (0, i), "Helvetica-Bold"))
        rt.setStyle(TableStyle(rt_style))
        elements.append(rt)
        elements.append(Spacer(1, 6*mm))

        # Detection image
        def cv2_img_to_rl(cv_img, width_mm=160):
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) if len(cv_img.shape) == 3 else cv_img
            pil_img = Image.fromarray(img_rgb)
            img_buf = io.BytesIO()
            pil_img.save(img_buf, format="PNG")
            img_buf.seek(0)
            aspect = pil_img.height / max(pil_img.width, 1)
            return RLImage(img_buf, width=width_mm*mm, height=width_mm*mm*aspect)

        elements.append(Paragraph("Detection Output — YOLOv8-OBB", styles["SectionHead"]))
        elements.append(cv2_img_to_rl(det_vis_img, width_mm=160))
        elements.append(Spacer(1, 6*mm))

        elements.append(PageBreak())

        # Trajectory + Heatmap
        elements.append(Paragraph("Trajectory Trails (Risk-Colored)", styles["SectionHead"]))
        elements.append(cv2_img_to_rl(traj_img, width_mm=155))
        elements.append(Spacer(1, 6*mm))

        elements.append(Paragraph("Spatial Anomaly Heatmap", styles["SectionHead"]))
        elements.append(cv2_img_to_rl(heatmap_img, width_mm=155))
        elements.append(Spacer(1, 6*mm))

        # Threat Table
        elements.append(PageBreak())
        elements.append(Paragraph("Anomaly Scores — All Tracked Objects", styles["SectionHead"]))
        threat_header = ["Track", "Class", "Risk Score", "Level", "Speed", "Dir", "Loiter", "Conv", "Trans"]
        threat_table = [threat_header]
        for t in report_data["threats"]:
            bd = t["anomaly_breakdown"]
            threat_table.append([
                str(t["track_id"]),
                t["class"],
                f"{t['risk_score']:.4f}",
                t["risk_level"],
                f"{bd.get('speed', 0):.3f}",
                f"{bd.get('direction', 0):.3f}",
                f"{bd.get('loitering', 0):.3f}",
                f"{bd.get('convergence', 0):.3f}",
                f"{bd.get('transformer', 0):.3f}",
            ])
        col_w = [14*mm, 22*mm, 20*mm, 18*mm, 16*mm, 14*mm, 16*mm, 14*mm, 14*mm]
        tt = Table(threat_table, colWidths=col_w)
        tt_style = [
            ("BACKGROUND", (0, 0), (-1, 0), HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#aabbcc")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#ffffff"), HexColor("#f0f4f8")]),
        ]
        # Color risk level cells
        for row_idx in range(1, len(threat_table)):
            level = threat_table[row_idx][3]
            color = risk_colors.get(level, "#888888")
            tt_style.append(("TEXTCOLOR", (3, row_idx), (3, row_idx), HexColor(color)))
            tt_style.append(("FONTNAME", (3, row_idx), (3, row_idx), "Helvetica-Bold"))
        tt.setStyle(TableStyle(tt_style))
        elements.append(tt)

        # Footer
        elements.append(Spacer(1, 12*mm))
        elements.append(Paragraph(
            "STAD — Spatio-Temporal Anomaly Detection | "
            "NAIP 0.6m Aerial Imagery | YOLOv8-OBB | ByteTrack | Transformer",
            ParagraphStyle(name="Footer", parent=styles["Normal"],
                           fontSize=8, textColor=HexColor("#888888"), alignment=TA_CENTER),
        ))

        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    pdf_bytes = generate_pdf_report(report, last_det_vis, heatmap_vis, traj_vis)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"STAD_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )
    with c2:
        st.download_button(
            "Download JSON Report",
            data=json.dumps(report, indent=2),
            file_name=f"STAD_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )


# ── Footer ──
st.markdown("""
<div style="text-align: center; color: #2a4a6a; padding: 16px; font-size: 0.75em;
            border-top: 1px solid #1a3050; margin-top: 24px;">
    STAD &mdash; Spatio-Temporal Anomaly Detection &bull;
    NAIP 0.6m High-Resolution Imagery &bull; YOLOv8-OBB &bull; ByteTrack &bull; Transformer &bull; Anomaly Scoring
</div>
""", unsafe_allow_html=True)
