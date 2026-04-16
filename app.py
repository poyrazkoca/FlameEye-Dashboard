import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# External integrations requested by spec (with safe fallbacks).
try:
    from Constants import REFERENCE_LAT, REFERENCE_LON  # type: ignore
except Exception:
    REFERENCE_LAT, REFERENCE_LON = 34.05, -118.24

try:
    from FlameEye_Sim import SimulationController  # type: ignore
except Exception:
    class SimulationController:  # fallback shim
        def __init__(self) -> None:
            self.running = False

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

        def get_frame(self) -> np.ndarray:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "Simulation Frame",
                (170, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )
            return frame

try:
    from FlameEye_Detector import Detector  # type: ignore
except Exception:
    class Detector:  # fallback shim
        def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
            return []

try:
    from MappingModule import pixel_to_lat_lon  # type: ignore
except Exception:
    def pixel_to_lat_lon(
        x: int,
        y: int,
        width: int,
        height: int,
        ref_lat: float,
        ref_lon: float,
    ) -> Tuple[float, float]:
        lat_offset = (y - height / 2) / max(height, 1) * 0.02
        lon_offset = (x - width / 2) / max(width, 1) * 0.02
        return ref_lat - lat_offset, ref_lon + lon_offset

try:
    from JsonHandler import JsonHandler  # type: ignore
except Exception:
    import json
    from pathlib import Path

    class JsonHandler:  # fallback shim
        def __init__(self, json_path: str = "archived_fires.json") -> None:
            self.json_path = Path(json_path)

        def get_archived_fires(self) -> List[Dict[str, Any]]:
            if not self.json_path.exists():
                return []
            try:
                with self.json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except Exception:
                return []


st.set_page_config(page_title="FlameEye AI Dashboard", layout="wide")
st.title("FlameEye AI - Real-time Wildfire Monitoring & Incident Response")


def initialize_state() -> None:
    if "controller" not in st.session_state:
        st.session_state.controller = SimulationController()
    if "detector" not in st.session_state:
        st.session_state.detector = Detector()
    if "json_handler" not in st.session_state:
        st.session_state.json_handler = JsonHandler()
    if "system_state" not in st.session_state:
        st.session_state.system_state = "Initializing"
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "selected_alert_idx" not in st.session_state:
        st.session_state.selected_alert_idx = 0
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Optical"


def updateMapLayer(
    base_lat: float,
    base_lon: float,
    live_alerts: List[Dict[str, Any]],
    archived_alerts: List[Dict[str, Any]],
) -> folium.Map:
    m = folium.Map(location=[base_lat, base_lon], zoom_start=12, control_scale=True)

    for archived in archived_alerts:
        lat = archived.get("lat", base_lat)
        lon = archived.get("lon", base_lon)
        alarm_id = archived.get("alarm_id", "ARCHIVED")
        confidence = archived.get("confidence", 0.0)
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#FF4B2B",
            fill=True,
            fill_color="#FF4B2B",
            fill_opacity=0.6,
            popup=f"Archived Fire Alarm {alarm_id} | Confidence: {confidence:.2f}",
        ).add_to(m)

    for alert in live_alerts:
        folium.Marker(
            location=[alert["lat"], alert["lon"]],
            icon=folium.Icon(color="red", icon="fire"),
            popup=(
                f"Fire Alarm ID: {alert['alarm_id']}<br>"
                f"Confidence: {alert['confidence']:.2f}"
            ),
        ).add_to(m)

    return m


def draw_detection_boxes(
    frame: np.ndarray, detections: List[Dict[str, Any]], mode: str
) -> np.ndarray:
    output = frame.copy()
    if mode == "Thermal":
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        output = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        confidence = det["confidence"]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            output,
            f"Fire {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
    return output


def sidebar_ui() -> Tuple[Tuple[datetime, datetime], str, str]:
    st.sidebar.header("Global Stats & Filters")
    active_fires = len(st.session_state.alerts)
    area_affected_km2 = active_fires * 0.75
    new_alerts = sum(
        1
        for a in st.session_state.alerts
        if (datetime.utcnow() - a["timestamp"]).total_seconds() <= 300
    )

    st.sidebar.metric("Active Fires", active_fires)
    st.sidebar.metric("Area Affected (km^2)", f"{area_affected_km2:.2f}")
    st.sidebar.metric("New Alerts", new_alerts)

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(datetime.utcnow().date(), datetime.utcnow().date()),
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = datetime.combine(date_range[0], datetime.min.time())
        end_date = datetime.combine(date_range[1], datetime.max.time())
    else:
        now = datetime.utcnow()
        start_date, end_date = now, now

    severity = st.sidebar.selectbox("Severity", options=["Low", "High"], index=1)
    status = st.sidebar.selectbox(
        "Status", options=["Open", "Investigating", "Resolved"], index=0
    )

    st.sidebar.subheader("Live Alerts")
    if st.session_state.alerts:
        sorted_alerts = sorted(
            st.session_state.alerts, key=lambda x: x["timestamp"], reverse=True
        )
        for alert in sorted_alerts[:20]:
            st.sidebar.caption(
                f"[{alert['timestamp'].strftime('%H:%M:%S')}] "
                f"{alert['alarm_id']} | {alert['confidence']:.2f}"
            )
    else:
        st.sidebar.caption("No active alerts.")

    return (start_date, end_date), severity, status


def run_dashboard() -> None:
    initialize_state()
    (start_date, end_date), severity, status = sidebar_ui()

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.write(f"System State: **{st.session_state.system_state}**")
    with top_right:
        run_toggle = st.toggle("Start/Stop Simulation", value=False)

    if run_toggle and st.session_state.system_state != "Running":
        st.session_state.system_state = "Running"
        if hasattr(st.session_state.controller, "start"):
            st.session_state.controller.start()
    elif not run_toggle and st.session_state.system_state == "Running":
        st.session_state.system_state = "Stopped"
        if hasattr(st.session_state.controller, "stop"):
            st.session_state.controller.stop()

    left_col, right_col = st.columns([3, 2])  # ~60/40
    map_placeholder = left_col.empty()
    video_placeholder = right_col.empty()

    st.subheader("Incident Response")
    if st.session_state.alerts:
        alert_labels = [
            f"{a['alarm_id']} ({a['confidence']:.2f})" for a in st.session_state.alerts
        ]
        st.session_state.selected_alert_idx = st.selectbox(
            "Selected Alarm",
            options=range(len(alert_labels)),
            index=min(st.session_state.selected_alert_idx, len(alert_labels) - 1),
            format_func=lambda i: alert_labels[i],
        )
        selected_alert = st.session_state.alerts[st.session_state.selected_alert_idx]
        st.write(f"ID: `{selected_alert['alarm_id']}`")
        st.write(f"Confidence Score: `{selected_alert['confidence']:.2f}`")
        st.write(
            f"GPS Coordinates: `{selected_alert['lat']:.4f}, {selected_alert['lon']:.4f}`"
        )
    else:
        st.write("ID: `N/A`")
        st.write("Confidence Score: `N/A`")
        st.write("GPS Coordinates: `34.05 N, 118.24 W`")

    act1, act2 = st.columns(2)
    if act1.button("Dispatch Team", type="primary"):
        st.success("Response team dispatched.")
    if act2.button("Mark as False Alarm"):
        if st.session_state.alerts:
            st.session_state.alerts.pop(st.session_state.selected_alert_idx)
            st.info("Alert marked as false alarm and removed.")
        else:
            st.info("No alert selected.")

    st.markdown("---")
    st.subheader("Sensor Data Viewer")
    st.session_state.view_mode = st.radio(
        "Telemetry Mode", options=["Thermal", "Optical"], horizontal=True
    )
    st.caption(
        f"Filters applied | Date: {start_date.date()} - {end_date.date()} | "
        f"Severity: {severity} | Status: {status}"
    )

    # Required running-state loop from sequence design.
    if st.session_state.system_state == "Running":
        while True:
            frame = st.session_state.controller.get_frame()
            detections = st.session_state.detector.detect(frame) or []

            current_alerts: List[Dict[str, Any]] = []
            h, w = frame.shape[:2]

            for idx, det in enumerate(detections):
                bbox = det.get("bbox", [0, 0, 10, 10])
                confidence = float(det.get("confidence", 0.0))
                x1, y1, x2, y2 = [int(v) for v in bbox]
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                lat, lon = pixel_to_lat_lon(
                    center_x, center_y, w, h, REFERENCE_LAT, REFERENCE_LON
                )

                current_alerts.append(
                    {
                        "alarm_id": f"FIRE-{int(time.time())}-{idx}",
                        "confidence": confidence,
                        "lat": lat,
                        "lon": lon,
                        "bbox": [x1, y1, x2, y2],
                        "timestamp": datetime.utcnow(),
                    }
                )

            if current_alerts:
                st.session_state.alerts.extend(current_alerts)
                st.session_state.alerts = sorted(
                    st.session_state.alerts,
                    key=lambda x: x["timestamp"],
                    reverse=True,
                )[:200]

            rendered_frame = draw_detection_boxes(
                frame,
                [
                    {"bbox": a["bbox"], "confidence": a["confidence"]}
                    for a in current_alerts
                ],
                st.session_state.view_mode,
            )
            video_placeholder.image(
                cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB),
                channels="RGB",
                caption="Live Video Feed (YOLOv8 Processed)",
                use_container_width=True,
            )

            archived = st.session_state.json_handler.get_archived_fires()
            map_obj = updateMapLayer(
                REFERENCE_LAT, REFERENCE_LON, st.session_state.alerts, archived
            )
            with map_placeholder.container():
                st_folium(map_obj, use_container_width=True, height=520, key="fire_map")

            # 1-2 FPS target
            time.sleep(0.7)
            if st.session_state.system_state != "Running":
                break
    else:
        archived = st.session_state.json_handler.get_archived_fires()
        map_obj = updateMapLayer(REFERENCE_LAT, REFERENCE_LON, st.session_state.alerts, archived)
        with map_placeholder.container():
            st_folium(map_obj, use_container_width=True, height=520, key="idle_map")
        video_placeholder.info("Simulation stopped. Toggle Start/Stop to begin streaming.")


if __name__ == "__main__":
    run_dashboard()
