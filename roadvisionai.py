import pandas as pd
import plotly.express as px
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import st_folium
import folium
import cv2
from PIL import Image, ExifTags
import numpy as np
from ultralytics import YOLO
import os
import io
import time
import json
import zipfile
import tempfile
from collections import deque
from datetime import datetime

import streamlit as st
st.set_page_config(page_title="RoadVision AI — Full Suite",
                   page_icon="🛣️", layout="wide")

# Core ML / CV

# Mapping & viz

# WebRTC for browser camera
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    WEBRTC_OK = True
except Exception:
    WEBRTC_OK = False

# Optional OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# Optional JS -> Streamlit bridge for live browser GPS (install streamlit-javascript)
BROWSER_GPS_OK = False
try:
    # This package provides a helper to run JS and return values
    from streamlit_javascript import st_javascript
    BROWSER_GPS_OK = True
except Exception:
    BROWSER_GPS_OK = False

# --------------------------
# CONFIG: update model filenames here (put your .pt files in weights/)
# --------------------------
MODEL_PATHS = {
    "Road Markings (YOLOv11)": "road_markings_model.pt",
    "Road Defects (YOLOv11)": "road_defects_model .pt",
}

os.makedirs("outputs", exist_ok=True)

# --------------------------
# Utility & helper functions
# --------------------------


@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)


def image_bytes_to_np(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def np_to_bytes(img_np, fmt="PNG"):
    im = Image.fromarray(img_np)
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


def save_image_file(img_np, outpath):
    Image.fromarray(img_np).save(outpath)


def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def severity_score(box, cls_name, conf, img_shape):
    x1, y1, x2, y2 = box
    area = max(0, (x2-x1)*(y2-y1))
    img_area = img_shape[0]*img_shape[1]
    rel_area = area / max(img_area, 1)
    base = rel_area * 100
    if "pothole" in cls_name.lower():
        base *= 2.5
    if "crack" in cls_name.lower():
        base *= 1.8
    score = (base * conf) * 10
    return min(100, score)


def boxes_from_result(result):
    boxes = []
    if hasattr(result, "boxes") and len(result.boxes) > 0:
        for b in result.boxes:
            try:
                xy = b.xyxy[0].cpu().numpy() if hasattr(
                    b, "xyxy") else np.array([0, 0, 0, 0])
                conf = float(b.conf[0]) if hasattr(
                    b, "conf") else float(b.conf)
                cls = int(b.cls[0]) if hasattr(b, "cls") else int(b.cls)
            except Exception:
                try:
                    arr = b.cpu().numpy()
                    xy = arr[:4]
                    conf = float(arr[4])
                    cls = int(arr[5])
                except Exception:
                    continue
            boxes.append((float(xy[0]), float(xy[1]),
                         float(xy[2]), float(xy[3]), conf, cls))
    return boxes


def result_to_geojson(detections):
    features = []
    for d in detections:
        if d.get("lat") is None or d.get("lng") is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {k: v for k, v in d.items() if k not in ("lat", "lng")},
            "geometry": {"type": "Point", "coordinates": [d["lng"], d["lat"]]}
        })
    return {"type": "FeatureCollection", "features": features}


def download_button_bytes(data: bytes, file_name: str, mime: str):
    st.download_button(
        label=f"Download {file_name}", data=data, file_name=file_name, mime=mime)


def simple_exp_smooth(series, alpha=0.4):
    if len(series) == 0:
        return []
    s = series[0]
    preds = []
    for x in series:
        s = alpha*x + (1-alpha)*s
        preds.append(s)
    return preds

# --------------------------
# EXIF GPS extraction
# --------------------------
# helper to parse EXIF GPS from PIL image


def get_exif_gps(file_like):
    try:
        img = Image.open(file_like)
        exif = img._getexif()
        if not exif:
            return None, None
        gps_info = {}
        for tag_id, value in exif.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for key in value:
                    subtag = ExifTags.GPSTAGS.get(key, key)
                    gps_info[subtag] = value[key]
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            def _conv(coord):
                d = coord[0][0] / coord[0][1]
                m = coord[1][0] / coord[1][1]
                s = coord[2][0] / coord[2][1]
                return d + m/60.0 + s/3600.0
            lat = _conv(gps_info["GPSLatitude"])
            lng = _conv(gps_info["GPSLongitude"])
            if gps_info.get("GPSLatitudeRef") == "S":
                lat = -lat
            if gps_info.get("GPSLongitudeRef") == "W":
                lng = -lng
            return lat, lng
    except Exception:
        return None, None
    return None, None

# --------------------------
# Browser live GPS helper (optional)
# - requires 'streamlit-javascript' package: pip install streamlit-javascript
# - If installed, the app polls browser geolocation and returns latest coords.
# --------------------------


def get_browser_gps():
    """
    Returns (lat, lng) if the optional JS bridge is installed and user allows browser location.
    Otherwise returns (None, None).
    """
    if not BROWSER_GPS_OK:
        return None, None
    try:
        # This JS will ask for geolocation and return {lat, lng}
        js = """
        async () => {
          if (!navigator.geolocation) return {lat: null, lng: null};
          const p = new Promise((res, rej) => {
            navigator.geolocation.getCurrentPosition((pos)=> {
              res({lat: pos.coords.latitude, lng: pos.coords.longitude});
            }, (err) => {
              res({lat: null, lng: null});
            }, {enableHighAccuracy: true});
          });
          return await p;
        }
        """
        val = st_javascript(js, key="gps_js_call")
        if isinstance(val, dict):
            lat = val.get("lat")
            lng = val.get("lng")
            if lat and lng:
                return float(lat), float(lng)
    except Exception:
        return None, None
    return None, None


# --------------------------
# Sidebar: settings & models
# --------------------------
st.sidebar.header("Settings & Models")
selected_model = st.sidebar.selectbox(
    "Primary model", list(MODEL_PATHS.keys()))
secondary_model = st.sidebar.selectbox("Secondary model", list(
    MODEL_PATHS.keys()), index=1 if len(MODEL_PATHS) > 1 else 0)
conf_thresh = st.sidebar.slider(
    "Confidence threshold", 0.1, 1.0, 0.4, step=0.05)
show_heatmap = st.sidebar.checkbox("Show heatmap on map", value=True)
use_gps = st.sidebar.checkbox(
    "Attach GPS (try EXIF / Browser / Manual fallback)", value=False)
simulate_gps = st.sidebar.checkbox(
    "Simulate GPS if none found (dev)", value=False)
manual_lat = st.sidebar.number_input(
    "Manual lat (optional)", value=0.0, format="%.6f")
manual_lng = st.sidebar.number_input(
    "Manual lng (optional)", value=0.0, format="%.6f")
use_manual_if_missing = st.sidebar.checkbox(
    "Use manual location when GPS missing", value=False)
openai_key_input = st.sidebar.text_input(
    "OpenAI API Key (optional)", type="password")
if openai_key_input:
    os.environ["OPENAI_API_KEY"] = openai_key_input
    if OPENAI_AVAILABLE:
        openai.api_key = openai_key_input

# Load models (cached)
model_primary = load_yolo_model(MODEL_PATHS[selected_model])
model_secondary = load_yolo_model(MODEL_PATHS[secondary_model])

# --------------------------
# App layout
# --------------------------
st.title("🛣️ RoadVision AI — Live Inspection Suite")
st.markdown(
    "Dual YOLOv11 models for road markings & defects + mobile live camera, mapping & analytics")

tabs = st.tabs(["Demo & Live", "Map & Geo", "Dashboard",
               "Batch / Export", "AI Assistant", "About"])

# session storage for detections
if "detections" not in st.session_state:
    st.session_state["detections"] = []  # list[dict]

# session storage for manual map pin
if "manual_pin" not in st.session_state:
    st.session_state["manual_pin"] = {"lat": None, "lng": None}

# --------------------------
# TAB: Demo & Live
# --------------------------
with tabs[0]:
    st.header("Live Demo & Inputs")
    colL, colR = st.columns([0.65, 0.35])
    with colL:
        mode = st.radio(
            "Input mode", ["Upload Image", "Upload Video", "Live Camera", "Sample"])
        if mode == "Upload Image":
            up = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
            if up:
                img_np = image_bytes_to_np(up)
                st.image(img_np, caption="Input image", use_column_width=True)
                if st.button("Run detection on image"):
                    t0 = time.time()
                    res = model_primary.predict(
                        img_np, conf=conf_thresh, verbose=False)
                    annotated = res[0].plot()
                    st.image(annotated, caption="Detections",
                             use_column_width=True)
                    boxes = boxes_from_result(res[0])
                    names = getattr(res[0], "names", {})
                    # EXIF GPS for the uploaded image (if present)
                    exif_lat, exif_lng = get_exif_gps(
                        up) if use_gps else (None, None)
                    # get latest browser gps if available
                    browser_lat, browser_lng = (
                        get_browser_gps() if use_gps else (None, None))
                    for (x1, y1, x2, y2, conf, cls) in boxes:
                        cls_name = names.get(cls, str(cls))
                        # choose gps source: EXIF -> Browser -> Manual -> Simulate -> None
                        lat, lng = None, None
                        if exif_lat and exif_lng:
                            lat, lng = exif_lat, exif_lng
                        elif browser_lat and browser_lng:
                            lat, lng = browser_lat, browser_lng
                        elif use_manual_if_missing and st.session_state["manual_pin"]["lat"] and st.session_state["manual_pin"]["lng"]:
                            lat = st.session_state["manual_pin"]["lat"]
                            lng = st.session_state["manual_pin"]["lng"]
                        elif simulate_gps:
                            lat = 5.6037 + (np.random.rand()-0.5)*0.02
                            lng = -0.1870 + (np.random.rand()-0.5)*0.02
                        sev = severity_score(
                            (x1, y1, x2, y2), cls_name, conf, img_np.shape)
                        st.session_state["detections"].append({
                            "ts": timestamp(), "class": cls_name, "conf": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "lat": lat, "lng": lng, "severity": float(sev),
                            "image": getattr(up, "name", "uploaded_image")
                        })
                    st.success(
                        f"Done — {len(boxes)} objects — {time.time()-t0:.2f}s")

        elif mode == "Upload Video":
            v = st.file_uploader("Upload MP4 (short preferred)", type=[
                                 "mp4", "mov", "avi"])
            if v:
                tmp = tempfile.NamedTemporaryFile(delete=False)
                tmp.write(v.read())
                tmp.flush()
                st.video(tmp.name)
                if st.button("Process Video"):
                    st.info("Processing video frames — this may be slow on CPU.")
                    cap = cv2.VideoCapture(tmp.name)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_path = tmp.name + "_out.mp4"
                    writer = cv2.VideoWriter(
                        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    pbar = st.progress(0)
                    i = 0
                    # Try to extract video-level GPS? Many phones do not embed GPS in MP4; so we skip EXIF for video
                    # For now, use browser GPS or manual / simulate as fallback
                    browser_lat, browser_lng = (
                        get_browser_gps() if use_gps else (None, None))
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        res = model_primary.predict(
                            rgb, conf=conf_thresh, verbose=False)
                        ann = res[0].plot()
                        writer.write(cv2.cvtColor(ann, cv2.COLOR_RGB2BGR))
                        boxes = boxes_from_result(res[0])
                        names = getattr(res[0], "names", {})
                        for (x1, y1, x2, y2, conf, cls) in boxes:
                            cls_name = names.get(cls, str(cls))
                            lat, lng = None, None
                            if browser_lat and browser_lng:
                                lat, lng = browser_lat, browser_lng
                            elif use_manual_if_missing and st.session_state["manual_pin"]["lat"] and st.session_state["manual_pin"]["lng"]:
                                lat = st.session_state["manual_pin"]["lat"]
                                lng = st.session_state["manual_pin"]["lng"]
                            elif simulate_gps:
                                lat = 5.6037 + (np.random.rand()-0.5)*0.02
                                lng = -0.1870 + (np.random.rand()-0.5)*0.02
                            sev = severity_score(
                                (x1, y1, x2, y2), cls_name, conf, rgb.shape)
                            st.session_state["detections"].append({
                                "ts": timestamp(), "class": cls_name, "conf": float(conf),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "lat": lat, "lng": lng, "severity": float(sev), "image": os.path.basename(tmp.name)
                            })
                        i += 1
                        if total > 0:
                            pbar.progress(min(i/total, 1.0))
                    writer.release()
                    cap.release()
                    st.success("Video processed")
                    st.video(out_path)
                    with open(out_path, "rb") as f:
                        st.download_button(
                            "Download processed video", f, file_name="processed_video.mp4")

        elif mode == "Live Camera":
            if not WEBRTC_OK:
                st.warning(
                    "streamlit-webrtc or av is not installed. Install streamlit-webrtc and av to use Live Camera.")
            else:
                st.info(
                    "Start live camera from your browser. Choose camera facing mode and allow camera permission when prompted.")
                camera_choice = st.radio(
                    "Camera", ["Front (selfie)", "Back (rear / environment)"], index=1)
                if camera_choice == "Back (rear / environment)":
                    media_constraints = {"video": {"facingMode": {
                        "ideal": "environment"}}, "audio": False}
                else:
                    media_constraints = {
                        "video": {"facingMode": "user"}, "audio": False}

                class YOLOProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.model = model_primary

                    def recv(self, frame):
                        img = frame.to_ndarray(format="bgr24")
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        results = self.model.predict(
                            rgb, conf=conf_thresh, verbose=False)
                        annotated = results[0].plot()

                        # push detections to session_state (lightweight)
                        boxes = boxes_from_result(results[0])
                        names = getattr(results[0], "names", {})
                        # read browser gps (optional)
                        browser_lat, browser_lng = (
                            get_browser_gps() if use_gps else (None, None))
                        for (x1, y1, x2, y2, conf, cls) in boxes:
                            cls_name = names.get(cls, str(cls))
                            lat, lng = None, None
                            if browser_lat and browser_lng:
                                lat, lng = browser_lat, browser_lng
                            elif use_manual_if_missing and st.session_state["manual_pin"]["lat"] and st.session_state["manual_pin"]["lng"]:
                                lat = st.session_state["manual_pin"]["lat"]
                                lng = st.session_state["manual_pin"]["lng"]
                            elif simulate_gps:
                                lat = 5.6037 + (np.random.rand()-0.5)*0.02
                                lng = -0.1870 + (np.random.rand()-0.5)*0.02
                            sev = severity_score(
                                (x1, y1, x2, y2), cls_name, conf, rgb.shape)
                            st.session_state["detections"].append({
                                "ts": timestamp(), "class": cls_name, "conf": float(conf),
                                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                "lat": lat, "lng": lng, "severity": float(sev),
                                "image": "live_camera"
                            })
                        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

                webrtc_ctx = webrtc_streamer(
                    key="yolo-live",
                    video_processor_factory=YOLOProcessor,
                    media_stream_constraints=media_constraints,
                    async_processing=True
                )

        else:
            st.info("Sample images for quick demo.")
            sample_dir = "sample_data"
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir, exist_ok=True)
                blank = np.full((480, 640, 3), 220, dtype=np.uint8)
                save_image_file(blank, os.path.join(
                    sample_dir, "blank_sample.png"))
            samples = [os.path.join(sample_dir, f) for f in os.listdir(
                sample_dir) if f.lower().endswith((".png", ".jpg"))]
            choice = st.selectbox("Pick a sample", samples)
            if choice:
                img_np = cv2.cvtColor(cv2.imread(choice), cv2.COLOR_BGR2RGB)
                st.image(img_np, use_column_width=True)
                if st.button("Run detection on sample"):
                    res = model_primary.predict(
                        img_np, conf=conf_thresh, verbose=False)
                    st.image(res[0].plot(), use_column_width=True)

    with colR:
        st.subheader("Session Info")
        st.write(f"Primary model: **{selected_model}**")
        st.write(f"Secondary model: **{secondary_model}**")
        st.write(f"Confidence: **{conf_thresh:.2f}**")
        st.write(
            f"Detections stored: **{len(st.session_state['detections'])}**")
        if st.button("Clear detections"):
            st.session_state["detections"] = []
            st.success("Cleared")
        st.markdown("---")
        if st.button("Export detections CSV"):
            df = pd.DataFrame(st.session_state["detections"])
            csv = df.to_csv(index=False).encode("utf-8")
            download_button_bytes(csv, "detections.csv", "text/csv")
        if st.button("Export GeoJSON"):
            gj = result_to_geojson(st.session_state["detections"])
            data = json.dumps(gj).encode("utf-8")
            download_button_bytes(
                data, "detections.geojson", "application/geo+json")

# --------------------------
# TAB: Map & Geo
# --------------------------
with tabs[1]:
    st.header("Geospatial Map — Hotspots & Clusters")
    map_col, ctrl_col = st.columns([0.75, 0.25])
    with ctrl_col:
        cluster = st.checkbox("Cluster markers", value=True)
        heat = st.checkbox("Heatmap", value=show_heatmap)
        center_lat = st.number_input("Center lat", value=5.6037, format="%.6f")
        center_lng = st.number_input(
            "Center lng", value=-0.1870, format="%.6f")
        zoom = st.slider("Zoom", 6, 18, 13)
        if st.button("Clear geotagged detections"):
            st.session_state["detections"] = [
                d for d in st.session_state["detections"] if d.get("lat") is None]
            st.success("Cleared")
        st.markdown("### Manual tagging / fallback")
        st.write(
            "Click on the map to set a manual pin (useful when GPS not available).")
        st.write(
            "Manual pin will be used as fallback if 'Use manual location when GPS missing' is checked in the sidebar.")
        # show manual lat/lng inputs too
        if st.button("Reset manual pin"):
            st.session_state["manual_pin"] = {"lat": None, "lng": None}
            st.success("Manual pin cleared")

    with map_col:
        m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom)
        dets = [d for d in st.session_state["detections"]
                if d.get("lat") is not None]
        if cluster:
            mc = MarkerCluster()
            for d in dets:
                popup = f"Class: {d['class']}<br>Conf: {d['conf']:.2f}<br>Severity: {d['severity']:.1f}<br>Time: {d['ts']}"
                folium.Marker([d["lat"], d["lng"]], popup=popup).add_to(mc)
            m.add_child(mc)
        else:
            for d in dets:
                folium.CircleMarker(location=[d["lat"], d["lng"]], radius=6, color="red" if d["severity"]
                                    > 60 else "orange", popup=f"{d['class']} ({d['conf']:.2f})").add_to(m)
        if heat and len(dets) > 0:
            heat_data = [[d["lat"], d["lng"], d["severity"]] for d in dets]
            HeatMap(heat_data, radius=25, blur=15, max_zoom=14).add_to(m)

        # add click handling: streamlit_folium returns last_clicked
        map_data = st_folium(m, width=900, height=600)
        if map_data and map_data.get("last_clicked"):
            lat_clicked = map_data["last_clicked"]["lat"]
            lng_clicked = map_data["last_clicked"]["lng"]
            st.session_state["manual_pin"] = {
                "lat": lat_clicked, "lng": lng_clicked}
            st.success(f"Manual pin set: {lat_clicked:.6f}, {lng_clicked:.6f}")

# --------------------------
# TAB: Dashboard
# --------------------------
with tabs[2]:
    st.header("Analytics Dashboard")
    dets = pd.DataFrame(st.session_state["detections"])
    if dets.empty:
        st.info("No detections yet.")
    else:
        st.subheader("Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total detections", len(dets))
        geocount = len(dets[dets['lat'].notnull()]
                       ) if 'lat' in dets.columns else 0
        c2.metric("Geotagged", geocount)
        avg_sev = dets['severity'].mean() if 'severity' in dets.columns else 0
        c3.metric("Avg severity", f"{avg_sev:.1f}")
        if 'class' in dets.columns:
            cc = dets['class'].value_counts().reset_index()
            cc.columns = ['class', 'count']
            fig = px.pie(cc, names='class', values='count',
                         title='Detections by class')
            st.plotly_chart(fig, use_container_width=True)
        if 'ts' in dets.columns:
            dets['ts_dt'] = pd.to_datetime(dets['ts'])
            ts = dets.set_index('ts_dt').resample(
                '1T').size().rename('count').reset_index()
            fig2 = px.line(ts, x='ts_dt', y='count',
                           title='Detections per minute')
            st.plotly_chart(fig2, use_container_width=True)
            preds = simple_exp_smooth(list(ts['count']), alpha=0.4)
            if preds:
                fig2.add_scatter(x=ts['ts_dt'], y=preds,
                                 mode='lines', name='Smoothed')
                st.plotly_chart(fig2, use_container_width=True)
        if 'severity' in dets.columns:
            fig3 = px.histogram(dets, x='severity', nbins=20,
                                title='Severity histogram')
            st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# TAB: Batch / Export
# --------------------------
with tabs[3]:
    st.header("Batch Processing & Reports")
    colA, colB = st.columns([0.6, 0.4])
    with colA:
        zip_file = st.file_uploader("Upload ZIP of images", type=["zip"])
        if zip_file and st.button("Run batch detection"):
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp.write(zip_file.read())
            tmp.flush()
            outzip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            with zipfile.ZipFile(tmp.name, "r") as zin, zipfile.ZipFile(outzip.name, "w") as zout:
                names = [f for f in zin.namelist() if f.lower().endswith(
                    (".jpg", ".jpeg", ".png"))]
                p = st.progress(0)
                for i, name in enumerate(names):
                    data = zin.read(name)
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    np_img = np.array(img)
                    res = model_secondary.predict(
                        np_img, conf=conf_thresh, verbose=False)
                    ann = res[0].plot()
                    zout.writestr(
                        f"detected_{os.path.basename(name)}", np_to_bytes(ann))
                    boxes = boxes_from_result(res[0])
                    for (x1, y1, x2, y2, conf, cls) in boxes:
                        cls_name = res[0].names.get(cls, str(cls))
                        sev = severity_score(
                            (x1, y1, x2, y2), cls_name, conf, np_img.shape)
                        st.session_state["detections"].append({
                            "ts": timestamp(), "class": cls_name, "conf": float(conf),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "lat": None, "lng": None, "severity": float(sev),
                            "image": os.path.basename(name)
                        })
                    p.progress((i+1)/len(names))
            st.success("Batch done")
            with open(outzip.name, "rb") as f:
                st.download_button("Download results ZIP", f,
                                   file_name="batch_results.zip")
    with colB:
        st.subheader("Reports")
        if st.button("Generate HTML report (quick)"):
            df = pd.DataFrame(st.session_state["detections"])
            if df.empty:
                st.warning("No detections to report")
            else:
                html = df.to_html(index=False)
                path = os.path.join(
                    "outputs", f"report_{int(time.time())}.html")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(
                        f"<h1>RoadVision Quick Report</h1><p>Generated: {datetime.utcnow()}</p>{html}")
                with open(path, "rb") as f:
                    st.download_button(
                        "Download report", f, file_name=os.path.basename(path), mime="text/html")
                st.success("Report ready")

# --------------------------
# TAB: AI Assistant
# --------------------------
with tabs[4]:
    st.header("AI Assistant")
    question = st.text_input(
        "Ask about detections (e.g., 'Summarize', 'Which segments urgent?')")
    if st.button("Ask") and question.strip() != "":
        df = pd.DataFrame(st.session_state["detections"])
        summary = "No detections yet."
        if not df.empty:
            top = df['class'].value_counts().head(5).to_dict()
            avg = df['severity'].mean()
            geoc = df['lat'].count() if 'lat' in df.columns else 0
            summary = f"Total: {len(df)}. Top classes: {top}. Avg severity: {avg:.1f}. Geotagged: {geoc}."
        prompt = f"You are RoadVision Assistant. Context: {summary}\nUser question: {question}\nProvide concise answer & next steps."
        reply = None
        if OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            try:
                resp = openai.ChatCompletion.create(model="gpt-4o-mini", messages=[
                                                    {"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": prompt}], max_tokens=250)
                reply = resp['choices'][0]['message']['content']
            except Exception as e:
                reply = f"(OpenAI error) {e}\nFallback: {summary}"
        else:
            if "summar" in question.lower():
                reply = "Summary: " + summary + \
                    " Next steps: collect more geotagged samples, prioritize severity>60."
            elif "urgent" in question.lower() or "repair" in question.lower():
                if not df.empty:
                    high = df[df['severity'] > 60]
                    reply = f"{len(high)} high-severity detections. Recommend field inspection."
                else:
                    reply = "No detections."
            else:
                reply = "Try: 'Summarize' or 'Which are urgent?'"
        st.markdown("**Assistant:**")
        st.write(reply)

# --------------------------
# TAB: About
# --------------------------
with tabs[5]:
    st.header("About & Deployment Tips")
    st.markdown("""
    - For mobile live camera: streamlit-webrtc + browser camera is used (front/back toggle).
    - Browser Live GPS: optional helper 'streamlit-javascript' makes it easy to request geolocation from the browser.
      Install: pip install streamlit-javascript
    - Streamlit Cloud: create runtime.txt with 'python-3.11' and use opencv-python-headless in requirements.
    - For production real-time high FPS, deploy to GPU instance and serve inference with FastAPI/ONNX.
    """)
    st.markdown("**Where to change model paths:**")
    st.code("MODEL_PATHS = {\n  'Road Markings (YOLOv11)': 'weights/road_markings_yolo11.pt',\n  'Road Defects (YOLOv11)': 'weights/road_defects_yolo11.pt'\n}")

# End of app
