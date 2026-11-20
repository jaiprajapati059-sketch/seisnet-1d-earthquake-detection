"""
SeisNet-1D: Interactive Seismic Waveform Analyzer & EEW Simulator Dashboard
Streamlit Application for Real-Time Earthquake Detection & Phase Picking
Features:
- Preloaded Test Split Benchmark Inspection
- Custom Waveform Upload (.npy / .csv)
- Live ObsPy SeedLink Streaming (GEOFON, IRIS DMC, custom servers) with continuous circular ring buffer
"""

import os
import sys
import time
import datetime
from pathlib import Path
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import torch

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import preprocess_stream
from src.model import SeisNet1D, get_model_summary
from src.evaluate import detect_peaks_1d
from src.dataset import SteadDataset, split_catalog_by_group, get_dataloaders
from src.generate_demo_chunk import generate_demo_dataset
from src.live_stream import SeismicRingBuffer, ObsPySeedLinkStreamer, run_live_inference


st.set_page_config(
    page_title="SeisNet-1D | Earthquake Detection & P/S Picking",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for seismology theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .eew-alert-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 8px;
        padding: 16px;
        color: #856404;
        margin-top: 15px;
        margin-bottom: 15px;
    }
    .live-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    config_path = ROOT_DIR / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


@st.cache_resource
def load_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = ROOT_DIR / config["paths"]["best_model"]
    
    model = SeisNet1D(
        in_channels=config["model"]["in_channels"],
        encoder_channels=config["model"]["encoder_channels"],
        bottleneck_channels=config["model"]["bottleneck_channels"],
        bottleneck_dilations=config["model"]["bottleneck_dilations"],
        dropout=config["model"]["dropout"],
        phase_classes=config["model"]["phase_classes"]
    ).to(device)

    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        st.warning(f"Trained checkpoint not found at {model_path}. Using uninitialized model.")
    model.eval()
    return model, device


@st.cache_data
def load_dataset_samples(config, mode="demo"):
    """Loads metadata catalog and traces for testing."""
    hdf5_path = ROOT_DIR / config["paths"]["demo_hdf5"]
    csv_path = ROOT_DIR / config["paths"]["demo_csv"]
    
    if not hdf5_path.exists() or not csv_path.exists():
        generate_demo_dataset(str(hdf5_path), str(csv_path), n_earthquakes=800, n_noise=400, seed=42)

    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = split_catalog_by_group(df, group_col="source_id", seed=42)
    test_ds = SteadDataset(test_df, str(hdf5_path), cache_waveforms=True)
    return test_ds, test_df


def init_session_state():
    """Initializes persistent live streaming state."""
    if "ring_buffer" not in st.session_state:
        st.session_state["ring_buffer"] = SeismicRingBuffer(max_samples=6000, sampling_rate=100.0)
    if "streamer" not in st.session_state:
        st.session_state["streamer"] = None
    if "streaming_active" not in st.session_state:
        st.session_state["streaming_active"] = False
    if "last_alert_time" not in st.session_state:
        st.session_state["last_alert_time"] = 0.0


def main():
    init_session_state()
    config = load_config()
    model, device = load_model(config)

    st.markdown('<div class="main-header">🌋 SeisNet-1D: Deep Learning Earthquake Detection & Phase Picking</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Multi-Task 1D Residual U-Net processing 3-component seismic waveforms (6,000 samples @ 100 Hz) on STEAD & Live SeedLink</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("⚙️ Configuration & Parameters")
    data_source = st.sidebar.radio(
        "Data Input Source",
        ["Preloaded Test Split", "Upload Custom Waveform (NPY/CSV)", "Live SeedLink Stream"]
    )

    det_thresh = st.sidebar.slider("Detection Threshold P(EQ)", min_value=0.10, max_value=0.90, value=0.50, step=0.05)
    pick_thresh = st.sidebar.slider("Phase Picking Threshold", min_value=0.10, max_value=0.80, value=0.30, step=0.05)
    peak_dist = st.sidebar.slider("Peak Min Distance (samples)", min_value=20, max_value=100, value=50, step=5)
    vel_factor = st.sidebar.slider("Assumed Apparent Velocity (km/s)", min_value=6.0, max_value=10.0, value=8.0, step=0.2)

    with st.sidebar.expander("ℹ️ Model Architecture & Info"):
        summary = get_model_summary(model)
        st.write(f"**Architecture**: {summary['model_name']}")
        st.write(f"**Parameters**: {summary['trainable_parameters']:,}")
        st.write(f"**Size**: {summary['model_size_mb']} MB")
        st.write(f"**Device**: {device.type.upper()}")

    # =========================================================================
    # Mode 1: Preloaded Test Split
    # =========================================================================
    if data_source == "Preloaded Test Split":
        test_ds, test_df = load_dataset_samples(config)

        st.sidebar.subheader("Trace Selection")
        category_filter = st.sidebar.selectbox("Filter Category", ["All", "Earthquake (High SNR)", "Earthquake (Low SNR)", "Ambient Noise"])

        if category_filter == "Earthquake (High SNR)":
            candidate_df = test_df[(test_df["trace_category"] == "earthquake_local") & (test_df["snr_db"] >= 20.0)]
        elif category_filter == "Earthquake (Low SNR)":
            candidate_df = test_df[(test_df["trace_category"] == "earthquake_local") & (test_df["snr_db"] < 15.0)]
        elif category_filter == "Ambient Noise":
            candidate_df = test_df[test_df["trace_category"] == "noise"]
        else:
            candidate_df = test_df

        trace_options = candidate_df["trace_name"].tolist()
        if not trace_options:
            trace_options = test_df["trace_name"].tolist()

        selected_trace = st.sidebar.selectbox("Select Waveform Trace", trace_options)
        trace_idx = test_df[test_df["trace_name"] == selected_trace].index[0]

        item = test_ds[trace_idx]
        raw_waveform = item["waveform"].numpy()
        sample_meta = {
            "trace_name": item["trace_name"],
            "is_earthquake_gt": bool(item["is_earthquake"].item() > 0.5),
            "p_gt_sec": (item["p_arrival_sample"] / 100.0) if item["p_arrival_sample"] > 0 else None,
            "s_gt_sec": (item["s_arrival_sample"] / 100.0) if item["s_arrival_sample"] > 0 else None,
            "snr_db": item["snr_db"],
            "dist_km_gt": item["distance_km"]
        }

        # Run Standard Inference
        _render_static_inference(raw_waveform, sample_meta, model, device, det_thresh, pick_thresh, peak_dist, vel_factor)

    # =========================================================================
    # Mode 2: Custom Upload (NPY / CSV)
    # =========================================================================
    elif data_source == "Upload Custom Waveform (NPY/CSV)":
        st.sidebar.subheader("Upload Waveform")
        uploaded_file = st.sidebar.file_uploader("Upload 3-component array (.npy or .csv)", type=["npy", "csv"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                data = np.load(uploaded_file)
            else:
                data = pd.read_csv(uploaded_file).values
            data = np.asarray(data, dtype=np.float32)
            if data.ndim == 2 and data.shape[0] != 3 and data.shape[1] == 3:
                data = data.T
            raw_waveform = preprocess_stream(data, fs=100.0)
            sample_meta = {
                "trace_name": uploaded_file.name,
                "is_earthquake_gt": None,
                "p_gt_sec": None,
                "s_gt_sec": None,
                "snr_db": None,
                "dist_km_gt": None
            }
            _render_static_inference(raw_waveform, sample_meta, model, device, det_thresh, pick_thresh, peak_dist, vel_factor)
        else:
            st.info("Please upload a 3-component seismic array of shape (3, 6000) or (6000, 3).")

    # =========================================================================
    # Mode 3: Live ObsPy SeedLink Stream
    # =========================================================================
    else:
        _render_live_seedlink_stream(config, model, device, det_thresh, pick_thresh, peak_dist, vel_factor)


def _render_static_inference(raw_waveform, sample_meta, model, device, det_thresh, pick_thresh, peak_dist, vel_factor):
    """Executes and renders inference on static/preloaded traces."""
    tensor_in = torch.from_numpy(raw_waveform).float().unsqueeze(0).to(device)
    with torch.no_grad():
        det_logit, phase_logits = model(tensor_in)
        det_prob = float(torch.sigmoid(det_logit)[0].cpu().numpy())
        p_prob_curve = torch.sigmoid(phase_logits[0, 0, :]).cpu().numpy()
        s_prob_curve = torch.sigmoid(phase_logits[0, 1, :]).cpu().numpy()

    is_detected = (det_prob >= det_thresh)
    p_peak_samp, p_conf = detect_peaks_1d(p_prob_curve, threshold=pick_thresh, distance=peak_dist)
    s_peak_samp, s_conf = detect_peaks_1d(s_prob_curve, threshold=pick_thresh, distance=peak_dist)

    p_pred_sec = (p_peak_samp / 100.0) if p_peak_samp is not None else None
    s_pred_sec = (s_peak_samp / 100.0) if s_peak_samp is not None else None

    sp_lag_sec = None
    est_distance_km = None
    eew_lead_sec = None

    if p_pred_sec is not None and s_pred_sec is not None and s_pred_sec > p_pred_sec:
        sp_lag_sec = s_pred_sec - p_pred_sec
        est_distance_km = sp_lag_sec * vel_factor
        eew_lead_sec = sp_lag_sec

    # Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(label="Event Classification", value="EARTHQUAKE" if is_detected else "NOISE", delta=f"P(EQ) = {det_prob:.1%}")
    with col2:
        st.metric(label="P-Wave Arrival (t_P)", value=f"{p_pred_sec:.2f} s" if p_pred_sec else "Not Detected", delta=f"Conf: {p_conf:.1%}" if p_pred_sec else None)
    with col3:
        st.metric(label="S-Wave Arrival (t_S)", value=f"{s_pred_sec:.2f} s" if s_pred_sec else "Not Detected", delta=f"Conf: {s_conf:.1%}" if s_pred_sec else None)
    with col4:
        st.metric(label="S - P Lag Time (Δt)", value=f"{sp_lag_sec:.2f} s" if sp_lag_sec else "N/A", delta="Shear Delay" if sp_lag_sec else None)
    with col5:
        st.metric(label="Epicentral Distance", value=f"{est_distance_km:.1f} km" if est_distance_km else "N/A", delta=f"GT: {sample_meta['dist_km_gt']:.1f} km" if sample_meta.get('dist_km_gt') else None)

    # EEW Banner
    if is_detected and eew_lead_sec is not None and eew_lead_sec > 0:
        st.markdown(f"""
        <div class="eew-alert-box">
            <h4>🚨 Earthquake Early Warning (EEW) Alert Triggered</h4>
            <p><strong>Primary compressional P-wave verified at t = {p_pred_sec:.2f} s.</strong><br>
            Estimated epicentral distance: <strong>{est_distance_km:.1f} km</strong>.<br>
            Simulated EEW Warning Lead Time before destructive S-wave shear motion: <strong>{eew_lead_sec:.2f} seconds</strong>.<br>
            <em>Automated actions: Trigger high-speed transit braking, shut off gas main lines, initiate emergency elevator halting.</em></p>
        </div>
        """, unsafe_allow_html=True)
    elif not is_detected:
        st.info("Trace classified as Ambient Seismic Noise. No EEW action required.")

    _plot_waveform_and_probs(raw_waveform, p_prob_curve, s_prob_curve, pick_thresh, sample_meta, p_pred_sec, s_pred_sec)


def _render_live_seedlink_stream(config, model, device, det_thresh, pick_thresh, peak_dist, vel_factor):
    """Handles SeedLink live streaming configuration, controls, and real-time inference."""
    st.sidebar.subheader("📡 SeedLink Stream Controls")

    preset = st.sidebar.selectbox(
        "Server Presets",
        [
            "GEOFON (geofon.gfz.de:18000) - Station GRFO",
            "IRIS DMC (rtserve.iris.washington.edu:18000) - Station ANMO",
            "Custom SeedLink Server..."
        ]
    )

    if "GEOFON" in preset:
        default_srv = "geofon.gfz.de:18000"
        default_net = "IU"
        default_sta = "GRFO"
        default_chan = "BH?"
    elif "IRIS" in preset:
        default_srv = "rtserve.iris.washington.edu:18000"
        default_net = "IU"
        default_sta = "ANMO"
        default_chan = "BH?"
    else:
        default_srv = config.get("seedlink", {}).get("default_server", "geofon.gfz.de:18000")
        default_net = config.get("seedlink", {}).get("default_network", "IU")
        default_sta = config.get("seedlink", {}).get("default_station", "GRFO")
        default_chan = config.get("seedlink", {}).get("default_channels", "BH?")

    server_url = st.sidebar.text_input("Server Host:Port", value=default_srv)
    network = st.sidebar.text_input("Network Code", value=default_net)
    station = st.sidebar.text_input("Station Code", value=default_sta)
    channel_selector = st.sidebar.text_input("Channel Selector", value=default_chan)

    col_btn1, col_btn2 = st.sidebar.columns(2)
    is_active = st.session_state.get("streaming_active", False)

    if col_btn1.button("▶️ Start Stream", disabled=is_active, use_container_width=True):
        if st.session_state.get("streamer") is not None:
            st.session_state["streamer"].stop_background()
        rb = st.session_state.get("ring_buffer", SeismicRingBuffer())
        rb.clear()
        st.session_state["ring_buffer"] = rb

        streamer = ObsPySeedLinkStreamer(
            server_url=server_url,
            network=network,
            station=station,
            channel_selector=channel_selector,
            ring_buffer=rb,
            autoconnect=False
        )
        streamer.start_background()
        st.session_state["streamer"] = streamer
        st.session_state["streaming_active"] = True
        st.rerun()

    if col_btn2.button("⏹️ Stop Stream", disabled=not is_active, use_container_width=True):
        if st.session_state.get("streamer") is not None:
            st.session_state["streamer"].stop_background()
        st.session_state["streaming_active"] = False
        st.rerun()

    auto_refresh = st.sidebar.checkbox("Auto-refresh Display", value=True)
    refresh_interval = st.sidebar.slider("Refresh Rate (seconds)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    if st.sidebar.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

    # Main Status Display
    streamer: Optional[ObsPySeedLinkStreamer] = st.session_state.get("streamer")
    ring_buffer: SeismicRingBuffer = st.session_state["ring_buffer"]
    snapshot, fill_sec, last_ts = ring_buffer.get_snapshot()
    packets = streamer.packets_received if streamer else 0
    is_running = streamer.is_running if streamer else False
    last_err = streamer.last_error if streamer else None

    # Status Banner
    if is_active and is_running:
        st.markdown(f"""
        <div style="background-color: #d4edda; border-left: 5px solid #28a745; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
            <strong style="color: #155724;">🟢 CONNECTED & STREAMING LIVE</strong>: 
            <code>{network}.{station}.{channel_selector}</code> from <code>{server_url}</code>
        </div>
        """, unsafe_allow_html=True)
    elif is_active and not is_running:
        err_msg = last_err or "Connection closed or reconnecting..."
        st.markdown(f"""
        <div style="background-color: #f8d7da; border-left: 5px solid #dc3545; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
            <strong style="color: #721c24;">⚠️ STREAMER STATUS</strong>: {err_msg}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #e2e3e5; border-left: 5px solid #6c757d; padding: 12px 18px; border-radius: 6px; margin-bottom: 15px;">
            <strong style="color: #383d41;">⚪ STREAMER IDLE</strong>: Click <strong>"▶️ Start Stream"</strong> in the sidebar to begin receiving live seismic waveform packets.
        </div>
        """, unsafe_allow_html=True)

    # Telemetry Row
    ts_str = datetime.datetime.fromtimestamp(last_ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC") if last_ts else "No packets received yet"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connection Status", "ACTIVE" if (is_active and is_running) else "IDLE")
    c2.metric("Packets Received", f"{packets:,}")
    c3.metric("Ring Buffer Fill", f"{fill_sec:.1f} / 60.0 s")
    c4.metric("Latest Packet Time", ts_str)

    st.progress(min(1.0, fill_sec / 60.0))

    # Live Inference Execution
    if fill_sec < 5.0:
        if is_active:
            st.info(f"⏳ Accumulating waveform packets in 60s ring buffer ({fill_sec:.1f}s / 60.0s). Real-time inference activates at ≥ 5.0 seconds.")
        else:
            st.write("Start the stream to inspect incoming real-time seismic waves.")
        return

    # Run Real-Time Inference
    inference_cfg = {
        "waveform": {"sampling_rate": 100},
        "preprocessing": config.get("preprocessing", {"f_min": 1.0, "f_max": 45.0, "filter_order": 4}),
        "evaluation": {
            "detection_threshold": det_thresh,
            "pick_threshold": pick_thresh,
            "peak_distance": peak_dist,
            "velocity_estimate_km_s": vel_factor
        }
    }

    res = run_live_inference(
        snapshot_3x6000=snapshot,
        model=model,
        device=device,
        config=inference_cfg,
        last_alert_time=st.session_state.get("last_alert_time", 0.0),
        alert_cooldown_sec=config.get("seedlink", {}).get("alert_cooldown_sec", 15.0)
    )
    st.session_state["last_alert_time"] = res["last_alert_time"]

    # Inference Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric(
            label="Live Event Classification",
            value="EARTHQUAKE" if res["is_earthquake"] else "AMBIENT NOISE",
            delta=f"P(EQ) = {res['det_prob']:.1%}"
        )
    with col2:
        st.metric(
            label="P-Wave Arrival (t_P)",
            value=f"{res['p_arrival_sec']:.2f} s" if res["p_arrival_sec"] else "None",
            delta=f"Conf: {res['p_conf']:.1%}" if res["p_arrival_sec"] else None
        )
    with col3:
        st.metric(
            label="S-Wave Arrival (t_S)",
            value=f"{res['s_arrival_sec']:.2f} s" if res["s_arrival_sec"] else "None",
            delta=f"Conf: {res['s_conf']:.1%}" if res["s_arrival_sec"] else None
        )
    with col4:
        st.metric(
            label="S - P Lag (Δt)",
            value=f"{res['s_p_lag_sec']:.2f} s" if res["s_p_lag_sec"] else "N/A"
        )
    with col5:
        st.metric(
            label="Est. Distance",
            value=f"{res['est_distance_km']:.1f} km" if res["est_distance_km"] else "N/A"
        )

    # Live EEW Alert Banner (with Cooldown)
    if res["alert_triggered"]:
        st.markdown(f"""
        <div class="eew-alert-box">
            <h3>🚨 REAL-TIME EARTHQUAKE ALERT TRIGGERED</h3>
            <p><strong>P-wave arrival verified at t = {res['p_arrival_sec']:.2f} s</strong> (Confidence: {res['p_conf']:.1%}).<br>
            Estimated Epicentral Distance: <strong>{res['est_distance_km']:.1f} km</strong>.<br>
            Simulated EEW Lead Time before destructive S-waves: <strong>{res['eew_lead_sec']:.2f} seconds</strong>.<br>
            <em>Alert cooldown active for 15 seconds to prevent alert storms.</em></p>
        </div>
        """, unsafe_allow_html=True)

    # Plot Live Waveform and Phase Probability Curves
    sample_meta = {
        "trace_name": f"{network}.{station} (Live SeedLink)",
        "snr_db": None
    }
    _plot_waveform_and_probs(
        res["preprocessed_waveform"],
        res["p_prob_curve"],
        res["s_prob_curve"],
        pick_thresh,
        sample_meta,
        res["p_arrival_sec"],
        res["s_arrival_sec"],
        fill_sec=fill_sec
    )

    # Auto-refresh loop
    if is_active and auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def _plot_waveform_and_probs(raw_waveform, p_prob_curve, s_prob_curve, pick_thresh, sample_meta, p_pred_sec, s_pred_sec, fill_sec=None):
    """Renders 3-component seismic seismograms alongside continuous P/S probability curves."""
    st.subheader("📊 3-Component Seismic Seismograms & Phase Arrival Probabilities")

    time_axis = np.arange(6000) / 100.0
    fig, (ax_wf, ax_prob) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]}, dpi=150
    )

    offset = 3.5
    ax_wf.plot(time_axis, raw_waveform[0] + offset, label="E (East-West)", color="#1f77b4", lw=0.9, alpha=0.85)
    ax_wf.plot(time_axis, raw_waveform[1],          label="N (North-South)", color="#2ca02c", lw=0.9, alpha=0.85)
    ax_wf.plot(time_axis, raw_waveform[2] - offset, label="Z (Vertical)", color="#6a3d9a", lw=0.9, alpha=0.85)

    # Highlight filled vs unfilled region in live mode
    if fill_sec is not None and fill_sec < 59.9:
        ax_wf.axvspan(0, 60.0 - fill_sec, color="gray", alpha=0.15, label="Buffered History")

    # Ground Truth lines if present
    if sample_meta.get("p_gt_sec"):
        ax_wf.axvline(sample_meta["p_gt_sec"], color="#d62728", linestyle="--", lw=1.5, label=f"GT P ({sample_meta['p_gt_sec']:.2f}s)")
    if sample_meta.get("s_gt_sec"):
        ax_wf.axvline(sample_meta["s_gt_sec"], color="#ff7f0e", linestyle="--", lw=1.5, label=f"GT S ({sample_meta['s_gt_sec']:.2f}s)")

    # Predicted Pick lines
    if p_pred_sec:
        ax_wf.axvline(p_pred_sec, color="#e41a1c", linestyle="-", lw=2.2, label=f"Pred P ({p_pred_sec:.2f}s)")
    if s_pred_sec:
        ax_wf.axvline(s_pred_sec, color="#377eb8", linestyle="-", lw=2.2, label=f"Pred S ({s_pred_sec:.2f}s)")

    ax_wf.set_ylabel("Standardized Velocity Amplitude", fontsize=10)
    ax_wf.set_title(f"Waveform: {sample_meta.get('trace_name', 'Input')} | Preprocessed (1-45 Hz Bandpass, Detrended)", fontsize=11, fontweight="bold")
    ax_wf.legend(loc="upper right", ncol=4, fontsize=8)
    ax_wf.grid(True, linestyle="--", alpha=0.5)

    # Probability curves
    ax_prob.plot(time_axis, p_prob_curve, label="P-Wave Probability", color="#e41a1c", lw=1.8)
    ax_prob.plot(time_axis, s_prob_curve, label="S-Wave Probability", color="#377eb8", lw=1.8)
    ax_prob.axhline(pick_thresh, color="gray", linestyle=":", label=f"Pick Threshold ({pick_thresh:.2f})")
    ax_prob.set_ylabel("Probability", fontsize=10)
    ax_prob.set_xlabel("Time (seconds)", fontsize=11)
    ax_prob.set_xlim(0, 60)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc="upper right", ncol=3, fontsize=8)
    ax_prob.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
