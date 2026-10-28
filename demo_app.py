import os

import cv2
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image

from data.synthetic_generator import RoadAdverseWeatherSynthesizer
from models.generator import DeRainDeHazeGenerator
from utils.driving_eval import evaluate_autonomous_driving_readiness
from utils.metrics import calculate_psnr, calculate_ssim, tensor_to_numpy_image


st.set_page_config(
    page_title="FGAN · Road image restoration",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#211b18; --muted:#786d64; --cream:#fff6df; --paper:#fffdf8;
            --orange:#db5b1b; --orange2:#f18a22; --line:#eee5d8; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
    .stApp { background:var(--cream); }
    [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] * { color:var(--ink); }
    [data-testid="stAppViewContainer"] .hero,
    [data-testid="stAppViewContainer"] .hero * { color:#fffaf1 !important; }
    [data-testid="stAppViewContainer"] .hero p { color:#ffebd2 !important; }
    [data-testid="stAppViewContainer"] .eyebrow { color:#a63d12 !important; }
    [data-testid="stAppViewContainer"] .subtle,
    [data-testid="stAppViewContainer"] .step { color:#5e5047 !important; }
    [data-testid="stRadio"] label,
    [data-testid="stRadio"] label p,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] small,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stTextInput"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stSlider"] label,
    [data-testid="stCaptionContainer"] { color:#302720 !important; }
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] { color:#243847 !important; }
    [data-testid="stFileUploader"] button { color:#fff !important; }
    [data-testid="stHeader"] { background:transparent; }
    .block-container { max-width:1240px; padding:2rem 2rem 4rem; }
    .hero { min-height:350px; border-radius:30px; padding:30px 5vw 46px; color:#fffaf1;
      background:radial-gradient(ellipse at 85% 5%,rgba(255,194,97,.38),transparent 28%),
      radial-gradient(ellipse at 8% 100%,rgba(111,29,10,.4),transparent 37%),
      linear-gradient(115deg,#b84314 0%,#df6817 52%,#c64b12 100%);
      box-shadow:0 24px 55px rgba(144,69,22,.18); text-align:center; position:relative; overflow:hidden; }
    .hero:before,.hero:after { content:''; position:absolute; width:390px;height:390px;border:1px solid rgba(255,242,215,.16);border-radius:50%; }
    .hero:before { left:-150px;bottom:-290px; } .hero:after { right:-100px;top:-280px; }
    .brand { letter-spacing:.24em;text-transform:uppercase;font-size:.75rem;font-weight:700;opacity:.85;margin:8px 0 28px; }
    .hero h1 { color:#fffaf1;font:800 clamp(2.3rem,5vw,4.3rem)/1.04 'Manrope',sans-serif;letter-spacing:-.055em;margin:0 auto 15px;max-width:850px; }
    .hero p { color:#ffebd2;font-size:1.08rem;margin:0 auto;max-width:660px; }
    .eyebrow { color:var(--orange);font-size:.76rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase; }
    .section-title { color:#211b18 !important;font:700 clamp(1.7rem,3vw,2.5rem)/1.16 'Manrope',sans-serif;letter-spacing:-.045em;margin:.35rem 0 .45rem; }
    .subtle { color:var(--muted);font-size:1rem; }
    .intro { background:#f6e9dc;border-radius:0 0 24px 24px;padding:36px 7%;margin:0 0 36px; }
    .stButton>button { border-radius:999px;border:0;background:var(--orange);color:white;font-weight:700;padding:.68rem 1.45rem; }
    .stButton>button:hover { background:#b94717;color:white;border:0; }
    [data-testid="stFileUploader"] section { background:rgba(255,253,248,.75);border:1.5px dashed #d8b89d;border-radius:18px; }
    [data-testid="stImage"] img { border-radius:18px; }
    [data-testid="stMetric"] { background:var(--paper);border:1px solid var(--line);padding:16px 18px;border-radius:16px; }
    [data-testid="stMetricLabel"] { color:var(--muted); }
    div[data-testid="stTabs"] button { border-radius:999px; }
    .step { color:var(--muted);font-size:.88rem;margin:.2rem 0 1rem; }
    footer { visibility:hidden; }
    @media (max-width:700px) { .block-container{padding:1rem 1rem 3rem}.hero{padding:25px 6% 38px;min-height:300px}.intro{padding:25px 7%} }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_trained_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeRainDeHazeGenerator(in_channels=3, out_channels=3, features=64).to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "generator" in state:
            state = state["generator"]
        model.load_state_dict(state)
        status = f"Ready · {device.type.upper()} · trained model loaded"
    else:
        status = f"Checkpoint not found · using untrained model · {device.type.upper()}"
    model.eval()
    return model, device, status


st.markdown(
    """
    <section class="hero">
      <div class="brand">FGAN · Road vision studio</div>
      <h1>See the road<br>through the weather.</h1>
      <p>Bring a rainy or hazy frame into focus with one simple restoration preview.</p>
    </section>
    <section class="intro">
      <div class="eyebrow">A clearer view, in a few steps</div>
      <div class="section-title">A little less weather.<br>A lot more road.</div>
      <div class="subtle">Choose a sample or add your own image. FGAN restores it locally and shows the before and after side by side.</div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.expander("Model status and settings", expanded=False):
    checkpoint_file = st.text_input("Checkpoint file", "checkpoints/best_generator.pth")
    model, device, load_status = load_trained_model(checkpoint_file)
    st.caption(load_status)

# Load the default model outside the expander as well, so inference is always ready.
checkpoint_file = locals().get("checkpoint_file", "checkpoints/best_generator.pth")
model, device, load_status = load_trained_model(checkpoint_file)

st.markdown('<div class="eyebrow">01 / Pick your image</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Where should we start?</div>', unsafe_allow_html=True)
st.markdown('<div class="step">Use a sample, upload a weathered photo, or add a clean image to try a weather simulation.</div>', unsafe_allow_html=True)

mode = st.radio(
    "Image source",
    ["Try a sample", "Upload a photo", "Add rain & haze"],
    horizontal=True,
    label_visibility="collapsed",
)

clean_img = None
degraded_img = None
sample_dir = os.path.join("dataset", "test", "degraded")

if mode == "Try a sample":
    files = sorted(
        f for f in os.listdir(sample_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ) if os.path.isdir(sample_dir) else []
    if files:
        choose_col, preview_col = st.columns([1, 1.4], gap="large")
        with choose_col:
            chosen = st.selectbox("Choose a road scene", files, format_func=lambda f: f.replace("road_pair_", "Scene ").split(".")[0])
            sample = Image.open(os.path.join(sample_dir, chosen)).convert("RGB")
            clean_path = os.path.join("dataset", "test", "clean", chosen)
            if os.path.exists(clean_path):
                clean_img = Image.open(clean_path).convert("RGB")
            st.caption("A paired test image gives you a reference for comparison.")
        degraded_img = sample
        with preview_col:
            st.image(sample, caption="Selected weather affected image", use_container_width=True)
    else:
        st.info("No test samples found. Upload an image to get started.")

elif mode == "Upload a photo":
    uploaded = st.file_uploader("Drop a rainy or hazy road image here", type=["jpg", "jpeg", "png"], help="Your image stays on this machine.")
    if uploaded:
        degraded_img = Image.open(uploaded).convert("RGB")

else:
    uploaded = st.file_uploader("Add a clean road image to weather", type=["jpg", "jpeg", "png"])
    control_col, control_col2, control_col3 = st.columns(3)
    with control_col:
        beta = st.slider("Haze amount", 0.5, 2.5, 1.2, help="Higher values make the haze denser.")
    with control_col2:
        rain_density = st.slider("Rain density", 0.005, 0.05, 0.02)
    with control_col3:
        rain_angle = st.slider("Rain angle", 60, 120, 85)
    if uploaded:
        clean_img = Image.open(uploaded).convert("RGB")
        clean_np = np.array(clean_img)
        synth = RoadAdverseWeatherSynthesizer()
        hazy = synth.add_haze(clean_np, beta=beta)
        degraded_img = Image.fromarray(synth.add_rain_streaks(hazy, angle=rain_angle, density=rain_density))

if degraded_img is not None:
    st.markdown('<div class="eyebrow">02 / Your restored view</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">A clearer look at the scene.</div>', unsafe_allow_html=True)
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    input_tensor = transform(degraded_img).unsqueeze(0).to(device)
    with st.spinner("Restoring your image…"):
        with torch.inference_mode():
            restored_tensor = model(input_tensor)

    degraded_np = tensor_to_numpy_image(input_tensor[0])
    restored_np = tensor_to_numpy_image(restored_tensor[0])
    before_col, after_col = st.columns(2, gap="large")
    with before_col:
        st.caption("BEFORE · WEATHER AFFECTED")
        st.image(degraded_np, use_container_width=True)
    with after_col:
        st.caption("AFTER · FGAN RESTORATION")
        st.image(restored_np, use_container_width=True)

    if clean_img is not None:
        with st.expander("Show clean reference"):
            st.image(clean_img.resize((256, 256)), caption="Ground truth reference", use_container_width=True)

    stats = evaluate_autonomous_driving_readiness(degraded_np, restored_np)
    st.markdown("#### Scene quality")
    metric_cols = st.columns(3)
    if clean_img is not None:
        clean_np = np.array(clean_img.resize((256, 256)))
        psnr_before = calculate_psnr(degraded_np, clean_np)
        psnr_after = calculate_psnr(restored_np, clean_np)
        ssim_before = calculate_ssim(degraded_np, clean_np)
        ssim_after = calculate_ssim(restored_np, clean_np)
        metric_cols[0].metric("PSNR", f"{psnr_after:.2f} dB", f"{psnr_after - psnr_before:+.2f} dB")
        metric_cols[1].metric("SSIM", f"{ssim_after:.3f}", f"{ssim_after - ssim_before:+.3f}")
    else:
        metric_cols[0].metric("Road contrast", f"{stats['contrast_gain_pct']:+.1f}%")
        metric_cols[1].metric("Edge detail", f"{stats['edge_feature_gain_pct']:+.1f}%")
    metric_cols[2].metric("Processing", device.type.upper())

    with st.expander("Explore lane and edge detail"):
        edge_before, edge_after = st.columns(2, gap="large")
        edge_before.image(cv2.Canny(cv2.cvtColor(degraded_np, cv2.COLOR_RGB2GRAY), 50, 150), caption="Before restoration")
        edge_after.image(cv2.Canny(cv2.cvtColor(restored_np, cv2.COLOR_RGB2GRAY), 50, 150), caption="After restoration")
elif mode != "Try a sample" or os.path.isdir(sample_dir):
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.info("Choose a sample or add an image above. Your restoration preview will appear here.")

st.markdown("<div style='text-align:center;color:#88786b;padding:42px 0 0'>FGAN · Road image restoration, right on your machine</div>", unsafe_allow_html=True)
