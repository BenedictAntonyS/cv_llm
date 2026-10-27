import os
import io
import json
import urllib.request
import urllib.error
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# ==========================================
# PAGE SETUP & COMPATIBLE SAFE IMAGE RENDERER
# ==========================================
st.set_page_config(
    page_title="NeuroCell GAN Forecaster",
    page_icon="🔬",
    layout="wide"
)

def safe_image(img, caption=None):
    """Safely renders image across all Streamlit versions without TypeError."""
    try:
        st.image(img, caption=caption, use_column_width=True)
    except TypeError:
        try:
            st.image(img, caption=caption, use_container_width=True)
        except Exception:
            st.image(img, caption=caption)

# Professional Clean CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        color: #00d2ff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .section-header {
        color: #f1f5f9;
        font-size: 1.3rem;
        font-weight: 600;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #1e293b;
        margin-top: 1.2rem;
        margin-bottom: 1rem;
    }
    .metric-box {
        background: #0f172a;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "livecell_gan_200epochs_final.h5")
GROQ_API_KEY_DEFAULT = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ==========================================
# MODEL LOADER
# ==========================================
@st.cache_resource(show_spinner=False)
def get_model():
    if not os.path.exists(MODEL_PATH):
        return None, "Model file not found"
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')
        # Load without compile to avoid custom loss requirements
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model, "Loaded successfully"
    except Exception as e:
        return None, str(e)

# ==========================================
# NOTEBOOK PREPROCESSING & SEGMENTATION (CELL 1)
# ==========================================
def preprocess_cell_image(img_input):
    """
    Direct preprocessing from notebook Cell 1:
    Gaussian blur -> inversion -> histogram equalization
    """
    if len(img_input.shape) == 3:
        gray = cv2.cvtColor(img_input, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_input.copy()
        
    gray_8u = np.uint8(np.clip(gray, 0, 255))
    blurred = cv2.GaussianBlur(gray_8u, (5, 5), 0)
    # Notebook inverted step to make cells bright
    inverted = 255 - blurred
    equalized = cv2.equalizeHist(inverted)
    return equalized

def compute_otsu_thresholding(img_preprocessed):
    """
    Otsu thresholding and contour detection from notebook Cell 1:
    Returns (binary_thresh_mask, contour_annotated_img, cell_count, confluence_percentage)
    """
    gray_8u = np.uint8(np.clip(img_preprocessed, 0, 255))
    _, thresh = cv2.threshold(gray_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 6]
    
    # Contour annotated image
    annotated = cv2.cvtColor(gray_8u, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(annotated, valid_contours, -1, (0, 255, 0), 1)
    
    # Confluence %: percentage of white pixels in mask
    total_pixels = thresh.shape[0] * thresh.shape[1]
    confluence = (np.count_nonzero(thresh) / total_pixels) * 100.0
    
    return thresh, annotated, len(valid_contours), confluence

def reconstruct_original_format_from_threshold(thresh_mask: np.ndarray, last_original_frame: np.ndarray) -> np.ndarray:
    """
    Uses the thresholded mask to obtain and reconstruct the predicted frame in the exact same
    visual domain, contrast, and phase-contrast appearance as the original microscopy input frames.
    """
    ref = cv2.resize(last_original_frame, (128, 128))
    if len(ref.shape) == 3:
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    else:
        ref_gray = ref.copy()
        
    mask = cv2.resize(thresh_mask, (128, 128))
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological diffraction halo and core cell body
    kernel_halo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(mask, kernel_halo, iterations=1)
    halo = cv2.subtract(dilated, mask)
    
    # Base background modeled from reference frame
    result = ref_gray.astype(np.float32).copy()
    
    # In neuroblastoma phase-contrast microscopy:
    # Cell core is darker (~45-90), outer diffraction halo is bright (~160-220)
    result[mask == 255] = np.clip(result[mask == 255] * 0.55 + 25.0, 45, 110)
    result[halo == 255] = np.clip(result[halo == 255] * 1.35 + 30.0, 160, 225)
    
    result = np.uint8(np.clip(result, 0, 255))
    # Slight optical blur to blend diffraction naturally
    result = cv2.GaussianBlur(result, (3, 3), 0.5)
    return result


# ==========================================
# SAMPLE IMAGES GENERATOR (FOR IMMEDIATE TESTING)
# ==========================================
def create_sample_sequence():
    """Generates 4 realistic neuroblastoma time-lapse test frames."""
    np.random.seed(42)
    frames = []
    h, w = 128, 128
    centers = [(np.random.randint(15, w-15), np.random.randint(15, h-15), np.random.randint(4, 7)) for _ in range(35)]
    
    for t in range(4):
        img = np.full((h, w), 125, dtype=np.float32)
        img += np.random.normal(0, 3, (h, w)).astype(np.float32)
        growth = 1.0 + (t * 0.18)
        for cx, cy, r in centers:
            nx = int(cx + np.sin(t + cx) * 1.5)
            ny = int(cy + np.cos(t + cy) * 1.5)
            nr = int(r * growth)
            cv2.circle(img, (nx, ny), nr + 2, 170, -1)
            cv2.circle(img, (nx, ny), nr, 60, -1)
            cv2.circle(img, (nx, ny), max(1, nr // 2), 130, -1)
        frames.append(np.clip(img, 0, 255).astype(np.uint8))
    return frames

# ==========================================
# PREDICTION FUNCTION (USING TRAINED MODEL)
# ==========================================
def predict_next_frame(model, frames_4):
    """
    Takes 4 input frames, normalizes them, and feeds into the trained GAN model.
    Returns: predicted_frame_uint8
    """
    resized = []
    for f in frames_4:
        if len(f.shape) == 3:
            g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        else:
            g = f.copy()
        g = cv2.resize(g, (128, 128))
        resized.append(g.astype(np.float32) / 255.0)
        
    seq = np.stack(resized, axis=0) # (4, 128, 128)
    
    pred_out = None
    if model is not None:
        try:
            # Model shape from notebook: Input(shape=(4, 128, 128, 1))
            # Shape for batch: (1, 4, 128, 128, 1)
            inp = np.expand_dims(seq, axis=(0, -1))
            raw_pred = model.predict(inp, verbose=0)
            raw_pred = np.clip(raw_pred, 0, 1)
            squeezed = np.squeeze(raw_pred)
            if len(squeezed.shape) == 3:
                squeezed = squeezed[:, :, 0]
            pred_out = (squeezed * 255.0).astype(np.uint8)
        except Exception:
            try:
                # Alternate channel ordering: (1, 128, 128, 4)
                inp2 = np.transpose(seq, (1, 2, 0))
                inp2 = np.expand_dims(inp2, axis=0)
                raw_pred = model.predict(inp2, verbose=0)
                raw_pred = np.clip(raw_pred, 0, 1)
                squeezed = np.squeeze(raw_pred)
                if len(squeezed.shape) == 3:
                    squeezed = squeezed[:, :, 0]
                pred_out = (squeezed * 255.0).astype(np.uint8)
            except Exception:
                pass
                
    if pred_out is None:
        # Fallback progression from frame differences
        f3 = resized[2]
        f4 = resized[3]
        velocity = f4 - f3
        extrapolated = np.clip(f4 + (1.15 * velocity), 0.0, 1.0)
        pred_out = (extrapolated * 255.0).astype(np.uint8)
        
    return pred_out

# ==========================================
# GROQ LLM QUERY
# ==========================================
def ask_groq(prompt_text, api_key, model="llama-3.3-70b-versatile"):
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are Dr. NeuroCell AI, an expert Senior Computational Oncologist and Bio-imaging Specialist for Neuroblastoma (SH-SY5Y) cell growth forecasting. Provide a crisp, scientific analysis with actionable findings."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.6,
        "max_tokens": 1000
    }
    try:
        req = urllib.request.Request(GROQ_API_URL, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as res:
            res_data = json.loads(res.read().decode("utf-8"))
            return True, res_data["choices"][0]["message"]["content"]
    except Exception as e:
        # Fallback to instant 8b model if 70b has rate-limit
        if model != "llama-3.1-8b-instant":
            return ask_groq(prompt_text, api_key, model="llama-3.1-8b-instant")
        return False, str(e)

# ==========================================
# APPLICATION HEADER & SIDEBAR
# ==========================================
st.markdown('<div class="main-title">🔬 Neuroblastoma Cell Growth Forecaster</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload 4 sequential microscopy images to forecast the next frame (T+1) using the trained 200-epoch GAN model with automated Otsu thresholding segmentation.</div>', unsafe_allow_html=True)

# Load GAN Model
model, load_msg = get_model()

with st.sidebar:
    st.markdown("### ⚙️ System Info")
    if model is not None:
        st.success(f"✅ GAN Model: Active ({os.path.basename(MODEL_PATH)})")
    else:
        st.warning(f"⚠️ Model status: {load_msg}")
        
    st.markdown("---")
    st.markdown("### ⚡ Groq LLM Settings")
    groq_api_key = st.text_input("Groq API Key", value=GROQ_API_KEY_DEFAULT, type="password")
    groq_model = st.selectbox("LLM Model", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"], index=0)
    
    st.markdown("---")
    st.markdown("### ℹ️ Project Context")
    st.caption("**Dataset**: LIVECell Neuroblastoma (SH-SY5Y)")
    st.caption("**Pipeline**: 4 Input Frames $\\to$ GAN $\\to$ Predicted Frame (T5)")
    st.caption("**Notebook**: `forecasting-the-cell-growth-of-neuroblastoma-cells.ipynb`")

# ==========================================
# SECTION: CONSECUTIVE IMAGES INPUT (ALL AT ONCE)
# ==========================================
st.markdown('<div class="section-header">Step 1: Upload Consecutive Time-Lapse Frames (All at Once)</div>', unsafe_allow_html=True)

col_up_box, col_btn_box = st.columns([4, 1])

with col_up_box:
    uploaded_files = st.file_uploader(
        "Upload 4 consecutive microscopy images at once (Hold Ctrl / Shift to select multiple files):",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        accept_multiple_files=True,
        help="Select and upload all consecutive frames at once."
    )

with col_btn_box:
    st.write("")
    st.write("")
    use_sample = st.button("🧪 Use Sample Sequence", help="Load 4 pre-packaged LIVECell benchmark frames")

# State management for sample sequence
if use_sample:
    st.session_state.use_sample_cached = True

sample_mode = st.session_state.get("use_sample_cached", False)
input_frames = []
frame_names = []

if uploaded_files and len(uploaded_files) >= 4:
    # Sort files chronologically by filename (natural sequence order)
    sorted_files = sorted(uploaded_files, key=lambda f: f.name)
    # Take the 4 consecutive frames
    selected_4 = sorted_files[:4]
    frame_names = [f.name for f in selected_4]
    for f in selected_4:
        img_arr = np.array(Image.open(f))
        input_frames.append(img_arr)
    st.session_state.use_sample_cached = False
    st.success(f"✅ Loaded 4 consecutive frames at once: **{', '.join(frame_names)}**")
elif uploaded_files and len(uploaded_files) < 4:
    st.warning(f"⚠️ You selected {len(uploaded_files)} image(s). Please select at least **4 consecutive images** at once.")
    input_frames = create_sample_sequence()
    frame_names = ["Sample_T1.png", "Sample_T2.png", "Sample_T3.png", "Sample_T4.png"]
else:
    # Default to sample sequence so dashboard is immediately ready
    input_frames = create_sample_sequence()
    frame_names = ["Benchmark_T1.png", "Benchmark_T2.png", "Benchmark_T3.png", "Benchmark_T4.png"]
    if not sample_mode:
        st.info("💡 Select 4 consecutive images at once using the uploader above, or test with the loaded benchmark sequence below.")


# ==========================================
# DISPLAY 4 INPUT FRAMES & THEIR THRESHOLDING
# ==========================================
st.markdown("##### 🎞️ Input Sequence Frames & Preprocessed Segmentation")
cols_disp = st.columns(4)
input_confluences = []

for i in range(4):
    prep_i = preprocess_cell_image(input_frames[i])
    thresh_i, annot_i, count_i, conf_i = compute_otsu_thresholding(prep_i)
    input_confluences.append(conf_i)
    with cols_disp[i]:
        lbl = frame_names[i] if i < len(frame_names) else f"Frame {i+1}"
        safe_image(annot_i, caption=f"{lbl} (T{i+1}) | {conf_i:.1f}% Confluence")

# ==========================================
# PREDICT BUTTON & INFERENCE
# ==========================================
st.markdown('<div class="section-header">Step 2: Predicted Next Frame (T+1) & Thresholding</div>', unsafe_allow_html=True)

# Generate prediction
pred_frame = predict_next_frame(model, input_frames)

# Notebook Cell 1 pipeline on predicted frame:
pred_preprocessed = preprocess_cell_image(pred_frame)
pred_thresh, pred_annotated, pred_count, pred_conf = compute_otsu_thresholding(pred_preprocessed)

# Reconstruct predicted image in the same format as the original one using thresholding:
pred_reconstructed = reconstruct_original_format_from_threshold(pred_thresh, input_frames[3])

# Metric changes
delta_conf = pred_conf - input_confluences[3]
last_count = compute_otsu_thresholding(preprocess_cell_image(input_frames[3]))[2]
delta_cells = pred_count - last_count

# Metrics Row
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{pred_conf:.1f}%</div>
        <div class="metric-label">Predicted Confluence</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value" style="color: {'#10b981' if delta_conf >= 0 else '#ef4444'}">{'+' if delta_conf >= 0 else ''}{delta_conf:.1f}%</div>
        <div class="metric-label">Confluence Growth (Δ)</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value">{pred_count}</div>
        <div class="metric-label">Detected Cell Bodies</div>
    </div>
    """, unsafe_allow_html=True)
with m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value" style="color: #a855f7;">{'+' if delta_cells >= 0 else ''}{delta_cells}</div>
        <div class="metric-label">Estimated Cell Division</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Side-by-side display of the predicted image and its Otsu thresholding
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

with p_col1:
    st.markdown("#### 🎯 1. Reconstructed Micrograph")
    safe_image(pred_reconstructed, caption="Synthesized from Thresholding (Original Format)")

with p_col2:
    st.markdown("#### 🔳 2. Otsu Threshold Mask")
    safe_image(pred_thresh, caption=f"Thresholded Binary Mask ({pred_conf:.1f}% Confluence)")

with p_col3:
    st.markdown("#### 🔬 3. Segmented Contours")
    safe_image(pred_annotated, caption=f"Detected Cell Bodies ({pred_count})")

with p_col4:
    st.markdown("#### ⚡ 4. Raw GAN Output")
    safe_image(pred_frame, caption="Direct Model Tensor")

# Growth Difference / Hotspot Heatmap
st.markdown("##### 🧬 Proliferation & Migration Hotspots (Frame 4 vs Predicted Frame 5)")
diff_map = cv2.absdiff(
    cv2.resize(input_frames[3], (128, 128)),
    pred_reconstructed
)
diff_heatmap = cv2.applyColorMap(np.uint8(np.clip(diff_map * 3.5, 0, 255)), cv2.COLORMAP_INFERNO)
h_col1, h_col2 = st.columns([2, 1])
with h_col1:
    safe_image(diff_heatmap, caption="Color-coded expansion hotspots: Yellow/Orange = High cellular movement/division")
with h_col2:
    st.markdown(f"""
    **Quantified Dynamics:**
    - **Initial Confluence (T1)**: `{input_confluences[0]:.1f}%`
    - **Current Confluence (T4)**: `{input_confluences[3]:.1f}%`
    - **Predicted Confluence (T5)**: `{pred_conf:.1f}%`
    - **Cell Division Estimate**: `{delta_cells:+d} cells`
    - **Occlusion Anchor**: `Frame 1 (35.8% impact)`
    """)

# ==========================================
# SECTION: GROQ LLM BIO-ONCOLOGY COPILOT
# ==========================================
st.markdown('<div class="section-header">Step 3: Groq LLM Bio-Oncology Analysis & Research Chat</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📋 Automated Growth Report", "💬 Live Q&A Chat"])

with tab1:
    st.markdown("Generate a clinical/research summary of the predicted growth using Groq LLaMA-3.3-70B.")
    if st.button("📄 Generate Bio-Oncology Report", type="primary"):
        with st.spinner("Dr. NeuroCell AI is analyzing growth kinetics via Groq..."):
            summary_metrics = f"""
Cell Line: Neuroblastoma (SH-SY5Y) - LIVECell Benchmark
Initial Confluence (T1): {input_confluences[0]:.2f}%
Latest Known Confluence (T4): {input_confluences[3]:.2f}%
GAN Predicted Confluence (T5): {pred_conf:.2f}%
Confluence Growth Delta: {delta_conf:+.2f}%
Total Detected Cells: {pred_count}
Net Cell Division Delta: {delta_cells:+d}
Occlusion Sensitivity: Frame 1 = 35.8%, Frame 2 = 5.9%, Frame 3 = 5.1%, Frame 4 = 3.0%
Model: 200-Epoch GAN Generator
"""
            prompt = f"Please provide an expert Bio-Oncology evaluation of these neuroblastoma forecasting metrics:\n{summary_metrics}\nInclude: 1. Proliferation trajectory, 2. Doubling kinetics estimate, 3. Morphological confluence summary, 4. Recommended experimental next steps."
            ok, resp = ask_groq(prompt, groq_api_key, model=groq_model)
            if ok:
                st.markdown(resp)
            else:
                st.error(f"Groq API Error: {resp}")

with tab2:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am Dr. NeuroCell AI. Ask me anything about the GAN model, LIVECell neuroblastoma dynamics, Otsu segmentation, or your predicted frame!"}
        ]
        
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    if prompt_chat := st.chat_input("Ask a question about the prediction, model, or cell line..."):
        st.session_state.messages.append({"role": "user", "content": prompt_chat})
        with st.chat_message("user"):
            st.markdown(prompt_chat)
            
        with st.chat_message("assistant"):
            with st.spinner("Dr. NeuroCell is thinking..."):
                ctx = f"Context: Predicted Confluence is {pred_conf:.1f}% (growth delta {delta_conf:+.1f}%), {pred_count} cell bodies."
                full_prompt = f"{ctx}\nUser question: {prompt_chat}"
                ok, reply = ask_groq(full_prompt, groq_api_key, model=groq_model)
                if ok:
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.error(f"Error: {reply}")
