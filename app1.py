import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image

# --- 1. Page Configuration ---
st.set_page_config(page_title="DeepDigit AI - Ultra Lab", page_icon="🧠", layout="wide")

# --- Initialize Session State for History & Canvas Control ---
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

if 'canvas_key' not in st.session_state:
    st.session_state.canvas_key = "canvas_v1"

# --- 2. Advanced CSS (Full Original Styles Restored) ---
st.markdown(
    """
    <style>
    /* Background with MNIST Pattern */
    .stApp {
        background: linear-gradient(rgba(10, 15, 30, 0.9), rgba(10, 15, 30, 0.9)), 
                    url("https://www.cs.ryerson.ca/~aharley/vis/mnist/img/mnist.png");
        background-size: 300px;
        background-attachment: fixed;
    }

    /* Main Glassmorphism Container */
    .main-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        padding: 40px;
        border-radius: 25px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        margin-top: 10px;
    }

    /* Glowing Text Effects */
    .glow-title {
        color: #fff;
        text-align: center;
        font-size: 45px !important;
        font-weight: 800;
        text-transform: uppercase;
        text-shadow: 0 0 10px #00d4ff, 0 0 20px #00d4ff, 0 0 40px #00d4ff;
        margin-bottom: 5px;
    }
    
    .glow-text {
        color: #00d4ff !important;
        text-shadow: 0 0 5px #00d4ff;
        font-weight: bold;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px 10px 0 0;
        color: white;
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00d4ff, #0045ff);
        color: white;
        border: none;
        border-radius: 12px;
        height: 3.5em;
        font-weight: bold;
        transition: 0.3s;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    .stButton>button:hover {
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.8);
        transform: scale(1.02);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid #00d4ff;
    }
    
    /* Control Label Styling */
    .control-label {
        color: #00d4ff;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 3. Sidebar (Original Content Restored) ---
with st.sidebar:
    st.markdown("<h2 class='glow-text'>Project Info</h2>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.markdown("---")
    st.write("**Model:** CNN (Deep Learning)")
    st.write("**Dataset:** MNIST (60k Train, 10k Test)")
    st.write("**Input Size:** 28x28 Grayscale")
    st.markdown("---")
    st.info("💡 **Tip:** Draw the digit as large and centered as possible for best accuracy.")
    st.success("✅ System Online")

# --- 4. Load Model ---
@st.cache_resource
def load_digit_model():
    try:
        return tf.keras.models.load_model('digit_model.h5')
    except:
        return None

model = load_digit_model()

# --- 5. Main App Content ---
st.markdown('<h1 class="glow-title" style="margin-top:10px;">🧠 DeepDigit Ultra Predictor</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; font-size:18px;'>AI Powered Handwritten Digit Recognition using CNN</p>", unsafe_allow_html=True)

if model is None:
    st.error("❌ 'digit_model.h5' not found! Please check your directory.")
else:
    tab1, tab2, tab3, tab4 = st.tabs(["🖌️ Interactive Drawing", "📤 Upload Digit Image", "⚙️ Model Architecture", "📊 Performance History"])

    # --- TAB 1: INTERACTIVE DRAWING ---
    with tab1:
        col1, col2 = st.columns([1.2, 1])
        
        with col1:
            st.markdown("<h3 class='glow-text'>Draw Single Digit</h3>", unsafe_allow_html=True)
            st.markdown('<p class="control-label">Undo / Redo Controls Below</p>', unsafe_allow_html=True)
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=22,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=300,
                width=300,
                drawing_mode="freedraw",
                display_toolbar=True,
                key=st.session_state.canvas_key,
            )
            
            if st.button("🗑️ Hard Reset Canvas"):
                st.session_state.canvas_key = f"canvas_{np.random.randint(0, 1000)}"
                st.rerun()

        with col2:
            st.markdown("<h3 class='glow-text'>Live Analysis</h3>", unsafe_allow_html=True)
            if canvas_result.image_data is not None and np.any(canvas_result.image_data > 0):
                img = cv2.cvtColor(canvas_result.image_data.astype('uint8'), cv2.COLOR_RGBA2GRAY)
                img = cv2.resize(img, (28, 28)) / 255.0
                img = np.reshape(img, (1, 28, 28, 1)) 
                
                preds = model.predict(img)[0]
                digit = np.argmax(preds)
                conf = np.max(preds) * 100
                
                st.markdown(f"<h2>Prediction: <span style='color:#00d4ff'>{digit}</span></h2>", unsafe_allow_html=True)
                st.write(f"**Confidence:** {conf:.2f}%")

                chart_data = pd.DataFrame(preds, index=[str(i) for i in range(10)], columns=["Probability"])
                st.bar_chart(chart_data)

                if st.button("Log Drawing"):
                    st.session_state.prediction_history.append({"Digit": digit, "Confidence": f"{conf:.2f}%", "Method": "Drawn"})
                    st.rerun() # Refresh to show in history immediately
            else:
                st.info("Awaiting input on canvas...")

    # --- TAB 2: IMAGE UPLOADER ---
    with tab2:
        st.markdown("<h3 class='glow-text'>Inference from File</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a handwritten digit (MNIST Style)", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            up_col1, up_col2 = st.columns(2)
            image = Image.open(uploaded_file).convert('L')
            with up_col1:
                st.image(image, caption='Original Image', width=200)
            
            img_up = np.array(image.resize((28, 28))) / 255.0
            if np.mean(img_up) > 0.5:
                img_up = 1.0 - img_up
            img_up = np.reshape(img_up, (1, 28, 28, 1))
            
            preds_up = model.predict(img_up)[0]
            detected_digit = np.argmax(preds_up)
            detected_conf = np.max(preds_up) * 100
            
            with up_col2:
                st.markdown(f"<h2>Detected: <span style='color:#00d4ff'>{detected_digit}</span></h2>", unsafe_allow_html=True)
                st.bar_chart(pd.DataFrame(preds_up, index=[str(i) for i in range(10)], columns=["Probability"]))
                
                if st.button("Log this Prediction"):
                    st.session_state.prediction_history.append({"Digit": detected_digit, "Confidence": f"{detected_conf:.2f}%", "Method": "Uploaded"})
                    st.success("Logged!")
                    st.rerun()

    # --- TAB 3: MODEL ARCHITECTURE ---
    with tab3:
        st.markdown("<h3 class='glow-text'>How the CNN Works</h3>", unsafe_allow_html=True)
        st.write("This application is powered by a Convolutional Neural Network trained on the MNIST dataset.")
        
        col_arch1, col_arch2 = st.columns(2)
        with col_arch1:
            st.markdown("#### Image Pre-processing")
            st.write("1. **Grayscale Conversion:** Single color channel.")
            st.write("2. **Resizing:** Exactly 28x28 pixels.")
            st.write("3. **Normalization:** Pixels scaled 0-1.")
            st.write("4. **Inversion:** Matches training data style.")
            
        with col_arch2:
            st.markdown("#### Network Layers")
            st.write("""
            ✔ Conv2D Layer → Feature extraction  
            ✔ MaxPooling → Dimension reduction  
            ✔ Flatten → 1D Vector conversion  
            ✔ Dense → Final Classification  
            ✔ Softmax → Probability distribution
            """)

    # --- TAB 4: PERFORMANCE HISTORY ---
    with tab4:
        if st.session_state.prediction_history:
            st.markdown("<h3 class='glow-text' style='text-align: center;'>Recent Session Log</h3>", unsafe_allow_html=True)
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df.tail(10).iloc[::-1], use_container_width=True)
        else:
            st.info("No predictions recorded in this session yet.")

# --- 6. Persistent Session History (Visible Below Tabs) ---
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("<h3 class='glow-text' style='text-align: center; margin-top: 30px;'>📜 Quick History (Last 5)</h3>", unsafe_allow_html=True)
    history_df_bottom = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(history_df_bottom.tail(5).iloc[::-1], use_container_width=True)