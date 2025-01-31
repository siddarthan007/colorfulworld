import streamlit as st
import numpy as np
import cv2
from PIL import Image
import io
import time

st.set_page_config(page_title="AI Image Colorizer", layout="wide", page_icon="🎨")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2a4a7d;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    .st-bb {background-color: white;}
    .st-at {background-color: #2a4a7d;}
    footer {visibility: hidden;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    .download-button {display: flex; justify-content: center;}
    .css-1aumxhk {background-color: #ffffff;}
    </style>
""", unsafe_allow_html=True)

def load_models():
    # Model loading with error handling
    try:
        prototxt_path = 'models/colorization_deploy_v2.prototxt'
        model_path = 'models/colorization_release_v2.caffemodel'
        kernel_path = 'models/pts_in_hull.npy'
        
        net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        points = np.load(kernel_path)
        
        points = points.transpose().reshape(2, 313, 1, 1)
        net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
        net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]
        return net
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

def is_grayscale(img, threshold=5):
    # Check if image is grayscale
    if len(img.shape) == 2 or img.shape[2] == 1:
        return True
    return np.std(img[:,:,0] - img[:,:,1]) < threshold and np.std(img[:,:,1] - img[:,:,2]) < threshold

def colorize_image(net, bw_image):
    # Colorization process with error handling
    try:
        normalized = bw_image.astype("float32") / 255.0
        lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
        
        resized = cv2.resize(lab, (224, 224))
        l = cv2.split(resized)[0]
        l -= 50
        
        net.setInput(cv2.dnn.blobFromImage(l))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        return (255.0 * colorized).astype("uint8")
    except Exception as e:
        st.error(f"Colorization failed: {str(e)}")
        st.stop()

# Sidebar with upload and settings
with st.sidebar:
    st.header("Upload Settings")
    uploaded_file = st.file_uploader("Choose an image", 
                                    type=["jpg", "jpeg", "png"],
                                    help="Select a black and white photo to colorize")

# Main content area
st.title("🎨 AI Image Colorizer")
st.markdown("Transform black & white photos into color using deep learning")

# Model loading with progress
with st.spinner("Loading AI model..."):
    net = load_models()

# Upload and processing
if uploaded_file is not None:
    # Read and verify image
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if original_image is None:
            st.error("Invalid image file. Please upload a valid JPEG or PNG image.")
            st.stop()
            
        if not is_grayscale(original_image):
            st.warning("⚠️ The uploaded image appears to already contain color. For best results, use a true black and white photo.")
            
        # Processing section with visual feedback
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Preprocessing image...")
        progress_bar.progress(20)
        time.sleep(0.5)
        
        status_text.text("Colorizing with AI...")
        colorized = colorize_image(net, original_image)
        progress_bar.progress(70)
        time.sleep(0.5)
        
        # Convert to RGB for display
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        result_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
        
        # Display results
        progress_bar.progress(100)
        time.sleep(0.2)
        progress_bar.empty()
        status_text.empty()
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_rgb, 
                    caption="Original Image",
                    use_container_width=True,
                    clamp=True)
            
        with col2:
            st.image(result_rgb, 
                    caption="Colorized Result", 
                    use_container_width=True,
                    clamp=True)
            
        # Download section
        st.markdown("---")
        st.subheader("Download Result")
        
        # Convert to bytes
        pil_image = Image.fromarray(result_rgb)
        img_bytes = io.BytesIO()
        pil_image.save(img_bytes, format="JPEG", quality=95)
        
        # Centered download button
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.download_button(
                label="⬇️ Download Colorized Image",
                data=img_bytes.getvalue(),
                file_name="colorized_image.jpg",
                mime="image/jpeg",
                use_container_width=True
            )
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    # Show upload instructions
    st.markdown("""
        <div style='text-align: center; padding: 50px 20px;'>
            <h3 style='color: #4a4a4a;'>How to use:</h3>
            <ol style='text-align: left; display: inline-block;'>
                <li>Click 'Browse files' in the left sidebar</li>
                <li>Select a black & white photo (JPEG or PNG)</li>
                <li>Wait for AI processing (10-30 seconds)</li>
                <li>Download your colorized photo!</li>
            </ol>
            <p style='margin-top: 30px; color: #666;'>
                💡 Tip: High-contrast photos work best!
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>By Siddartha Nepal</p>
    </div>
""", unsafe_allow_html=True)