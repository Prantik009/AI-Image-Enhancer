import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import torch
from super_image import EdsrModel, ImageLoader
from PIL import Image
from streamlit_image_comparison import image_comparison
import numpy as np
import time

# Page Setup
st.set_page_config(page_title="AI Image Enhancer Pro", page_icon="✨", layout="centered")

# Aggressive UI Customization (Hides Top Bar, Profile Icons, and Floating Footer)
st.markdown("""
    <style>
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    .stAppViewBlockContainer {
        padding-top: 1.5rem !important;
    }
    div[class*="viewerToolbar"], 
    div[class*="embeddedToolbar"], 
    iframe[title="Hosted with Streamlit"],
    div[data-testid="stConnectionStatus"],
    .stActionButton {
        display: none !important;
        visibility: hidden !important;
    }
    footer {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("✨ AI Super-Resolution Image Enhancer")
st.write("Upload an image, pick your scale factor, and look at the dramatic crispness boost.")

# Cache the AI model
@st.cache_resource
def load_ai_model():
    device = torch.device('cpu')
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    model = model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_ai_model()
except Exception as e:
    st.error(f"Error loading AI engine: {e}")

# NEW UPGRADE: Scale Selection Options Container
st.write("### 🎛️ Configuration Settings")
scale_selection = st.select_slider(
    "Choose your target upscale multiplier factor:",
    options=[2, 3, 4],
    value=2,
    help="2x runs on Deep Learning AI weights. 3x & 4x use advanced high-fidelity Lanczos filtering arrays."
)

# File Uploader Interface
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    width, height = image.size
    
    # Show dynamic target dimensions on the UI box
    st.write("### 📊 Image Pipeline Specifications")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Original Size", f"{width}x{height}")
    col_b.metric("Scale Factor", f"{scale_selection}x")
    col_c.metric("Output Resolution", f"{width * scale_selection}x{height * scale_selection}")

    if st.button("Enhance Image Resolution"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        # PIPELINE BRANCH 1: Premium 2x AI Upscaling Loop
        if scale_selection == 2:
            tile_size = 400 
            scale = 2
            output_image = Image.new('RGB', (width * scale, height * scale))
            
            total_tiles_x = int(np.ceil(width / tile_size))
            total_tiles_y = int(np.ceil(height / tile_size))
            total_tiles = total_tiles_x * total_tiles_y
            current_tile = 0
            
            with torch.no_grad():
                for y in range(0, height, tile_size):
                    for x in range(0, width, tile_size):
                        current_tile += 1
                        percent_complete = int((current_tile / total_tiles) * 100)
                        progress_bar.progress(percent_complete)
                        status_text.text(f"⏳ Running AI Matrix patches: {current_tile}/{total_tiles} ({percent_complete}%)")
                        
                        box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                        tile = image.crop(box)
                        
                        inputs = ImageLoader.load_image(tile).to(device)
                        preds = model(inputs)
                        
                        tile_output = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        tile_output = (tile_output * 255).clip(0, 255).astype('uint8')
                        tile_img = Image.fromarray(tile_output)
                        
                        output_image.paste(tile_img, (x * scale, y * scale))
                        
        # PIPELINE BRANCH 2: High-Fidelity 3x/4x Lanczos Filtration Array
        else:
            status_text.text(f"⏳ Running High-Fidelity {scale_selection}x Lanczos filtration array...")
            progress_bar.progress(40)
            time.sleep(0.3)
            
            new_w = width * scale_selection
            new_h = height * scale_selection
            output_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            progress_bar.progress(100)

        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        progress_bar.empty()
        status_text.success(f"✅ Success! Enhanced in {duration} seconds.")
        
        # NEW UPGRADE: Before vs After Image Comparison Slider Box
        st.write("### 🔍 Interactive Side-by-Side Comparison")
        st.write("Drag the slider handle horizontally to check the crispness gain.")
        
        # Prepare matching sizes for a clean comparison layout experience
        comparison_original = image.resize((output_image.width, output_image.height), Image.Resampling.NEAREST)
        
        image_comparison(
            img1=comparison_original,
            img2=output_image,
            label1="Before (Zoomed)",
            label2="After (Enhanced)",
            width=700
        )
        
        # Save memory buffer for browser download trigger hooks
        import io
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.write("")
        st.download_button(
            label="💾 Download Enhanced Image",
            data=img_byte_arr,
            file_name=f"enhanced_{scale_selection}x_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png"
        )
