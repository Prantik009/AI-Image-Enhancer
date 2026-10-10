import os
# CRITICAL: Prevent the Linux cloud engine from fragmentation faults during large tensor array initializations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import torch
from super_image import EdsrModel, ImageLoader
from PIL import Image
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(page_title="AI Image Enhancer", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    /* 1. Eliminate the top header area completely (Hides Github profile icon, Share, Deploy) */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
    /* 2. Target the exact layout wrapper class to clean up top page margins */
    .stAppViewBlockContainer {
        padding-top: 1.5rem !important;
    }

    /* 3. Aggressively target and hide the new "Hosted with Streamlit" floating menu container */
    div[class*="viewerToolbar"], 
    div[class*="embeddedToolbar"], 
    iframe[title="Hosted with Streamlit"],
    div[data-testid="stConnectionStatus"],
    .stActionButton {
        display: none !important;
        visibility: hidden !important;
    }

    /* 4. Catch-all for standard footer and status line containers */
    footer {visibility: hidden !important; display: none !important;}
    #MainMenu {visibility: hidden !important;}
    </style>
    """, unsafe_allow_html=True)

st.title("✨ AI Super-Resolution Image Enhancer")
st.write("Upload your photo, configure your target scale, and watch our network improve its resolution.")

# 2. Lazy-loaded model architectures to keep memory clean
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

# 3. Step 1: File Uploader Interface
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open original image asset
    image = Image.open(uploaded_file).convert('RGB')
    width, height = image.size
    
    st.write("---")
    st.write("### 📸 Step 1: Original Image Preview & Details")
    
    # Show dynamic image details layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Original Image", use_container_width=True)
    with col2:
        st.write("#### Image Details")
        st.write(f"📁 **Filename:** {uploaded_file.name}")
        st.write(f"📐 **Original Dimensions:** {width}x{height} pixels")

    st.write("---")
    st.write("### 🎛️ Step 2: Select Upscale Multiplier")
    
    # Step 2: Slider option positioned AFTER the preview details
    scale_selection = st.select_slider(
        "Select your target upscale factor:",
        options=[2, 3, 4],
        value=2,
        help="2x runs on deep learning AI weights. 3x & 4x use advanced high-fidelity Lanczos filtering arrays."
    )
    
    # Calculate new target values based on slider choice dynamically
    target_width = width * scale_selection
    target_height = height * scale_selection
    st.info(f"🎯 **Target Output Dimensions:** {target_width}x{target_height} pixels ({scale_selection}x enhancement)")

    st.write("---")
    st.write("### 🚀 Step 3: Generate Enhanced Image")

    # 4. Execution Processing Trigger Button
    if st.button("Enhance Image Resolution", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        # BRANCH A: Premium Tiled 2x AI Upscaling Execution Loop
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
                        status_text.text(f"⏳ Processing AI section {current_tile}/{total_tiles} ({percent_complete}%)")
                        
                        box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                        tile = image.crop(box)
                        
                        inputs = ImageLoader.load_image(tile).to(device)
                        preds = model(inputs)
                        
                        tile_output = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        tile_output = (tile_output * 255).clip(0, 255).astype('uint8')
                        tile_img = Image.fromarray(tile_output)
                        
                        output_image.paste(tile_img, (x * scale, y * scale))
                        
        # BRANCH B: High-Fidelity 3x/4x Lanczos Filtration Array
        else:
            status_text.text(f"⏳ Running High-Fidelity {scale_selection}x Lanczos filtering array...")
            progress_bar.progress(40)
            
            output_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            time.sleep(0.4)
            progress_bar.progress(100)
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Clean progress bars upon layout execution completion
        progress_bar.empty()
        status_text.success(f"✅ Success! Enhanced to {target_width}x{target_height} in {duration} seconds.")
        
        st.write("---")
        st.write("### 🔍 Step 4: Before / After Comparison")
        
        # Scale original preview up cleanly matching output canvas sizes for side-by-side viewing
        comparison_original = image.resize((target_width, target_height), Image.Resampling.NEAREST)
        
        # Render clean tabs so users can slide or toggle between views smoothly
        tab1, tab2 = st.tabs(["📊 Interactive Side-by-Side View", "🖼️ Full Enhanced View"])
        
        with tab1:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.image(comparison_original, caption="Before (Zoomed)", use_container_width=True)
            with comp_col2:
                st.image(output_image, caption="After (Enhanced)", use_container_width=True)
                
        with tab2:
            st.image(output_image, caption="Enhanced Resolution Output Image", use_container_width=True)
        
        # Memory download array stream hook
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
