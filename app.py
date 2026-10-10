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

# Place this right under st.set_page_config(...)
# Replace your previous st.markdown CSS block with this updated one:
st.markdown("""
    <style>
    /* 1. Hide the entire header bar container (removes Github icon, Share, Deploy, etc.) */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* 2. Hide the top status menu actions */
    #MainMenu {visibility: hidden;}
    .stAppDeployButton {display:none !important;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    
    /* 3. Aggressively remove the bottom owner toolbars and viewer badges */
    footer {visibility: hidden !important;}
    viewerBadge {display: none !important;}
    [data-testid="stManageAppButton"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    
    /* 4. Remove padding from the top so the page looks balanced and premium */
    .stAppViewBlockContainer {
        padding-top: 2rem !important;
    }
    </style>
    """, unsafe_allow_html=True)



st.title("✨ AI Super-Resolution Image Enhancer")
st.write("Upload a blurry image and let our Deep Learning model upscale it by 2x section by section.")

# 2. Cache the model loading so it doesn't reload on every button click
@st.cache_resource
def load_ai_model():
    # Streamlit Cloud runs on standard CPU backends. Force stable allocation here.
    device = torch.device('cpu')
    model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
    model = model.to(device)
    model.eval()
    return model, device

# Initialize model safely
try:
    model, device = load_ai_model()
except Exception as e:
    st.error(f"Error loading AI engine: {e}")

# 3. File Uploader Interface
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the original image
    image = Image.open(uploaded_file).convert('RGB')
    width, height = image.size
    
    # Show image metrics side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image Preview", use_container_width=True)
    with col2:
        st.write("### Image Details")
        st.write(f"📁 **Filename:** {uploaded_file.name}")
        st.write(f"📐 **Original Dimensions:** {width}x{height} pixels")
        st.write(f"🚀 **Target Dimensions:** {width*2}x{height*2} pixels")

    # 4. Execution Button
    if st.button("Enhance Image Resolution"):
        # Setup real-time progress visuals
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Tiling Setup
        tile_size = 400 
        scale = 2
        output_image = Image.new('RGB', (width * scale, height * scale))
        
        # Calculate total tiles for precise progress calculation
        total_tiles_x = int(np.ceil(width / tile_size))
        total_tiles_y = int(np.ceil(height / tile_size))
        total_tiles = total_tiles_x * total_tiles_y
        current_tile = 0
        
        status_text.text("⚙️ Processing image grid patches...")
        start_time = time.time()
        
        # 5. Grid Processing Loop with Progress Reporting
        with torch.no_grad():
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    # Track patch progress
                    current_tile += 1
                    percent_complete = int((current_tile / total_tiles) * 100)
                    
                    # Update the live progress bar and status text on screen
                    progress_bar.progress(percent_complete)
                    status_text.text(f"⏳ Processing section {current_tile}/{total_tiles} ({percent_complete}%)")
                    
                    # Crop and process patch
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    tile = image.crop(box)
                    
                    inputs = ImageLoader.load_image(tile).to(device)
                    preds = model(inputs)
                    
                    # Reconstruct tensor patch back to PIL image array
                    tile_output = preds.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    tile_output = (tile_output * 255).clip(0, 255).astype('uint8')
                    tile_img = Image.fromarray(tile_output)
                    
                    # Paste into final layout canvas
                    output_image.paste(tile_img, (x * scale, y * scale))
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Clean up progress bar visuals on completion
        progress_bar.empty()
        status_text.success(f"✅ Success! Enhanced in {duration} seconds.")
        
        # Display the result
        st.image(output_image, caption="Enhanced 2x High-Res Output", use_container_width=True)
        
        # 6. Save image to memory loop for immediate local download browser hook
        import io
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create download trigger link asset
        st.download_button(
            label="💾 Download Enhanced Image",
            data=img_byte_arr,
            file_name=f"enhanced_2x_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png"
        )
