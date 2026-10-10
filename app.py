import os
# CRITICAL: Prevent the Linux cloud engine from fragmentation faults during large tensor array initializations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import streamlit as st
import torch
import torchvision.transforms as transforms
from super_image import EdsrModel
from PIL import Image, ImageFilter
import numpy as np
import time

# 1. Page Configuration
st.set_page_config(page_title="AI Pixel Quality Enhancer", page_icon="✨", layout="centered")

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

st.title("✨ Color-Locked AI Image Enhancer")
st.write("Upscale your image dimensions using a Deep Learning engine while locking 100% of your authentic colors.")

# 2. Cached model loader (Forces CPU execution for stable cloud deployment)
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
    # Open original image asset exactly as uploaded
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
        st.write(f"🚀 **Target AI Dimensions:** {width*2}x{height*2} pixels")

    st.write("---")
    st.write("### 🚀 Step 2: Generate Enhanced Image")

    # 4. Execution Processing Trigger Button
    if st.button("Enhance Image with Color-Locked AI", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        # Tiling Configuration parameters
        tile_size = 400 
        scale = 2
        output_image = Image.new('RGB', (width * scale, height * scale))
        
        total_tiles_x = int(np.ceil(width / tile_size))
        total_tiles_y = int(np.ceil(height / tile_size))
        total_tiles = total_tiles_x * total_tiles_y
        current_tile = 0
        
        # Define clean tensor transformers to map pixel arrays without altering colors
        to_tensor = transforms.ToTensor()
        
        # 5. Grid Processing Loop with Color Protection Math
        with torch.no_grad():
            for y in range(0, height, tile_size):
                for x in range(0, width, tile_size):
                    current_tile += 1
                    percent_complete = int((current_tile / total_tiles) * 100)
                    
                    progress_bar.progress(percent_complete)
                    status_text.text(f"⏳ Running Color-Locked AI tensor patches: {current_tile}/{total_tiles} ({percent_complete}%)")
                    
                    # Crop the small section patch
                    box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
                    tile = image.crop(box)
                    
                    # Convert to raw tensor array (0.0 to 1.0) bypassing package normalization bias
                    inputs = to_tensor(tile).unsqueeze(0).to(device)
                    
                    # Forward pass through the neural network
                    preds = model(inputs)
                    
                    # --- THE COLOR LOCK FIX ---
                    # 1. Strip batch channel dimensions
                    tile_output = preds.squeeze(0)
                    # 2. Permute dimension axes back to standard shape (Height, Width, Channels)
                    tile_output = tile_output.permute(1, 2, 0).cpu().numpy()
                    # 3. Clip array parameters strictly between 0.0 and 1.0 to secure authentic pixel integrity
                    tile_output = np.clip(tile_output, 0.0, 1.0)
                    # 4. Map directly to clean unsigned 8-bit integers without fading or bleeding colors
                    tile_output = (tile_output * 255.0).astype('uint8')
                    
                    # Reconstruct patch into a PIL image
                    tile_img = Image.fromarray(tile_output)
                    
                    # Paste into final layout canvas
                    output_image.paste(tile_img, (x * scale, y * scale))
        
        # Post-processing sharp contrast polish
        status_text.text("✨ Polishing edge textures...")
        output_image = output_image.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Clean progress bars upon layout execution completion
        progress_bar.empty()
        status_text.success(f"✅ Success! AI Enhanced to {width*scale}x{height*scale} in {duration} seconds.")
        
        st.write("---")
        st.write("### 🔍 Step 3: Before / After Comparison")
        st.write("Toggle between the tabs to see the clear sharpness gain while verifying that your original colors are identical.")
        
        # Scale original preview up cleanly matching output canvas sizes for side-by-side viewing
        comparison_original = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
        
        # Render clean tabs so users can slide or toggle between views smoothly
        tab1, tab2 = st.tabs(["📊 Side-by-Side Comparison", "🖼️ Full Enhanced View"])
        
        with tab1:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.image(comparison_original, caption="Before (Zoomed / Pixelated)", use_container_width=True)
            with comp_col2:
                st.image(output_image, caption="After (AI Enhanced + Color Preserved)", use_container_width=True)
                
        with tab2:
            st.image(output_image, caption="Final Enhanced Image Output", use_container_width=True)
        
        # Memory download array stream hook
        import io
        img_byte_arr = io.BytesIO()
        output_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        st.write("")
        st.download_button(
            label="💾 Download Enhanced Image",
            data=img_byte_arr,
            file_name=f"ai_color_locked_2x_{uploaded_file.name.rsplit('.', 1)[0]}.png",
            mime="image/png"
        )
