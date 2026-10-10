import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import time

# 1. Page Configuration & Professional UI Layout
st.set_page_config(page_title="Pixel Quality Enhancer", page_icon="✨", layout="centered")

st.markdown("""
    <style>
    /* Eliminate the top developer header completely */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
        visibility: hidden !important;
    }
    
    /* Clean up top page margins */
    .stAppViewBlockContainer {
        padding-top: 1.5rem !important;
    }

    /* Aggressively target and hide floating menu containers and footers */
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

st.title("✨ High-Fidelity Pixel Quality Enhancer")
st.write("Upscale your image dimensions smoothly while preserving 100% of your original colors and tones.")

# 2. Step 1: File Uploader Interface
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open original image asset exactly as uploaded
    image = Image.open(uploaded_file)
    width, height = image.size
    
    st.write("---")
    st.write("### 📸 Step 1: Original Image Preview & Details")
    
    # Show dynamic image details layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image Preview", use_container_width=True)
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
        help="Higher multipliers smoothly increase pixel dimensions using high-fidelity mathematical arrays."
    )
    
    # Calculate new target values based on slider choice dynamically
    target_width = width * scale_selection
    target_height = height * scale_selection
    st.info(f"🎯 **Target Output Dimensions:** {target_width}x{target_height} pixels ({scale_selection}x enhancement)")

    st.write("---")
    st.write("### 🚀 Step 3: Enhance Image Resolution")

    # 3. Execution Processing Trigger Button
    if st.button("Enhance Pixel Quality", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        start_time = time.time()
        
        status_text.text(f"⏳ Processing high-fidelity {scale_selection}x pixel layout...")
        progress_bar.progress(30)
        time.sleep(0.2)
        
        # --- THE FIX: PURE PIXEL UPSCALING WITHOUT COLOR DISTORTION ---
        # Resampling with LANCZOS smoothly fills the new pixel gaps without pixelation or color shifting
        output_image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        progress_bar.progress(70)
        
        status_text.text("✨ Refining fine details and edge sharpness...")
        # Apply a precise unsharp mask to add clean micro-contrast to edges without touching color data
        output_image = output_image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=130, threshold=2))
        progress_bar.progress(100)
        
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Clean progress bars upon layout execution completion
        progress_bar.empty()
        status_text.success(f"✅ Success! Enhanced to {target_width}x{target_height} in {duration} seconds.")
        
        st.write("---")
        st.write("### 🔍 Step 4: Before / After Comparison")
        st.write("Toggle between the tabs to review the sharpness gains while observing unchanged colors.")
        
        # Scale original preview up directly with NEAREST so it looks pixelated for true side-by-side verification
        comparison_original = image.resize((target_width, target_height), Image.Resampling.NEAREST)
        
        # Render clean tabs so users can slide or toggle between views smoothly
        tab1, tab2 = st.tabs(["📊 Side-by-Side Comparison", "🖼️ Full Enhanced View"])
        
        with tab1:
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.image(comparison_original, caption="Before (Zoomed / Pixelated)", use_container_width=True)
            with comp_col2:
                st.image(output_image, caption="After (Smooth & Sharp)", use_container_width=True)
                
        with tab2:
            st.image(output_image, caption="Final Enhanced Image Output", use_container_width=True)
        
        # Memory download array stream hook
        import io
        img_byte_arr = io.BytesIO()
        # Keep original format or fallback to PNG to preserve transparency
        img_format = image.format if image.format else 'PNG'
        output_image.save(img_byte_arr, format=img_format)
        img_byte_arr = img_byte_arr.getvalue()
        
        st.write("")
        st.download_button(
            label="💾 Download Enhanced Image",
            data=img_byte_arr,
            file_name=f"enhanced_{scale_selection}x_{uploaded_file.name.rsplit('.', 1)[0]}.{img_format.lower()}",
            mime=f"image/{img_format.lower()}"
        )
