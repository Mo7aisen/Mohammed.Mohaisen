import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import streamlit as st
from captum.attr import IntegratedGradients, Saliency
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.ndimage
import gc
import io
import time
import base64
from skimage import measure
import cv2
import warnings
import zipfile

warnings.filterwarnings('ignore')

# --- CENTRALIZED COLOR PALETTE ---
PALETTE = {
    "background": "#F0F2F6",  # Light grey for the main app background
    "primary": "#007BFF",  # Professional blue for primary actions, headers, plots
    "secondary": "#17A2B8",  # A cool cyan/teal for secondary elements
    "text": "#212529",  # Dark charcoal for all text for high readability
    "light_gray": "#E9ECEF",  # A very light grey for card backgrounds or dividers
    "success": "#28A745",  # A clear green for success messages or True Positives
    "danger": "#DC3545",  # A clear red for error messages or False Positives
    "info": "#17A2B8",  # Cyan/teal for informational elements or False Negatives
    "warning": "#FFC107",  # Amber/yellow for warnings
    "tp": [0.16, 0.67, 0.27],  # RGB version of 'success' for Matplotlib
    "fp": [0.86, 0.21, 0.27],  # RGB version of 'danger' for Matplotlib
    "fn": [0.09, 0.64, 0.72],  # RGB version of 'info' for Matplotlib
}


# --- U-Net Model Definition ---
class DoubleConv(nn.Module):
    """Double convolution block with BatchNorm and ReLU"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer"""

    def __init__(self, in_channels, n_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for segmentation"""

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --- Pixel Output Model for SHAP ---
class PixelOutputModel(nn.Module):
    """Model wrapper for pixel-specific output"""

    def __init__(self, model, y, x):
        super(PixelOutputModel, self).__init__()
        self.model = model
        self.y = y
        self.x = x

    def forward(self, x):
        output = self.model(x)
        return output[:, 0, self.y, self.x].unsqueeze(-1)


# --- App Configuration ---
MODEL_DIR = "/home/mohaisen_mohammed/medical_images_segmentation/XAI/trained_models/"
IMG_MASK_BASE_DIR = "./processed_data_single_cell/"
IMG_SIZE = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Medical XAI - Lung Segmentation", layout="wide", initial_sidebar_state="expanded")

# ENHANCED CSS with PALETTE colors
st.markdown(f"""
<style>
    /* Main App Background */
    .main .block-container {{
        background-color: {PALETTE['background']};
        color: {PALETTE['text']};
    }}

    /* Main Header */
    .main-header {{
        background: #FFFFFF;
        border: 1px solid #DDD;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 2rem;
    }}

    .main-header h1 {{
        color: {PALETTE['primary']};
        margin: 0;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: {PALETTE['primary']};
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}

    .stButton > button:hover {{
        background-color: #0056b3;
    }}

    /* Sidebar */
    .css-1d391kg {{
        background-color: #FFFFFF;
        border-right: 1px solid {PALETTE['light_gray']};
    }}

    /* Progress Bar */
    .stProgress .st-bo {{
        background-color: {PALETTE['primary']};
    }}

    /* Metric Cards */
    .metric-card {{
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid {PALETTE['primary']};
        margin: 10px 0;
    }}

    /* Text Color */
    body {{
        color: {PALETTE['text']};
    }}

    .stSelectbox label, .stSlider label, .stCheckbox label {{
        color: {PALETTE['text']};
        font-weight: 500;
    }}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-header"><h1>🫁 Medical AI Explainability Platform</h1><p>Advanced Lung Segmentation Analysis with Pixel-Level Attribution</p></div>',
    unsafe_allow_html=True)

# Device check with more flexibility
if DEVICE.type != 'cuda':
    st.warning("⚠️ Running on CPU. GPU recommended for optimal performance.")
else:
    st.success(f"✅ Running on GPU: {torch.cuda.get_device_name()}")


# --- Memory Management ---
def clear_gpu_memory():
    """Clear GPU memory to prevent out-of-memory errors"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# --- Model Loading ---
@st.cache_resource
def load_model(model_path, device):
    """Load and cache the U-Net model"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = UNet(n_channels=1, n_classes=1)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None


# --- Enhanced Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🎛️ Analysis Controls")

    # --- NEW MODEL SELECTION LOGIC ---
    try:
        if not os.path.isdir(MODEL_DIR):
            st.error(f"Model directory not found: {MODEL_DIR}")
            st.stop()

        model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")])

        if not model_files:
            st.error(f"No model (.pth) files found in directory: {MODEL_DIR}")
            st.stop()

        selected_model_name = st.selectbox("🧠 Select Trained Model", model_files)
        model_path = os.path.join(MODEL_DIR, selected_model_name)

        # Determine dataset from model name
        if "jsrt" in selected_model_name.lower():
            dataset_name = "jsrt"
        elif "montgomery" in selected_model_name.lower():
            dataset_name = "montgomery"
        else:
            st.warning(
                f"Could not determine dataset from model name '{selected_model_name}'. Defaulting to JSRT. Please name models like 'jsrt_...' or 'montgomery_...'.")
            dataset_name = "jsrt"

        img_dir = os.path.join(IMG_MASK_BASE_DIR, dataset_name, "test/images")
        mask_dir = os.path.join(IMG_MASK_BASE_DIR, dataset_name, "test/masks")

    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()

    # Check if data directories exist
    if not os.path.exists(img_dir):
        st.error(f"Image directory for dataset '{dataset_name}' not found: {img_dir}")
        st.stop()

    if not os.path.exists(mask_dir):
        st.warning(
            f"Mask directory for dataset '{dataset_name}' not found: {mask_dir}. Proceeding without ground truth.")

    try:
        img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not img_files:
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        if not img_files:
            img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpeg")))

        if not img_files:
            st.error(f"No images found in directory: {img_dir}")
            st.stop()

        img_name = st.selectbox("🖼️ Select Image", [os.path.basename(f) for f in img_files])
        img_path = os.path.join(img_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
    except Exception as e:
        st.error(f"Error loading image files: {e}")
        st.stop()

    st.markdown("### 📍 Pixel Analysis")
    col1, col2 = st.columns(2)
    with col1:
        x = st.slider("X Coordinate", 0, IMG_SIZE[0] - 1, 128, step=5)
    with col2:
        y = st.slider("Y Coordinate", 0, IMG_SIZE[1] - 1, 128, step=5)

    st.markdown("### 🔬 XAI Method")
    method = st.selectbox(
        "Attribution Method",
        ["Integrated Gradients", "Saliency", "SHAP (GradientExplainer)"],
        help="Choose explainability technique"
    )

    if method == "Integrated Gradients":
        n_steps = st.slider("Integration Steps", 10, 100, 50, help="More steps = higher accuracy")
    elif method.startswith("SHAP"):
        shap_samples = st.slider("SHAP Samples", 10, 100, 50, help="More samples = better accuracy")
    else:
        n_steps = 10
        shap_samples = 10

    st.markdown("### 🎨 Visualization")
    segmentation_threshold = st.slider("Segmentation Threshold", 0.0, 1.0, 0.5, step=0.05,
                                       help="Threshold for converting model output probabilities to a binary mask.")
    blur_sigma = st.slider("Gaussian Blur σ", 0.0, 3.0, 0.8, step=0.2)
    radius = st.slider("Analysis Radius", 5, 50, 15)
    band_width = st.slider("Band Width", 5, 30, 10)
    show_pred_contour = st.checkbox("Show Prediction Contour", value=True)

    st.markdown("### 📊 Attribution Filtering")
    min_abs_attr = st.slider("Min |Attribution|", 0.0, 2.0, 0.1, step=0.1)
    max_abs_attr = st.slider("Max |Attribution|", 0.5, 3.0, 2.0, step=0.1)

    if min_abs_attr >= max_abs_attr:
        st.warning("⚠️ Min must be less than Max")
        max_abs_attr = min_abs_attr + 0.1

    st.markdown("### ⚙️ Advanced")
    attribution_alpha = st.slider("Attribution Overlay Opacity", 0.3, 1.0, 0.7, step=0.1)
    use_gpu_for_shap = st.checkbox("GPU Acceleration (SHAP)", help="Experimental feature")
    show_raw = st.checkbox("Show Raw Data")
    save_raw_masks_in_zip = st.checkbox("Save Raw Masks in ZIP (Debug)",
                                        help="Include raw prediction and GT masks in the ZIP download.")

    st.markdown("---")
    analyze_button = st.button("🔍 **ANALYZE PIXEL**", use_container_width=True, type="primary")
    export_button = st.button("💾 Download Results as ZIP", use_container_width=True)


# --- Enhanced Analysis Functions ---
def pixel_forward_func_factory(model, y, x):
    """Factory function to create pixel-specific forward pass"""

    def pixel_forward_func(input_tensor):
        output = model(input_tensor)
        return output[0, 0, y, x].unsqueeze(0)

    return pixel_forward_func


def run_ig(model, input_tensor, y, x, n_steps):
    """Run Integrated Gradients attribution"""
    clear_gpu_memory()
    pixel_forward_func = pixel_forward_func_factory(model, y, x)
    ig = IntegratedGradients(pixel_forward_func)
    try:
        attr = ig.attribute(input_tensor, n_steps=n_steps, internal_batch_size=1)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            st.warning("Memory limit reached. Reducing steps...")
            clear_gpu_memory()
            attr = ig.attribute(input_tensor, n_steps=max(5, n_steps // 2), internal_batch_size=1)
        else:
            raise e
    del ig
    clear_gpu_memory()
    return attr


def run_saliency(model, input_tensor, y, x):
    """Run Saliency attribution"""
    clear_gpu_memory()
    pixel_forward_func = pixel_forward_func_factory(model, y, x)
    sal = Saliency(pixel_forward_func)
    attr = sal.attribute(input_tensor)
    del sal
    clear_gpu_memory()
    return attr


def run_shap(model, input_tensor, y, x, shap_samples, use_gpu=False):
    """Run SHAP attribution with robust memory management."""
    try:
        import shap
    except ImportError:
        st.error("SHAP library not installed. Please run: pip install shap")
        return None, "SHAP (Failed: Library not found)"

    clear_gpu_memory()

    # --- Attempt GPU first if requested ---
    if use_gpu and torch.cuda.is_available():
        try:
            st.info("Attempting SHAP on GPU...")
            pixel_model_gpu = PixelOutputModel(model, y, x).to(DEVICE)
            input_tensor_gpu = input_tensor.to(DEVICE)
            background_gpu = torch.zeros_like(input_tensor_gpu)

            explainer_gpu = shap.GradientExplainer(pixel_model_gpu, background_gpu)
            # Corrected call: removed 'batch_size'
            shap_values = explainer_gpu.shap_values(input_tensor_gpu, nsamples=shap_samples)

            del pixel_model_gpu, input_tensor_gpu, background_gpu, explainer_gpu
            clear_gpu_memory()
            st.success("SHAP on GPU successful.")
            return shap_values, "SHAP (GradientExplainer on GPU)"
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                st.warning(
                    "CUDA out of memory during SHAP on GPU. Automatically falling back to CPU. This may be slower.")
                clear_gpu_memory()
                # Fall through to the CPU implementation below
            else:
                # Re-raise other runtime errors
                st.error(f"A runtime error occurred during SHAP on GPU: {e}")
                return None, "SHAP (Failed)"
        except TypeError as e:
            if "unexpected keyword argument 'nsamples'" in str(e):
                st.error(
                    "Your version of SHAP might be outdated and not support `nsamples`. Please try updating SHAP or use a different method.")
            else:
                st.error(f"A TypeError occurred during SHAP on GPU: {e}")
            return None, "SHAP (Failed)"

    # --- CPU Implementation (Default or Fallback) ---
    st.info("Running SHAP on CPU...")
    cpu_model = None
    pixel_model_cpu = None
    try:
        # Move model to CPU
        cpu_model = UNet(n_channels=1, n_classes=1).cpu()
        cpu_model.load_state_dict(model.state_dict())
        cpu_model.eval()

        pixel_model_cpu = PixelOutputModel(cpu_model, y, x).cpu()
        input_tensor_cpu = input_tensor.detach().cpu()
        background_cpu = torch.zeros_like(input_tensor_cpu)

        explainer_cpu = shap.GradientExplainer(pixel_model_cpu, background_cpu)
        # Corrected call: removed 'batch_size'
        shap_values = explainer_cpu.shap_values(input_tensor_cpu, nsamples=shap_samples)

        st.success("SHAP on CPU successful.")
        return shap_values, "SHAP (GradientExplainer on CPU)"
    except Exception as e:
        st.error(f"SHAP failed on CPU with error: {str(e)}")
        return None, "SHAP (Failed)"
    finally:
        # Clean up CPU models
        if pixel_model_cpu is not None:
            del pixel_model_cpu
        if cpu_model is not None:
            del cpu_model
        gc.collect()


# --- Enhanced Visualization Functions ---
def circular_mask(shape, center, radius):
    """Create a circular mask around the center"""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return dist_from_center <= radius


def annulus_mask(shape, center, inner_radius, outer_radius):
    """Create an annular mask between inner and outer radii"""
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    return (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)


def create_medical_grade_overlay(image1_array, image2_array, title1, title2, x, y, radius):
    """Create medical-grade visualization with enhanced contrast using PALETTE colors"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('white')

    plot_configs = [
        (image1_array, title1),
        (image2_array, title2)
    ]

    for ax, (img, title) in zip(axes, plot_configs):
        # Display the image using a grayscale colormap.
        # Masks (0/1 values) will show up as black and white.
        ax.imshow(img, cmap="gray")

        # Yellow point marker using PALETTE warning color
        ax.scatter(x, y, c=PALETTE['warning'], s=400, marker='o', linewidths=3,
                   edgecolors='black', alpha=1.0, zorder=10)

        # Professional analysis circle with PALETTE secondary color
        circle = plt.Circle((x, y), radius, color=PALETTE['secondary'], fill=False,
                            linestyle='--', linewidth=4, alpha=0.9)
        ax.add_patch(circle)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20, color=PALETTE['text'])
        ax.axis('off')

    # Add coordinate information
    fig.suptitle(f'Analysis Point: ({x}, {y}) | Radius: {radius}px',
                 fontsize=16, fontweight='bold', y=0.95, color=PALETTE['primary'])

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def create_enhanced_combined_mask(pred_mask, gt_mask):
    """
    Create medical-grade combined visualization using PALETTE colors.
    - Green (TP): Pixel is correctly identified as lung.
    - Red (FP): Pixel is incorrectly identified as lung (model error).
    - Blue (FN): Pixel is lung but was missed by the model (model error).
    - Black (TN): Background pixel, correctly ignored.
    """
    combined = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))

    # Medical imaging color scheme using PALETTE
    pred_only = (pred_mask == 1) & (gt_mask == 0)  # False Positive - PALETTE danger
    gt_only = (gt_mask == 1) & (pred_mask == 0)  # False Negative - PALETTE info
    both = (pred_mask == 1) & (gt_mask == 1)  # True Positive - PALETTE success

    # Use PALETTE colors
    combined[pred_only] = PALETTE['fp']  # False Positive - Red
    combined[gt_only] = PALETTE['fn']  # False Negative - Blue
    combined[both] = PALETTE['tp']  # True Positive - Green

    return combined


def create_professional_attribution_overlay(image, attribution, alpha=0.7):
    """Create professional medical-grade attribution overlay with coolwarm colormap"""
    # Normalize attribution symmetrically around zero for diverging colormap
    attr_max = np.max(np.abs(attribution))
    if attr_max > 0:
        attr_norm = attribution / attr_max
    else:
        attr_norm = np.zeros_like(attribution)

    # Use coolwarm diverging colormap as specified
    colored_attr = plt.cm.coolwarm((attr_norm + 1) / 2)  # Normalize to [0, 1] for colormap

    # Convert grayscale image to RGB
    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=-1)
    else:
        image_rgb = image

    # Normalize image
    image_rgb = image_rgb / 255.0 if image_rgb.max() > 1 else image_rgb

    # Create overlay with attribution intensity modulation
    attr_strength = np.abs(attr_norm)[..., np.newaxis]
    # Only show attribution where it's significant
    significant_mask = np.abs(attr_norm) > 0.1

    overlay = image_rgb.copy()
    # Apply colored attribution only where significant
    overlay[significant_mask] = (alpha * attr_strength[significant_mask] * colored_attr[significant_mask, :3] +
                                 (1 - alpha * attr_strength[significant_mask]) * image_rgb[significant_mask])

    return np.clip(overlay, 0, 1)


def array_to_data_url_enhanced(array, cmap='gray', enhance_contrast=False):
    """Enhanced array to data URL conversion with medical imaging optimizations"""
    buffered = io.BytesIO()

    try:
        if array.ndim == 3:  # RGB image
            if enhance_contrast:
                # Enhance contrast for medical imaging
                array = np.clip(array * 1.2, 0, 1)
            img = Image.fromarray((array * 255).astype(np.uint8))
            img.save(buffered, format="PNG")
        else:
            # Grayscale with enhanced contrast
            if enhance_contrast and array.max() > 0:
                array_norm = (array - array.min()) / (array.max() - array.min())
                array = cv2.equalizeHist((array_norm * 255).astype(np.uint8)) / 255.0
            img = Image.fromarray(np.uint8(array * 255) if array.max() <= 1 else array).convert('L')
            img.save(buffered, format="PNG")

        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        buffered.close()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        st.warning(f"Image conversion warning: {str(e)}")
        return ""


def calculate_median_distance(attr_z, x, y, min_abs_attr, max_abs_attr):
    """
    Calculates the median distance of influencing pixels, weighted by their attribution magnitude.
    This replaces the previous calculate_weighted_median function as requested.
    """
    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)
    if np.any(selected_mask):
        selected_indices = np.where(selected_mask)
        distances = np.sqrt((selected_indices[0] - y) ** 2 + (selected_indices[1] - x) ** 2)
        weights = np.abs(attr_z[selected_mask])
        if len(distances) > 0:
            sorted_idx = np.argsort(distances)
            sorted_distances = distances[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1]
            if total_weight > 0:
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0][0]
                return f"{sorted_distances[median_idx]:.2f}"
    return 'N/A'


def create_medical_metrics_dashboard(metrics_data):
    """Create professional medical metrics dashboard using PALETTE colors"""

    # Clinical significance indicators
    clinical_significance = {
        'high': '🔴 High Clinical Significance',
        'medium': '🟡 Medium Clinical Significance',
        'low': '🟢 Low Clinical Significance'
    }

    # Determine clinical significance based on attribution values
    abs_attr = abs(metrics_data['attr_at_pixel'])
    if abs_attr > 1.5:
        significance = clinical_significance['high']
    elif abs_attr > 0.5:
        significance = clinical_significance['medium']
    else:
        significance = clinical_significance['low']

    html_dashboard = f"""
    <div style="background: {PALETTE['primary']}; padding: 25px; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="color: white; margin: 0; font-family: sans-serif;">📊 Medical AI Analysis Dashboard</h2>
            <p style="color: rgba(255,255,255,0.9); margin: 5px 0;">{significance}</p>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
            <!-- Primary Metrics -->
            <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid {PALETTE['primary']};">
                <h4 style="color: {PALETTE['text']}; margin-bottom: 15px; border-bottom: 2px solid {PALETTE['primary']}; padding-bottom: 5px;">🎯 Primary Analysis</h4>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Model Prediction</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px; color: {PALETTE['primary']};">{metrics_data['pixel_prediction']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Attribution Score</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px; color: {PALETTE['success']};">{metrics_data['attr_at_pixel']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">|Attribution|</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px; color: {PALETTE['warning']};">{metrics_data['abs_attr_at_pixel']:.4f}</span>
                </div>
            </div>

            <!-- Regional Analysis -->
            <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid {PALETTE['danger']};">
                <h4 style="color: {PALETTE['text']}; margin-bottom: 15px; border-bottom: 2px solid {PALETTE['danger']}; padding-bottom: 5px;">🔍 Regional Analysis</h4>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Circle Mean (r={metrics_data['radius']})</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['avg_attr_circle']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Band Mean ({metrics_data['radius']}-{metrics_data['radius'] + metrics_data['band_width']})</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['avg_attr_band']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Range (Min-Max)</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['min_attr_circle']:.3f} to {metrics_data['max_attr_circle']:.3f}</span>
                </div>
            </div>

            <!-- Statistical Analysis -->
            <div style="background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-left: 4px solid {PALETTE['secondary']};">
                <h4 style="color: {PALETTE['text']}; margin-bottom: 15px; border-bottom: 2px solid {PALETTE['secondary']}; padding-bottom: 5px;">📈 Statistical Metrics</h4>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Global Average</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['global_avg_attr']:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Weighted Median Distance</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['weighted_median']}</span>
                </div>
                <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: {PALETTE['light_gray']}; border-radius: 6px;">
                    <span style="font-weight: 600; color: {PALETTE['text']};">Selected Pixels</span>
                    <span style="font-family: monospace; background: {PALETTE['background']}; padding: 4px 8px; border-radius: 4px;">{metrics_data['selected_count']:,}</span>
                </div>
            </div>
        </div>

        <div style="text-align: center; margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 14px;">
                📍 Analysis Coordinates: ({metrics_data['x']}, {metrics_data['y']}) | 
                🔍 Analysis Radius: {metrics_data['radius']}px | 
                📏 Band Width: {metrics_data['band_width']}px
            </p>
        </div>
    </div>
    """

    return html_dashboard


def create_zip_download(images_dict, metrics_data, analysis_results, save_raw_masks=False):
    """Create ZIP file with all results for download"""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Save images
        for img_name, img_array in images_dict.items():
            img_buffer = io.BytesIO()
            if img_array.ndim == 3:
                img = Image.fromarray((img_array * 255).astype(np.uint8))
            else:
                img = Image.fromarray(np.uint8(img_array * 255) if img_array.max() <= 1 else img_array).convert('L')

            img.save(img_buffer, format="PNG")
            zip_file.writestr(f"{img_name}.png", img_buffer.getvalue())

        # Save raw masks for debugging if requested
        if save_raw_masks:
            if 'prediction_mask' in images_dict:
                pred_mask_array = images_dict['prediction_mask']
                img_buffer_pred = io.BytesIO()
                img_pred = Image.fromarray(np.uint8(pred_mask_array * 255)).convert('L')
                img_pred.save(img_buffer_pred, format="PNG")
                zip_file.writestr("debug_prediction_mask.png", img_buffer_pred.getvalue())

            if 'ground_truth_mask' in images_dict:
                gt_mask_array = (images_dict['ground_truth_mask'] > 0).astype(np.uint8)  # Ensure binary
                img_buffer_gt = io.BytesIO()
                img_gt = Image.fromarray(gt_mask_array * 255).convert('L')
                img_gt.save(img_buffer_gt, format="PNG")
                zip_file.writestr("debug_ground_truth_mask.png", img_buffer_gt.getvalue())

        # Save metrics as CSV
        metrics_csv = f"""Metric,Value
Model Prediction,{metrics_data.get('pixel_prediction', 0):.6f}
Attribution Score,{metrics_data.get('attr_at_pixel', 0):.6f}
Absolute Attribution,{metrics_data.get('abs_attr_at_pixel', 0):.6f}
Circle Mean Attribution,{metrics_data.get('avg_attr_circle', 0):.6f}
Band Mean Attribution,{metrics_data.get('avg_attr_band', 0):.6f}
Min Circle Attribution,{metrics_data.get('min_attr_circle', 0):.6f}
Max Circle Attribution,{metrics_data.get('max_attr_circle', 0):.6f}
Global Average Attribution,{metrics_data.get('global_avg_attr', 0):.6f}
Weighted Median Distance,{metrics_data.get('weighted_median', 'N/A')}
Selected Pixel Count,{metrics_data.get('selected_count', 0)}
Analysis X Coordinate,{metrics_data.get('x', 0)}
Analysis Y Coordinate,{metrics_data.get('y', 0)}
Analysis Radius,{metrics_data.get('radius', 0)}
Band Width,{metrics_data.get('band_width', 0)}
"""
        zip_file.writestr("analysis_metrics.csv", metrics_csv)

        # Save analysis parameters
        params_txt = f"""Medical XAI Analysis Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Analysis Parameters:
- Dataset: {analysis_results.get('dataset', 'N/A')}
- Image: {analysis_results.get('image_name', 'N/A')}
- XAI Method: {analysis_results.get('method', 'N/A')}
- Analysis Point: ({metrics_data.get('x', 0)}, {metrics_data.get('y', 0)})
- Analysis Radius: {metrics_data.get('radius', 0)}px
- Band Width: {metrics_data.get('band_width', 0)}px

Key Results:
- Model Prediction: {metrics_data.get('pixel_prediction', 0):.4f}
- Attribution Score: {metrics_data.get('attr_at_pixel', 0):.4f}
- Clinical Significance: {analysis_results.get('significance', 'N/A')}

Color Palette Used:
- Primary: {PALETTE['primary']}
- Success (TP): {PALETTE['success']}
- Danger (FP): {PALETTE['danger']}
- Info (FN): {PALETTE['info']}
"""
        zip_file.writestr("analysis_report.txt", params_txt)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# Global variables to store analysis results
analysis_results_store = {}
images_store = {}

# --- Main Application Logic ---
try:
    # Load model
    model = load_model(model_path, DEVICE)
    if model is None:
        st.stop()

    # Load images with error handling
    try:
        image = Image.open(img_path).convert("L").resize(IMG_SIZE)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L").resize(IMG_SIZE)
        else:
            mask = Image.new("L", IMG_SIZE, 0)
            st.info("ℹ️ Ground truth mask not found. Proceeding without GT analysis.")
    except Exception as e:
        st.error(f"Error loading images: {str(e)}")
        st.stop()

    # Prepare input tensor
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_()

    # Compute prediction mask
    with torch.no_grad():
        pred_logits = model(input_tensor)
        pred_mask = (torch.sigmoid(pred_logits) > segmentation_threshold).float().squeeze().cpu().numpy()

    gt_mask_binary = (np.array(mask) > 127).astype(np.uint8)

    # Display enhanced image analysis
    st.markdown("## 🖼️ Medical Image Analysis")
    create_medical_grade_overlay(np.array(image), gt_mask_binary, "Original Image", "Ground Truth Mask", x, y, radius)

    # Keep the metric calculations but remove the visual plot
    combined_mask = create_enhanced_combined_mask(pred_mask, gt_mask_binary)

    # Calculate comprehensive metrics including True Negatives
    tp_mask = (pred_mask == 1) & (gt_mask_binary == 1)
    fp_mask = (pred_mask == 1) & (gt_mask_binary == 0)
    fn_mask = (pred_mask == 0) & (gt_mask_binary == 1)
    tn_mask = (pred_mask == 0) & (gt_mask_binary == 0)

    intersection = np.sum(tp_mask)
    union = np.sum((pred_mask == 1) | (gt_mask_binary == 1))
    pred_sum = np.sum(pred_mask)
    gt_sum = np.sum(gt_mask_binary)

    iou = intersection / union if union > 0 else 0
    dice = 2 * intersection / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
    sensitivity = intersection / gt_sum if gt_sum > 0 else 0
    precision = intersection / pred_sum if pred_sum > 0 else 0

    # Analysis execution
    if analyze_button:
        with st.spinner("🔬 Performing Medical AI Analysis..."):
            try:
                start_time = time.time()

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Initializing attribution analysis...")
                progress_bar.progress(10)

                attr = None
                effective_method = method

                if method == "Integrated Gradients":
                    status_text.text(f"Computing Integrated Gradients with {n_steps} steps...")
                    progress_bar.progress(30)
                    attr = run_ig(model, input_tensor, y, x, n_steps)

                elif method == "Saliency":
                    status_text.text("Computing Saliency maps...")
                    progress_bar.progress(30)
                    attr = run_saliency(model, input_tensor, y, x)

                elif method.startswith("SHAP"):
                    status_text.text(f"Computing SHAP with {shap_samples} samples...")
                    progress_bar.progress(30)
                    attr, effective_method = run_shap(model, input_tensor, y, x, shap_samples, use_gpu_for_shap)
                    if attr is None:
                        st.warning("SHAP failed. Using Integrated Gradients fallback.")
                        effective_method = "Integrated Gradients (Fallback)"
                        attr = run_ig(model, input_tensor, y, x, 10)

                progress_bar.progress(60)
                status_text.text("Processing attribution results...")

                if attr is not None:
                    # Enhanced attribution processing
                    if torch.is_tensor(attr):
                        attr_np = attr.squeeze().detach().cpu().numpy()
                    else:
                        attr_np = np.squeeze(attr)

                    if len(attr_np.shape) > 2:
                        attr_np = np.mean(attr_np, axis=0)

                    # Enhanced normalization for medical imaging
                    mean, std = np.mean(attr_np), np.std(attr_np)
                    if std == 0:
                        attr_z = np.zeros_like(attr_np)
                    else:
                        attr_z = (attr_np - mean) / std

                    # Extended range for better medical visualization
                    attr_z = np.clip(attr_z, -4, 4)

                    if blur_sigma > 0:
                        attr_z = scipy.ndimage.gaussian_filter(attr_z, sigma=blur_sigma)

                    progress_bar.progress(80)
                    status_text.text("Applying medical filters...")

                    # Apply attribution filtering
                    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)
                    attr_z_masked = np.where(selected_mask, attr_z, 0)

                    # Calculate weighted median distance using the new user-provided function
                    weighted_median_str = calculate_median_distance(attr_z, x, y, min_abs_attr, max_abs_attr)

                    # Regional analysis
                    mask_circle = circular_mask(attr_z.shape, (x, y), radius)
                    mask_band = annulus_mask(attr_z.shape, (x, y), radius, radius + band_width)

                    progress_bar.progress(90)
                    status_text.text("Generating medical dashboard...")

                    st.markdown(f"## 🎯 Medical XAI Results: **{effective_method}**")

                    with torch.no_grad():
                        pixel_prediction = model(input_tensor)[0, 0, y, x].item()

                    # Prepare comprehensive metrics
                    metrics_data = {
                        'pixel_prediction': pixel_prediction,
                        'attr_at_pixel': attr_z[y, x],
                        'abs_attr_at_pixel': np.abs(attr_z[y, x]),
                        'avg_attr_circle': np.mean(attr_z[mask_circle]) if np.any(mask_circle) else 0.0,
                        'avg_attr_band': np.mean(attr_z[mask_band]) if np.any(mask_band) else 0.0,
                        'min_attr_circle': np.min(attr_z[mask_circle]) if np.any(mask_circle) else 0.0,
                        'max_attr_circle': np.max(attr_z[mask_circle]) if np.any(mask_circle) else 0.0,
                        'global_avg_attr': np.mean(attr_z),
                        'weighted_median': weighted_median_str,
                        'selected_count': np.sum(selected_mask),
                        'x': x, 'y': y, 'radius': radius, 'band_width': band_width
                    }

                    # Display medical dashboard
                    medical_dashboard = create_medical_metrics_dashboard(metrics_data)
                    st.components.v1.html(medical_dashboard, height=500)

                    progress_bar.progress(95)
                    status_text.text("Creating interactive visualizations...")

                    # Professional attribution overlay with coolwarm colormap
                    image_url = array_to_data_url_enhanced(np.array(image), enhance_contrast=True)

                    # Create enhanced attribution overlay with coolwarm colormap
                    attr_overlay = create_professional_attribution_overlay(np.array(image), attr_z_masked,
                                                                           attribution_alpha)
                    attr_url = array_to_data_url_enhanced(attr_overlay)

                    combined_mask_url = array_to_data_url_enhanced(combined_mask, enhance_contrast=True)

                    st.markdown("## 🖼️ Interactive Medical Visualization")

                    # Professional medical canvas with PALETTE colors and escaped JS braces
                    html_code = f"""
                    <div style="background: {PALETTE['background']}; padding: 30px; border-radius: 20px; margin: 20px 0;">
                        <div style="display: flex; justify-content: space-around; align-items: flex-start;">
                            <div style="text-align: center; background: white; padding: 20px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); border: 1px solid {PALETTE['light_gray']};">
                                <h3 style="margin-bottom: 15px; color: {PALETTE['primary']}; font-weight: bold;">🔬 Attribution Analysis</h3>
                                <canvas id="xaiCanvasLeft" width="512" height="512" style="border: 4px solid {PALETTE['primary']}; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"></canvas>
                                <div style="margin-top: 15px; padding: 10px; background: {PALETTE['light_gray']}; border-radius: 8px;">
                                    <p style="margin: 0; font-size: 12px; color: {PALETTE['text']};"><strong>Coolwarm Attribution Heatmap</strong></p>
                                    <p style="margin: 5px 0 0 0; font-size: 11px; color: {PALETTE['text']};">🔵 Negative Attribution → 🔴 Positive Attribution</p>
                                </div>
                            </div>

                            <div style="text-align: center; background: white; padding: 20px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1); border: 1px solid {PALETTE['light_gray']};">
                                <h3 style="margin-bottom: 15px; color: {PALETTE['success']}; font-weight: bold;">🎯 Segmentation Analysis</h3>
                                <canvas id="xaiCanvasRight" width="512" height="512" style="border: 4px solid {PALETTE['success']}; border-radius: 12px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);"></canvas>
                                <div style="margin-top: 15px; padding: 10px; background: {PALETTE['light_gray']}; border-radius: 8px;">
                                    <p style="margin: 0; font-size: 12px; color: {PALETTE['text']};"><strong>Clinical Performance Map</strong></p>
                                    <p style="margin: 5px 0 0 0; font-size: 11px; color: {PALETTE['text']};">🔴 False Pos. | 🔵 False Neg. | 🟢 Correct</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <script>
                        const canvasLeft = document.getElementById('xaiCanvasLeft');
                        const ctxLeft = canvasLeft.getContext('2d');
                        const canvasRight = document.getElementById('xaiCanvasRight');
                        const ctxRight = canvasRight.getContext('2d');

                        if (canvasLeft && ctxLeft && canvasRight && ctxRight) {{
                            // Left canvas - Attribution with professional markers
                            const imgLeft = new Image();
                            imgLeft.src = "{attr_url}";
                            imgLeft.onload = function() {{
                                ctxLeft.drawImage(imgLeft, 0, 0, 512, 512);

                                // Yellow point marker using PALETTE warning color
                                ctxLeft.fillStyle = '{PALETTE['warning']}';
                                ctxLeft.strokeStyle = '#000000';
                                ctxLeft.lineWidth = 3;
                                ctxLeft.beginPath();
                                ctxLeft.arc({x * 2}, {y * 2}, 12, 0, 2 * Math.PI);
                                ctxLeft.fill();
                                ctxLeft.stroke();

                                // Professional analysis circles using PALETTE colors
                                ctxLeft.beginPath();
                                ctxLeft.arc({x * 2}, {y * 2}, {radius * 2}, 0, 2 * Math.PI);
                                ctxLeft.strokeStyle = '{PALETTE['secondary']}';
                                ctxLeft.lineWidth = 4;
                                ctxLeft.setLineDash([15, 10]);
                                ctxLeft.stroke();
                                ctxLeft.setLineDash([]);

                                ctxLeft.beginPath();
                                ctxLeft.arc({x * 2}, {y * 2}, {(radius + band_width) * 2}, 0, 2 * Math.PI);
                                ctxLeft.strokeStyle = '{PALETTE['success']}';
                                ctxLeft.lineWidth = 4;
                                ctxLeft.setLineDash([10, 15]);
                                ctxLeft.stroke();
                                ctxLeft.setLineDash([]);

                                // Professional annotation
                                ctxLeft.fillStyle = 'rgba(0, 0, 0, 0.8)';
                                ctxLeft.fillRect(10, 440, 450, 65);
                                ctxLeft.fillStyle = 'white';
                                ctxLeft.font = 'bold 16px Arial';
                                ctxLeft.fillText('🟡 Analysis Point | 🔵 Inner Circle (r={radius})', 20, 465);
                                ctxLeft.fillText('🟢 Extended Band (r={radius + band_width}) | Coolwarm Heatmap', 20, 485);
                            }};

                            // Right canvas - Segmentation with professional markers
                            const imgRight = new Image();
                            imgRight.src = "{combined_mask_url}";
                            imgRight.onload = function() {{
                                ctxRight.drawImage(imgRight, 0, 0, 512, 512);

                                // Yellow point marker (consistent with left)
                                ctxRight.fillStyle = '{PALETTE['warning']}';
                                ctxRight.strokeStyle = '#000000';
                                ctxRight.lineWidth = 3;
                                ctxRight.beginPath();
                                ctxRight.arc({x * 2}, {y * 2}, 12, 0, 2 * Math.PI);
                                ctxRight.fill();
                                ctxRight.stroke();

                                // Professional analysis circles (same colors for consistency)
                                ctxRight.beginPath();
                                ctxRight.arc({x * 2}, {y * 2}, {radius * 2}, 0, 2 * Math.PI);
                                ctxRight.strokeStyle = '{PALETTE['secondary']}';
                                ctxRight.lineWidth = 4;
                                ctxRight.setLineDash([15, 10]);
                                ctxRight.stroke();
                                ctxRight.setLineDash([]);

                                ctxRight.beginPath();
                                ctxRight.arc({x * 2}, {y * 2}, {(radius + band_width) * 2}, 0, 2 * Math.PI);
                                ctxRight.strokeStyle = '{PALETTE['success']}';
                                ctxRight.lineWidth = 4;
                                ctxRight.setLineDash([10, 15]);
                                ctxRight.stroke();
                                ctxRight.setLineDash([]);
                            }};
                        }}
                    </script>
                    """
                    st.components.v1.html(html_code, height=700)

                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")

                    # Store analysis results for download
                    analysis_results_store.clear()
                    images_store.clear()

                    # Determine clinical significance for report
                    abs_attr_val = abs(metrics_data['attr_at_pixel'])
                    if abs_attr_val > 1.5:
                        significance = 'High Clinical Significance'
                    elif abs_attr_val > 0.5:
                        significance = 'Medium Clinical Significance'
                    else:
                        significance = 'Low Clinical Significance'

                    analysis_results_store.update({
                        'dataset': dataset_name,
                        'image_name': img_name,
                        'method': effective_method,
                        'significance': significance,
                        'metrics_data': metrics_data,
                        'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S')
                    })

                    images_store.update({
                        'original_image': np.array(image),
                        'ground_truth_mask': gt_mask_binary,
                        'prediction_mask': pred_mask,
                        'combined_mask': combined_mask,
                        'attribution_overlay': attr_overlay,
                        'attribution_map': attr_z_masked
                    })

                    # Enhanced statistical analysis with PALETTE colors
                    selected_attrs = attr_z[selected_mask]
                    if len(selected_attrs) > 0:
                        st.markdown("## 📊 Advanced Statistical Analysis")

                        # Create comprehensive statistical visualization with PALETTE colors
                        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))
                        fig.patch.set_facecolor('white')

                        # Enhanced histogram with PALETTE colors
                        n, bins, patches = ax1.hist(selected_attrs.flatten(), bins=60,
                                                    color=PALETTE['primary'], alpha=0.7, edgecolor=PALETTE['text'],
                                                    linewidth=1.5)

                        ax1.axvline(x=0, color=PALETTE['text'], linestyle='--', linewidth=3, label='Zero Attribution')
                        ax1.axvline(x=np.mean(selected_attrs), color=PALETTE['danger'], linestyle='-', linewidth=3,
                                    label='Mean')
                        ax1.axvline(x=np.median(selected_attrs), color=PALETTE['info'], linestyle='-.', linewidth=3,
                                    label='Median')

                        ax1.set_title(f"Attribution Distribution (n={len(selected_attrs):,})", fontsize=14,
                                      fontweight='bold', color=PALETTE['text'])
                        ax1.set_xlabel("Attribution Value", fontsize=12, color=PALETTE['text'])
                        ax1.set_ylabel("Frequency", fontsize=12, color=PALETTE['text'])
                        ax1.legend(fontsize=11)
                        ax1.grid(True, alpha=0.3)

                        # Enhanced box plot with PALETTE colors
                        pos_attrs = selected_attrs[selected_attrs > 0]
                        neg_attrs = selected_attrs[selected_attrs < 0]

                        box_data = []
                        box_labels = []
                        if len(neg_attrs) > 0:
                            box_data.append(neg_attrs)
                            box_labels.append(f'Negative\n(n={len(neg_attrs)})')
                        if len(pos_attrs) > 0:
                            box_data.append(pos_attrs)
                            box_labels.append(f'Positive\n(n={len(pos_attrs)})')

                        if box_data:
                            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, notch=True)
                            colors = [PALETTE['info'], PALETTE['danger']]
                            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                                patch.set_facecolor(color)
                                patch.set_alpha(0.7)

                        ax2.set_title("Attribution Distribution by Sign", fontsize=14, fontweight='bold',
                                      color=PALETTE['text'])
                        ax2.set_ylabel("Attribution Value", fontsize=12, color=PALETTE['text'])
                        ax2.grid(True, alpha=0.3)

                        # Spatial distribution with coolwarm colormap
                        spatial_map = np.zeros_like(attr_z)
                        spatial_map[selected_mask] = attr_z[selected_mask]  # Show signed values
                        im3 = ax3.imshow(spatial_map, cmap='coolwarm', vmin=-np.max(np.abs(spatial_map)),
                                         vmax=np.max(np.abs(spatial_map)))
                        ax3.scatter(x, y, c=PALETTE['warning'], s=300, marker='o', linewidths=3, edgecolors='black')
                        circle = plt.Circle((x, y), radius, color=PALETTE['secondary'], fill=False, linewidth=3)
                        ax3.add_patch(circle)
                        ax3.set_title("Spatial Attribution Map (Coolwarm)", fontsize=14, fontweight='bold',
                                      color=PALETTE['text'])
                        ax3.axis('off')
                        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

                        # Distance analysis with coolwarm colors
                        if np.any(selected_mask):
                            selected_indices = np.where(selected_mask)
                            distances = np.sqrt((selected_indices[0] - y) ** 2 + (selected_indices[1] - x) ** 2)
                            attr_values = np.abs(attr_z[selected_indices])

                            ax4.scatter(distances, attr_values, alpha=0.6, c=attr_z[selected_indices],
                                        cmap='coolwarm', s=30, vmin=-np.max(np.abs(attr_z[selected_indices])),
                                        vmax=np.max(np.abs(attr_z[selected_indices])))
                            ax4.set_xlabel("Distance from Analysis Point", fontsize=12, color=PALETTE['text'])
                            ax4.set_ylabel("Absolute Attribution", fontsize=12, color=PALETTE['text'])
                            ax4.set_title("Attribution vs Distance (Coolwarm)", fontsize=14, fontweight='bold',
                                          color=PALETTE['text'])
                            ax4.grid(True, alpha=0.3)

                            # Add trend line
                            if len(distances) > 1:
                                z = np.polyfit(distances, attr_values, 1)
                                p = np.poly1d(z)
                                ax4.plot(distances, p(distances), color=PALETTE['text'], alpha=0.8, linewidth=2,
                                         linestyle='--')

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Enhanced statistics display with PALETTE colors
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("📊 Total Pixels", f"{len(selected_attrs):,}")
                            st.metric("📈 Mean Attribution", f"{np.mean(selected_attrs):.4f}")

                        with col2:
                            st.metric("📊 Std Deviation", f"{np.std(selected_attrs):.4f}")
                            st.metric("📉 Median Attribution", f"{np.median(selected_attrs):.4f}")

                        with col3:
                            st.metric("🔵 Minimum Value", f"{np.min(selected_attrs):.4f}")
                            st.metric("🔴 Maximum Value", f"{np.max(selected_attrs):.4f}")

                        with col4:
                            st.metric("🔵 Negative Count", f"{len(neg_attrs):,}")
                            st.metric("🔴 Positive Count", f"{len(pos_attrs):,}")

                    else:
                        st.info("ℹ️ No attributions within the selected range. Try adjusting the filter values.")

                    # Raw data section
                    if show_raw:
                        with st.expander("🔍 Raw Medical Data Analysis"):
                            st.markdown("### Raw Attribution Statistics")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Array Properties:**")
                                st.write(f"- Shape: {attr_np.shape}")
                                st.write(f"- Data type: {attr_np.dtype}")
                                st.write(f"- Memory usage: {attr_np.nbytes / 1024:.2f} KB")

                            with col2:
                                st.write("**Statistical Properties:**")
                                st.write(f"- Min: {float(attr_np.min()):.6f}")
                                st.write(f"- Max: {float(attr_np.max()):.6f}")
                                st.write(f"- Mean: {float(attr_np.mean()):.6f}")
                                st.write(f"- Std: {float(attr_np.std()):.6f}")

                            # Display raw attribution data
                            st.markdown("### Attribution Matrix Sample")
                            st.write("Sample of attribution values (center region):")
                            center_y, center_x = attr_np.shape[0] // 2, attr_np.shape[1] // 2
                            sample_region = attr_np[max(0, center_y - 5):min(attr_np.shape[0], center_y + 5),
                                            max(0, center_x - 5):min(attr_np.shape[1], center_x + 5)]
                            st.dataframe(sample_region)

                    # Analysis timing
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    st.success(f"✅ Professional analysis completed in {elapsed_time:.2f} seconds!")

                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()

                else:
                    st.error("❌ Attribution computation failed. Please try different settings.")

            except Exception as e:
                st.error(f"❌ Analysis failed with error: {str(e)}")
                st.error("Please check your model path, image directories, and try again.")
                # Clear progress indicators on error
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()

    # Enhanced Export functionality with ZIP download
    if export_button:
        if analysis_results_store and images_store:
            try:
                # Get the latest analysis data
                metrics_data = analysis_results_store.get('metrics_data', {})

                # Create and download ZIP file with the debug flag
                zip_data = create_zip_download(images_store, metrics_data, analysis_results_store,
                                               save_raw_masks_in_zip)

                st.download_button(
                    label="📦 Download Complete Analysis Results (ZIP)",
                    data=zip_data,
                    file_name=f"medical_xai_analysis_{img_name}_{time.strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

                st.success(
                    "✅ ZIP file prepared for download! Click the button above to save the complete analysis results.")

                # Show what's included in the download
                with st.expander("📋 Included in ZIP Download"):
                    st.write("**Images:**")
                    # Only list user-facing images, not internal arrays unless debugging
                    st.write("- original_image.png")
                    st.write("- ground_truth_mask.png")
                    st.write("- combined_mask.png")
                    st.write("- attribution_overlay.png")

                    if save_raw_masks_in_zip:
                        st.write("- debug_prediction_mask.png")
                        st.write("- debug_ground_truth_mask.png")

                    st.write("**Data Files:**")
                    st.write("- analysis_metrics.csv (Numerical results)")
                    st.write("- analysis_report.txt (Complete analysis report)")

                    st.write("**Analysis Summary:**")
                    st.write(f"- Dataset: {analysis_results_store.get('dataset', 'N/A')}")
                    st.write(f"- XAI Method: {analysis_results_store.get('method', 'N/A')}")
                    st.write(f"- Analysis Time: {analysis_results_store.get('analysis_time', 'N/A')}")

            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")
        else:
            st.info("📁 Please run an analysis first to generate results for export.")
            st.write("Available after analysis:")
            st.write("- ✅ Attribution maps as PNG")
            st.write("- ✅ Statistical analysis as CSV")
            st.write("- ✅ Complete analysis report")
            st.write("- ✅ All visualizations in one ZIP file")

except Exception as e:
    st.error(f"❌ Application initialization failed: {str(e)}")
    st.error("Please check your file paths and model files.")
    st.info("Make sure the following directories exist:")
    st.code("""
    ./processed_data_single_cell/jsrt/test/images/
    ./processed_data_single_cell/jsrt/test/masks/
    ./processed_data_single_cell/montgomery/test/images/
    ./processed_data_single_cell/montgomery/test/masks/
    /home/mohaisen_mohammed/medical_images_segmentation/XAI/trained_models/
    """)

# Application footer with PALETTE styling
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 20px; background: {PALETTE['primary']}; border-radius: 10px; margin-top: 30px;">
    <h4 style="color: white; margin: 0; font-family: sans-serif;">🏥 Medical AI Explainability APP</h4>
    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">
        Professional lung segmentation analysis with coolwarm attribution heatmaps
    </p>
</div>
""", unsafe_allow_html=True)
