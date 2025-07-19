import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.patches import Circle
import matplotlib.patches as patches

# --- Page Configuration ---
st.set_page_config(
    page_title="🫁 XAI Lung Segmentation Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants and Configuration ---
try:
    with open('config.yaml', 'r') as f:
        CONFIG = yaml.safe_load(f)
    OUTPUT_DIR = CONFIG['output_base_dir']
    DATASETS = CONFIG['datasets']
except Exception as e:
    OUTPUT_DIR = "./outputs"
    DATASETS = {}


# --- Enhanced XAI Analysis Functions ---
def calculate_weighted_median_distance(attr_map, center_x, center_y, min_abs_attr=None, max_abs_attr=None):
    """Calculate weighted median distance based on attribution values"""
    attr_z = attr_map.copy()

    if min_abs_attr is None:
        min_abs_attr = np.percentile(np.abs(attr_z), 10)
    if max_abs_attr is None:
        max_abs_attr = np.percentile(np.abs(attr_z), 90)

    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)

    if np.any(selected_mask):
        selected_indices = np.where(selected_mask)
        distances = np.sqrt((selected_indices[0] - center_y) ** 2 + (selected_indices[1] - center_x) ** 2)
        weights = np.abs(attr_z[selected_mask])

        if len(distances) > 0:
            sorted_idx = np.argsort(distances)
            sorted_distances = distances[sorted_idx]
            sorted_weights = weights[sorted_idx]
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1] if len(cumulative_weights) > 0 else 0

            if total_weight > 0:
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0]
                if len(median_idx) > 0:
                    median_distance = sorted_distances[median_idx[0]]
                    return median_distance, sorted_distances, cumulative_weights, total_weight

    return None, None, None, None


def calculate_enhanced_cumulative_histogram(attr_map, center_x, center_y):
    """Enhanced cumulative histogram calculation"""
    y_coords, x_coords = np.mgrid[0:attr_map.shape[0], 0:attr_map.shape[1]]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    weights = np.abs(attr_map)

    distances_flat = distances.flatten()
    weights_flat = weights.flatten()

    non_zero_mask = weights_flat > 0
    distances_clean = distances_flat[non_zero_mask]
    weights_clean = weights_flat[non_zero_mask]

    if len(distances_clean) == 0:
        return None, None

    sorted_idx = np.argsort(distances_clean)
    sorted_distances = distances_clean[sorted_idx]
    sorted_weights = weights_clean[sorted_idx]
    cumulative_weights = np.cumsum(sorted_weights)

    if cumulative_weights[-1] > 0:
        normalized_cumulative = cumulative_weights / cumulative_weights[-1]
    else:
        normalized_cumulative = cumulative_weights

    max_dist = int(np.ceil(sorted_distances.max()))
    distance_bins = np.arange(0, max_dist + 1)

    binned_cumulative = []
    for dist in distance_bins:
        mask = sorted_distances <= dist
        if np.any(mask):
            binned_cumulative.append(normalized_cumulative[mask][-1])
        else:
            binned_cumulative.append(0.0)

    return distance_bins, np.array(binned_cumulative)


# --- Helper Functions ---
def parse_run_name(run_name):
    """Parse experiment run name into components"""
    parts = run_name.replace("unet_", "").split('_')
    if len(parts) >= 3:
        return {
            "dataset": parts[0].title(),
            "data_size": parts[1].title(),
            "epochs": str(parts[2])
        }
    return {"dataset": "unknown", "data_size": "unknown", "epochs": "unknown"}


def safe_file_check(filepath):
    """Safely check if file exists"""
    try:
        return os.path.exists(filepath)
    except:
        return False


# --- Data Loading Functions (with Caching) ---
@st.cache_data
def get_available_runs():
    """Get list of available experiment runs"""
    if not os.path.isdir(OUTPUT_DIR):
        return []

    runs = []
    try:
        for d in os.listdir(OUTPUT_DIR):
            run_dir = os.path.join(OUTPUT_DIR, d)
            if os.path.isdir(run_dir) and d.startswith('unet_'):
                has_final_model = safe_file_check(os.path.join(run_dir, 'final_model.pth'))
                has_training_log = safe_file_check(os.path.join(run_dir, 'training_log.csv'))

                if has_final_model or has_training_log:
                    runs.append(d)
    except Exception as e:
        st.error(f"Error scanning runs: {e}")

    return sorted(runs)


@st.cache_data
def get_available_states(run_name):
    """Get available evaluation states for a run"""
    eval_dir = os.path.join(OUTPUT_DIR, run_name, "evaluation")
    if not safe_file_check(eval_dir):
        return []

    states = []
    try:
        for state in ['underfitting', 'good_fitting', 'overfitting']:
            state_dir = os.path.join(eval_dir, state)
            if safe_file_check(state_dir):
                states.append(state)
    except:
        pass

    return states


@st.cache_data
def load_run_data(run_name, state, split):
    """Load evaluation results for a specific run, state, and split"""
    summary_path = os.path.join(OUTPUT_DIR, run_name, "evaluation", state, split, "_evaluation_summary.json")
    if not safe_file_check(summary_path):
        return None

    try:
        with open(summary_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {summary_path}: {e}")
        return None


@st.cache_data
def load_npz_data(npz_path):
    """Load XAI results from NPZ file"""
    if not safe_file_check(npz_path):
        return None

    try:
        with np.load(npz_path) as data:
            return {key: data[key] for key in data}
    except Exception as e:
        st.error(f"Error loading NPZ file: {e}")
        return None


@st.cache_data
def load_training_log(run_name):
    """Load training log for a run"""
    log_path = os.path.join(OUTPUT_DIR, run_name, "training_log.csv")
    if safe_file_check(log_path):
        try:
            return pd.read_csv(log_path)
        except Exception as e:
            st.error(f"Error loading training log: {e}")
    return None


def get_original_image_path(image_name, dataset_name):
    """Get path to original image file"""
    if dataset_name.lower() in DATASETS:
        dataset_config = DATASETS[dataset_name.lower()]
        base_path = os.path.join(dataset_config['path'], dataset_config['images'])
        return os.path.join(base_path, image_name)

    if dataset_name.lower() == 'montgomery':
        return os.path.join("/home/mohaisen_mohammed/Datasets/MontgomeryDataset/CXR_png/", image_name)
    else:
        return os.path.join("/home/mohaisen_mohammed/Datasets/JSRT/images/", image_name)


def plot_training_curves(training_df, title):
    """Plot training and validation loss curves"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=training_df['epoch'],
        y=training_df['train_loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=3)
    ))

    fig.add_trace(go.Scatter(
        x=training_df['epoch'],
        y=training_df['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=2),
        marker=dict(size=3)
    ))

    best_epoch = training_df.loc[training_df['val_loss'].idxmin()]
    fig.add_vline(
        x=best_epoch['epoch'],
        line_dash="dash",
        line_color="green",
        annotation_text=f"Best Model (Epoch {int(best_epoch['epoch'])})"
    )

    fig.update_layout(
        title=f"Training Curves - {title}",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=300,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig


def create_interactive_coordinate_selector(image, current_x, current_y, column_id, image_size=(256, 256)):
    """Create an interactive coordinate selector with immediate visual feedback"""

    # Create figure with image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray', extent=[0, image_size[1], image_size[0], 0])

    # Add crosshair and target point
    ax.axhline(y=current_y, color='#FF6B35', linestyle='--', alpha=0.9, linewidth=2.5, label='Target Line')
    ax.axvline(x=current_x, color='#FF6B35', linestyle='--', alpha=0.9, linewidth=2.5)

    # Add a prominent target marker
    circle = Circle((current_x, current_y), radius=8, color='#FF6B35', fill=False, linewidth=3)
    ax.add_patch(circle)
    ax.plot(current_x, current_y, 'o', color='#FF6B35', markersize=12, markerfacecolor='white',
            markeredgecolor='#FF6B35', markeredgewidth=3, label=f'Target ({current_x}, {current_y})')

    # Add grid for better reference
    for i in range(0, image_size[0], 32):
        ax.axhline(y=i, color='white', alpha=0.2, linewidth=0.5)
    for j in range(0, image_size[1], 32):
        ax.axvline(x=j, color='white', alpha=0.2, linewidth=0.5)

    ax.set_title(f"📍 Interactive Point Selection\nCurrent Target: ({current_x}, {current_y})",
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

    # Set limits and ticks
    ax.set_xlim(0, image_size[1])
    ax.set_ylim(image_size[0], 0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12)

    # Style improvements
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)

    plt.tight_layout()
    return fig


def display_enhanced_attribution_map_v2(attr_map, center_x, center_y, title, image_size=(256, 256)):
    """Enhanced attribution map with better color scheme and visibility"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use a better colormap for attribution visualization
    # RdBu_r gives good contrast: red for positive, blue for negative
    vmax = np.max(np.abs(attr_map))
    vmin = -vmax

    im = ax.imshow(attr_map, cmap='RdBu_r', interpolation='bilinear',
                   vmin=vmin, vmax=vmax, alpha=0.9)

    # Add crosshair with better visibility
    ax.axhline(y=center_y, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)
    ax.axvline(x=center_x, color='#00FF00', linestyle='-', alpha=1.0, linewidth=3)

    # Add target point with multiple visual elements
    circle1 = Circle((center_x, center_y), radius=12, color='#00FF00', fill=False, linewidth=4)
    circle2 = Circle((center_x, center_y), radius=8, color='white', fill=False, linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.plot(center_x, center_y, 'o', color='#00FF00', markersize=8, markerfacecolor='white',
            markeredgecolor='#00FF00', markeredgewidth=3)

    # Add grid for reference
    grid_spacing = 32
    for i in range(0, image_size[0], grid_spacing):
        ax.axhline(y=i, color='white', alpha=0.15, linewidth=0.5)
    for j in range(0, image_size[1], grid_spacing):
        ax.axvline(x=j, color='white', alpha=0.15, linewidth=0.5)

    ax.set_title(f"{title}\n🎯 Analysis Center: ({center_x}, {center_y})",
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label('Attribution Score\n(Red: Positive, Blue: Negative)',
                   rotation=270, labelpad=25, fontsize=14, fontweight='bold')

    # Better tick formatting
    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlim(0, image_size[1])
    ax.set_ylim(image_size[0], 0)

    # Add attribution statistics as text
    pos_attr = np.sum(attr_map[attr_map > 0])
    neg_attr = np.sum(attr_map[attr_map < 0])

    stats_text = f"Positive Attribution: {pos_attr:.3f}\nNegative Attribution: {neg_attr:.3f}\nNet Attribution: {pos_attr + neg_attr:.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def display_enhanced_uncertainty_map_v2(uncertainty_map, title, image_size=(256, 256)):
    """Enhanced uncertainty map with better visualization"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Use viridis for uncertainty (dark blue to bright yellow)
    im = ax.imshow(uncertainty_map, cmap='plasma', interpolation='bilinear', alpha=0.9)

    # Add grid
    grid_spacing = 32
    for i in range(0, image_size[0], grid_spacing):
        ax.axhline(y=i, color='white', alpha=0.2, linewidth=0.5)
    for j in range(0, image_size[1], grid_spacing):
        ax.axvline(x=j, color='white', alpha=0.2, linewidth=0.5)

    ax.set_title(f"{title}\n📊 Model Prediction Uncertainty",
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label('Uncertainty Score\n(Higher = More Uncertain)',
                   rotation=270, labelpad=25, fontsize=14, fontweight='bold')

    cbar.ax.tick_params(labelsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)

    ax.set_xlim(0, image_size[1])
    ax.set_ylim(image_size[0], 0)

    # Add uncertainty statistics
    mean_uncertainty = np.mean(uncertainty_map)
    max_uncertainty = np.max(uncertainty_map)
    std_uncertainty = np.std(uncertainty_map)

    stats_text = f"Mean Uncertainty: {mean_uncertainty:.4f}\nMax Uncertainty: {max_uncertainty:.4f}\nStd Uncertainty: {std_uncertainty:.4f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    return fig


def create_combined_overlay(original_image, ground_truth, prediction, title):
    """Create a combined overlay visualization"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Display original image
    ax.imshow(original_image, cmap='gray', alpha=0.7)

    # Overlay ground truth in green
    gt_overlay = np.zeros((*ground_truth.shape, 4))
    gt_overlay[ground_truth > 0.5] = [0, 1, 0, 0.4]  # Green with transparency
    ax.imshow(gt_overlay)

    # Overlay prediction in red
    pred_overlay = np.zeros((*prediction.shape, 4))
    pred_overlay[prediction > 0.5] = [1, 0, 0, 0.4]  # Red with transparency
    ax.imshow(pred_overlay)

    ax.set_title(f"{title}\n🟢 Ground Truth | 🔴 Prediction",
                 fontsize=16, pad=20, fontweight='bold')
    ax.set_xlabel("X Coordinate (pixels)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.4, label='Ground Truth'),
                       Patch(facecolor='red', alpha=0.4, label='Prediction')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    ax.set_xlim(0, original_image.shape[1])
    ax.set_ylim(original_image.shape[0], 0)

    plt.tight_layout()
    return fig


def display_column(column_id, available_runs):
    """Display analysis column with enhanced interactive features"""
    st.markdown(f"### 🔬 Analysis Column {column_id}")

    # Run selection
    selected_run = st.selectbox(
        f"Select Model Run",
        [""] + available_runs,
        key=f"run_{column_id}",
        format_func=lambda name: " | ".join(parse_run_name(name).values()) if name else "Select a run..."
    )

    if not selected_run:
        st.info("Please select a model run.")
        return None

    # State selection
    available_states = get_available_states(selected_run)
    if not available_states:
        st.warning(f"No evaluation results found for {selected_run}")
        return None

    selected_state = st.selectbox(
        f"Select Model State",
        available_states,
        key=f"state_{column_id}",
        format_func=lambda x: x.replace('_', ' ').title()
    )

    # Split selection
    selected_split = st.selectbox(
        f"Select Data Split",
        ["test", "validation", "training"],
        key=f"split_{column_id}"
    )

    # Load data
    with st.spinner("Loading data..."):
        run_data = load_run_data(selected_run, selected_state, selected_split)

    if not run_data:
        st.error(f"Could not load data for {selected_run}/{selected_state}/{selected_split}")
        return None

    # Image selection
    results = run_data.get("per_sample_results", [])
    if not results:
        st.warning("No sample results found.")
        return None

    sorted_results = sorted(results, key=lambda x: x.get("dice_score", 0), reverse=True)

    selected_image_data = st.selectbox(
        f"Select Image",
        sorted_results,
        key=f"image_{column_id}",
        format_func=lambda x: f"{x['image_name']} | Dice: {x['dice_score']:.4f} | IoU: {x['iou_score']:.4f}"
    )

    if not selected_image_data:
        st.warning("No images available.")
        return None

    # Display individual image metrics prominently
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 Dice Score", f"{selected_image_data['dice_score']:.4f}")
    with col2:
        st.metric("📐 IoU Score", f"{selected_image_data['iou_score']:.4f}")

    # Load XAI data
    with st.spinner("Loading XAI data..."):
        npz_data = load_npz_data(selected_image_data["xai_results_path"])

    if not npz_data:
        st.error(f"Failed to load XAI data for {selected_image_data['image_name']}")
        return None

    # Display basic information
    parsed_name = parse_run_name(selected_run)
    epoch_used = run_data.get('epoch_used', 'N/A')

    st.markdown(f"**{parsed_name['dataset']} | {parsed_name['data_size']} | {parsed_name['epochs']} epochs**")
    st.markdown(
        f"State: `{selected_state.replace('_', ' ').title()}` | Split: `{selected_split.upper()}` | Epoch: `{epoch_used}`")

    # Load original image
    try:
        original_image_path = get_original_image_path(
            selected_image_data['image_name'],
            parsed_name['dataset']
        )

        if safe_file_check(original_image_path):
            original_image = Image.open(original_image_path)
            original_image_array = np.array(original_image)
        else:
            original_image_array = np.ones((256, 256)) * 128
            st.warning(f"Original image not found")

        gt_mask = npz_data['ground_truth']
        prediction = npz_data['prediction']

    except Exception as e:
        st.error(f"Error loading images: {e}")
        return None

    # ROW 1: Combined overlay and uncertainty
    st.markdown("**🖼️ Image Overview:**")
    col1, col2 = st.columns(2)

    with col1:
        try:
            fig = create_combined_overlay(original_image_array, gt_mask, prediction,
                                          "Combined Ground Truth & Prediction")
            st.pyplot(fig)
            plt.close()
        except Exception as e:
            st.error(f"Error displaying combined image: {e}")

    with col2:
        try:
            if 'uncertainty_map' in npz_data:
                fig = display_enhanced_uncertainty_map_v2(
                    npz_data['uncertainty_map'],
                    "Prediction Uncertainty"
                )
                st.pyplot(fig)
                plt.close()
            else:
                st.warning("Uncertainty map not available")
        except Exception as e:
            st.error(f"Error displaying uncertainty map: {e}")

    # ROW 2: Enhanced Interactive Coordinate Selection
    st.markdown("**🎯 Interactive Analysis Point Selection:**")

    # Initialize coordinates if not in session state
    if f"x_{column_id}" not in st.session_state:
        st.session_state[f"x_{column_id}"] = 128
    if f"y_{column_id}" not in st.session_state:
        st.session_state[f"y_{column_id}"] = 128

    # Enhanced coordinate controls with better layout
    coord_col1, coord_col2, coord_col3, coord_col4 = st.columns([2, 2, 2, 2])

    with coord_col1:
        st.markdown("**X Coordinate:**")
        x_coord = st.slider("", 0, 255, st.session_state[f"x_{column_id}"], key=f"x_slider_{column_id}")
        st.session_state[f"x_{column_id}"] = x_coord

    with coord_col2:
        st.markdown("**Y Coordinate:**")
        y_coord = st.slider("", 0, 255, st.session_state[f"y_{column_id}"], key=f"y_slider_{column_id}")
        st.session_state[f"y_{column_id}"] = y_coord

    with coord_col3:
        st.markdown("**Quick Presets:**")
        if st.button("🎯 Center (128,128)", key=f"center_{column_id}"):
            st.session_state[f"x_{column_id}"] = 128
            st.session_state[f"y_{column_id}"] = 128
            st.rerun()

    with coord_col4:
        st.markdown("**Fine Tune:**")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("⬅️ X-1", key=f"x_minus_{column_id}"):
                st.session_state[f"x_{column_id}"] = max(0, st.session_state[f"x_{column_id}"] - 1)
                st.rerun()
            if st.button("⬇️ Y+1", key=f"y_plus_{column_id}"):
                st.session_state[f"y_{column_id}"] = min(255, st.session_state[f"y_{column_id}"] + 1)
                st.rerun()
        with col_b:
            if st.button("➡️ X+1", key=f"x_plus_{column_id}"):
                st.session_state[f"x_{column_id}"] = min(255, st.session_state[f"x_{column_id}"] + 1)
                st.rerun()
            if st.button("⬆️ Y-1", key=f"y_minus_{column_id}"):
                st.session_state[f"y_{column_id}"] = max(0, st.session_state[f"y_{column_id}"] - 1)
                st.rerun()

    # Display interactive coordinate selector with instant feedback
    try:
        fig = create_interactive_coordinate_selector(
            original_image_array,
            st.session_state[f"x_{column_id}"],
            st.session_state[f"y_{column_id}"],
            column_id
        )
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error displaying coordinate selector: {e}")

    # ROW 3: Enhanced Integrated Gradients Analysis (Always visible)
    st.markdown("**🧠 Integrated Gradients Analysis:**")
    try:
        if 'ig_map' in npz_data:
            # Display enhanced IG map
            fig = display_enhanced_attribution_map_v2(
                npz_data['ig_map'],
                st.session_state[f"x_{column_id}"],
                st.session_state[f"y_{column_id}"],
                "Integrated Gradients Attribution"
            )
            st.pyplot(fig)
            plt.close()

            # Calculate and display metrics
            median_dist, sorted_distances, cumulative_weights, total_weight = calculate_weighted_median_distance(
                npz_data['ig_map'],
                st.session_state[f"x_{column_id}"],
                st.session_state[f"y_{column_id}"]
            )

            if median_dist is not None:
                st.metric("📏 Weighted Median Distance", f"{median_dist:.2f} pixels")

            # Enhanced cumulative histogram
            distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(
                npz_data['ig_map'],
                st.session_state[f"x_{column_id}"],
                st.session_state[f"y_{column_id}"]
            )

            if distance_bins is not None:
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.plot(distance_bins, cumulative_values, marker='o', linewidth=3,
                        markersize=6, color='#1f77b4', markerfacecolor='white',
                        markeredgecolor='#1f77b4', markeredgewidth=2)

                if median_dist is not None:
                    ax.axvline(x=median_dist, color='red', linestyle='--', linewidth=3,
                               label=f'Weighted Median: {median_dist:.2f}px')

                ax.set_title(
                    f"📈 Cumulative Weighted Attribution Analysis\nFrom Point ({st.session_state[f'x_{column_id}']}, {st.session_state[f'y_{column_id}']})",
                    fontsize=16, fontweight='bold', pad=20)
                ax.set_xlabel("Distance from Center (pixels)", fontsize=14, fontweight='bold')
                ax.set_ylabel("Normalized Cumulative Weight", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, linewidth=1)
                ax.set_ylim(0, 1.05)
                ax.legend(fontsize=12)

                # Style improvements
                ax.spines['top'].set_linewidth(2)
                ax.spines['right'].set_linewidth(2)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.tick_params(axis='both', which='major', labelsize=12)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

        else:
            st.warning("Integrated Gradients data not available")
    except Exception as e:
        st.error(f"Error displaying Integrated Gradients: {e}")

    # ROW 4: Training Curve
    st.markdown("**📈 Training History:**")
    try:
        training_df = load_training_log(selected_run)
        if training_df is not None:
            fig = plot_training_curves(training_df,
                                       f"{parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}ep")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Training log not available")
    except Exception as e:
        st.error(f"Error displaying training curve: {e}")

    # Return data for combined analysis
    try:
        distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(
            npz_data['ig_map'],
            st.session_state[f"x_{column_id}"],
            st.session_state[f"y_{column_id}"]
        )

        return {
            "run_name": f"Col {column_id}: {parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}",
            "hist_data": {"distance_bins": distance_bins, "cumulative_values": cumulative_values},
            "state": selected_state,
            "epoch_used": epoch_used,
            "center_point": (st.session_state[f"x_{column_id}"], st.session_state[f"y_{column_id}"]),
            "dice_score": selected_image_data['dice_score']
        }
    except Exception as e:
        st.error(f"Error calculating histogram: {e}")
        return None


def display_training_analysis():
    """Display training curve analysis"""
    st.header("📈 Training Analysis")

    available_runs = get_available_runs()
    if not available_runs:
        st.warning("No runs available for analysis.")
        return

    # Multi-select for comparing multiple runs
    selected_runs = st.multiselect(
        "Select runs to compare:",
        available_runs,
        default=available_runs[:4] if len(available_runs) > 4 else available_runs,
        format_func=lambda x: " | ".join(parse_run_name(x).values())
    )

    if not selected_runs:
        st.info("Please select at least one run to analyze.")
        return

    # Create comparison plot using Plotly
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, run_name in enumerate(selected_runs):
        try:
            training_df = load_training_log(run_name)
            if training_df is not None:
                parsed = parse_run_name(run_name)
                label = f"{parsed['dataset']} {parsed['data_size']} {parsed['epochs']}ep"
                color = colors[i % len(colors)]

                # Plot training curve
                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['train_loss'],
                    mode='lines+markers',
                    name=f"{label} (Train)",
                    line=dict(color=color, dash='solid', width=2),
                    marker=dict(size=4)
                ))

                # Plot validation curve
                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['val_loss'],
                    mode='lines+markers',
                    name=f"{label} (Val)",
                    line=dict(color=color, dash='dash', width=2),
                    marker=dict(size=4)
                ))

        except Exception as e:
            st.error(f"Error processing {run_name}: {e}")

    fig.update_layout(
        title="📊 Training Progress Comparison",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=600,
        font=dict(size=12)
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Main App ---
def main():
    st.title("🫁 Enhanced XAI Analysis for Lung Segmentation")
    st.markdown(
        "*Professional Interactive Analysis with Enhanced Attribution Mapping and Real-time Coordinate Selection*")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    st.sidebar.markdown("### Analysis Tools")

    page = st.sidebar.radio(
        "Choose Analysis Mode:",
        ["🔬 Comparative Analysis", "📈 Training Analysis"],
        help="Select the type of analysis you want to perform"
    )

    # Add helpful information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 How to Use")
    if page == "🔬 Comparative Analysis":
        st.sidebar.markdown("""
        **Interactive Features:**
        - 🎯 Use sliders for precise coordinate selection
        - ⬅️➡️⬆️⬇️ Fine-tune with arrow buttons
        - 🖱️ Visual feedback shows target instantly
        - 🎨 Enhanced color schemes for better readability
        - 📊 Real-time attribution analysis
        """)
    else:
        st.sidebar.markdown("""
        **Training Analysis:**
        - 📈 Compare multiple model training curves
        - 🔍 Identify overfitting and underfitting
        - ⚖️ Analyze training vs validation performance
        - 📊 Interactive Plotly visualizations
        """)

    if page == "🔬 Comparative Analysis":
        st.header("🔬 Professional Comparative XAI Analysis")
        st.markdown("*Compare multiple models with enhanced interactive controls and professional visualizations*")

        available_runs = get_available_runs()
        if not available_runs:
            st.error("❌ No evaluation results found. Please run experiments first.")
            st.info("💡 Make sure your experiment outputs are in the correct directory structure.")
            return

        # Add a brief instruction
        st.info(
            "🎯 **Instructions:** Select different models in each column to compare their XAI analysis. Use the interactive coordinate selectors to analyze specific points in the images.")

        # Create three columns for comparison
        col1, col2, col3 = st.columns(3, gap="large")

        all_column_data = []

        with col1:
            result1 = display_column(1, available_runs)
            if result1:
                all_column_data.append(result1)

        with col2:
            result2 = display_column(2, available_runs)
            if result2:
                all_column_data.append(result2)

        with col3:
            result3 = display_column(3, available_runs)
            if result3:
                all_column_data.append(result3)

        # Enhanced combined comparison plot
        if len(all_column_data) > 1:
            st.markdown("---")
            st.subheader("🔗 Professional Model Comparison")
            st.markdown("*Comparative analysis of cumulative weighted attributions across selected models*")

            try:
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                for i, item in enumerate(all_column_data):
                    color = colors[i % len(colors)]
                    hist_data = item["hist_data"]

                    if hist_data["distance_bins"] is not None and hist_data["cumulative_values"] is not None:
                        # Main attribution curve with enhanced styling
                        fig.add_trace(go.Scatter(
                            x=hist_data["distance_bins"],
                            y=hist_data["cumulative_values"],
                            mode='lines+markers',
                            name=f'{item["run_name"]} ({item["state"]})',
                            line=dict(color=color, width=3),
                            marker=dict(size=6, color=color, line=dict(width=2, color='white'))
                        ))

                fig.update_layout(
                    title="📊 Professional Comparison of Cumulative Weighted Attributions",
                    xaxis_title="Distance from Analysis Center (pixels)",
                    yaxis_title="Normalized Cumulative Attribution Weight",
                    hovermode='x unified',
                    height=600,
                    yaxis=dict(range=[0, 1.05]),
                    font=dict(size=12),
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )

                fig.update_traces(
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                  "Distance: %{x} pixels<br>" +
                                  "Cumulative Weight: %{y:.3f}<br>" +
                                  "<extra></extra>"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Enhanced summary statistics table
                st.subheader("📊 Detailed Comparison Statistics")
                stats_data = []
                for item in all_column_data:
                    stats_data.append({
                        "Model Configuration": item["run_name"],
                        "Model State": item["state"].replace('_', ' ').title(),
                        "Analysis Point": f"({item['center_point'][0]}, {item['center_point'][1]})",
                        "Image Dice Score": f"{item['dice_score']:.4f}",
                        "Training Epoch": item.get("epoch_used", "N/A")
                    })

                df_stats = pd.DataFrame(stats_data)
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

                # Additional insights
                st.markdown("### 🔍 Analysis Insights")
                best_model = max(all_column_data, key=lambda x: x['dice_score'])
                st.success(
                    f"🏆 **Best Performing Model:** {best_model['run_name']} with Dice Score: {best_model['dice_score']:.4f}")

                avg_dice = np.mean([item['dice_score'] for item in all_column_data])
                st.info(f"📈 **Average Dice Score:** {avg_dice:.4f}")

            except Exception as e:
                st.error(f"❌ Error creating comparison plot: {e}")

        elif len(all_column_data) == 1:
            st.info(
                "💡 **Tip:** Select models in multiple columns to enable comparative analysis and see model performance differences.")

    elif page == "📈 Training Analysis":
        display_training_analysis()


if __name__ == "__main__":
    # Set enhanced page style with professional theme
    st.markdown("""
    <style>
        /* Main app styling */
        .stApp > header {
            background-color: transparent;
        }
        .main > div {
            padding-top: 1rem;
        }

        /* Typography improvements */
        h1 {
            color: #1f77b4;
            font-weight: 700;
        }
        h2, h3 {
            color: #2c3e50;
            font-weight: 600;
        }

        /* Metric containers */
        .metric-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        /* Button styling */
        .stButton > button {
            width: 100%;
            border-radius: 0.75rem;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem;
            transition: all 0.3s ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        }

        /* Slider styling */
        .stSlider > div > div > div > div {
            background-color: #667eea;
        }

        /* Selectbox styling */
        .stSelectbox > div > div > div {
            border-radius: 0.5rem;
            border: 2px solid #e1e5e9;
        }

        /* Info boxes */
        .stInfo {
            background-color: #e8f4f8;
            border: 1px solid #bee5eb;
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* Success boxes */
        .stSuccess {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* Warning boxes */
        .stWarning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* Error boxes */
        .stError {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            border-radius: 0.75rem;
            padding: 1rem;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }

        /* Number input center alignment */
        .stNumberInput > div > div > input {
            text-align: center;
            font-weight: 600;
        }

        /* DataFrame styling */
        .dataframe {
            border-radius: 0.5rem;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }

        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            font-weight: 600;
        }

        /* Hide Streamlit menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

    try:
        main()
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.markdown("🔧 Please check your setup and configuration, then try again.")
        st.markdown("💡 **Common Issues:**")
        st.markdown("- Ensure config.yaml exists and is properly formatted")
        st.markdown("- Check that output directories contain evaluation results")
        st.markdown("- Verify dataset paths are accessible")