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
    """
    Calculate weighted median distance based on attribution values
    Enhanced version based on supervisor's feedback
    """
    attr_z = attr_map.copy()

    # Set thresholds if not provided
    if min_abs_attr is None:
        min_abs_attr = np.percentile(np.abs(attr_z), 10)  # Bottom 10% threshold
    if max_abs_attr is None:
        max_abs_attr = np.percentile(np.abs(attr_z), 90)  # Top 90% threshold

    # Apply thresholding as suggested
    selected_mask = (np.abs(attr_z) >= min_abs_attr) & (np.abs(attr_z) <= max_abs_attr)

    if np.any(selected_mask):
        selected_indices = np.where(selected_mask)

        # Calculate distances from center point
        distances = np.sqrt((selected_indices[0] - center_y) ** 2 + (selected_indices[1] - center_x) ** 2)

        # Use absolute attribution values as weights
        weights = np.abs(attr_z[selected_mask])

        if len(distances) > 0:
            # Sort by distance
            sorted_idx = np.argsort(distances)
            sorted_distances = distances[sorted_idx]
            sorted_weights = weights[sorted_idx]

            # Calculate cumulative weights
            cumulative_weights = np.cumsum(sorted_weights)
            total_weight = cumulative_weights[-1] if len(cumulative_weights) > 0 else 0

            if total_weight > 0:
                # Find weighted median (50% of total weight)
                median_idx = np.where(cumulative_weights >= 0.5 * total_weight)[0]
                if len(median_idx) > 0:
                    median_distance = sorted_distances[median_idx[0]]
                    return median_distance, sorted_distances, cumulative_weights, total_weight

    return None, None, None, None


def calculate_enhanced_cumulative_histogram(attr_map, center_x, center_y):
    """
    Enhanced cumulative histogram calculation based on supervisor's feedback
    """
    # Get all pixels and their distances/weights
    y_coords, x_coords = np.mgrid[0:attr_map.shape[0], 0:attr_map.shape[1]]
    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    weights = np.abs(attr_map)

    # Flatten arrays
    distances_flat = distances.flatten()
    weights_flat = weights.flatten()

    # Remove zero weights to avoid noise
    non_zero_mask = weights_flat > 0
    distances_clean = distances_flat[non_zero_mask]
    weights_clean = weights_flat[non_zero_mask]

    if len(distances_clean) == 0:
        return None, None

    # Sort by distance
    sorted_idx = np.argsort(distances_clean)
    sorted_distances = distances_clean[sorted_idx]
    sorted_weights = weights_clean[sorted_idx]

    # Calculate cumulative weights
    cumulative_weights = np.cumsum(sorted_weights)

    # Normalize to [0, 1]
    if cumulative_weights[-1] > 0:
        normalized_cumulative = cumulative_weights / cumulative_weights[-1]
    else:
        normalized_cumulative = cumulative_weights

    # Create histogram with distance bins
    max_dist = int(np.ceil(sorted_distances.max()))
    distance_bins = np.arange(0, max_dist + 1)

    # Bin the cumulative values
    binned_cumulative = []
    for dist in distance_bins:
        # Find cumulative value at this distance
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

    # Fallback paths
    if dataset_name.lower() == 'montgomery':
        return os.path.join("/home/mohaisen_mohammed/Datasets/MontgomeryDataset/CXR_png/", image_name)
    else:  # jsrt
        return os.path.join("/home/mohaisen_mohammed/Datasets/JSRT/images/", image_name)


def plot_training_curves(training_df, title):
    """Plot training and validation loss curves"""
    fig = go.Figure()

    # Add training loss
    fig.add_trace(go.Scatter(
        x=training_df['epoch'],
        y=training_df['train_loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='blue', width=2),
        marker=dict(size=3)
    ))

    # Add validation loss
    fig.add_trace(go.Scatter(
        x=training_df['epoch'],
        y=training_df['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='red', width=2),
        marker=dict(size=3)
    ))

    # Mark best epoch
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


def display_enhanced_attribution_map(attr_map, center_x, center_y, title, image_size=(256, 256)):
    """
    Display enhanced attribution map with grid centers and better visualization
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display attribution map with RdBu_r colormap (red-blue)
    im = ax.imshow(attr_map, cmap='RdBu_r', interpolation='nearest')

    # Add crosshair at center point
    ax.axhline(y=center_y, color='yellow', linestyle='--', alpha=0.9, linewidth=2)
    ax.axvline(x=center_x, color='yellow', linestyle='--', alpha=0.9, linewidth=2)
    ax.plot(center_x, center_y, 'yo', markersize=10, markerfacecolor='yellow',
            markeredgecolor='black', markeredgewidth=2)

    # Add grid to show cell centers (as requested by supervisor)
    grid_spacing = 16  # Show grid every 16 pixels
    for i in range(0, image_size[0], grid_spacing):
        ax.axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
    for j in range(0, image_size[1], grid_spacing):
        ax.axvline(x=j, color='white', alpha=0.3, linewidth=0.5)

    ax.set_title(f"{title}\nAnalysis Point: ({center_x}, {center_y})", fontsize=14, pad=20)
    ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

    # Add colorbar with better formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Attribution Score', rotation=270, labelpad=20, fontsize=12)

    # Set axis limits and ticks
    ax.set_xlim(0, image_size[1])
    ax.set_ylim(image_size[0], 0)  # Invert y-axis for image coordinates

    return fig


def display_enhanced_uncertainty_map(uncertainty_map, title, image_size=(256, 256)):
    """
    Display enhanced uncertainty map synchronized with attribution map size
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Display uncertainty map
    im = ax.imshow(uncertainty_map, cmap='viridis', interpolation='nearest')

    # Add grid to match attribution map (synchronized as requested)
    grid_spacing = 16
    for i in range(0, image_size[0], grid_spacing):
        ax.axhline(y=i, color='white', alpha=0.3, linewidth=0.5)
    for j in range(0, image_size[1], grid_spacing):
        ax.axvline(x=j, color='white', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Uncertainty Score', rotation=270, labelpad=20, fontsize=12)

    # Set axis limits
    ax.set_xlim(0, image_size[1])
    ax.set_ylim(image_size[0], 0)

    return fig


def display_image_with_crosshair(image, center_x, center_y, title):
    """Display image with crosshair at specified point"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap='gray')

    # Add crosshair
    ax.axhline(y=center_y, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(x=center_x, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax.plot(center_x, center_y, 'ro', markersize=10, markerfacecolor='red',
            markeredgecolor='white', markeredgewidth=2)

    ax.set_title(f"{title}\nCenter Point: ({center_x}, {center_y})", fontsize=14, pad=20)
    ax.set_xlabel("X Coordinate (pixels)", fontsize=12)
    ax.set_ylabel("Y Coordinate (pixels)", fontsize=12)

    return fig


def display_column(column_id, available_runs):
    """Display analysis column with enhanced features"""
    st.markdown(f"### Analysis Column {column_id}")

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

    # Image selection with enhanced display (showing Dice scores as requested)
    results = run_data.get("per_sample_results", [])
    if not results:
        st.warning("No sample results found.")
        return None

    # Sort by dice score for better selection
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

    # Display individual image Dice score prominently (as requested by supervisor)
    st.metric("Selected Image Dice Score", f"{selected_image_data['dice_score']:.4f}")

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

    # ROW 1: Original X-ray and Ground Truth (2 images only)
    st.markdown("**Original Image and Ground Truth:**")
    col1, col2 = st.columns(2)

    try:
        original_image_path = get_original_image_path(
            selected_image_data['image_name'],
            parsed_name['dataset']
        )

        if safe_file_check(original_image_path):
            original_image = Image.open(original_image_path)
        else:
            original_image = Image.new('L', (256, 256), color=128)
            st.warning(f"Original image not found")

        gt_mask = npz_data['ground_truth']

        with col1:
            st.image(original_image, caption="Original X-ray", use_container_width=True)

        with col2:
            st.image(gt_mask, caption="Ground Truth", use_container_width=True, clamp=True)

    except Exception as e:
        st.error(f"Error displaying images: {e}")

    # ROW 2: Enhanced Uncertainty Map
    st.markdown("**Uncertainty Analysis:**")
    try:
        if 'uncertainty_map' in npz_data:
            fig = display_enhanced_uncertainty_map(
                npz_data['uncertainty_map'],
                "Prediction Uncertainty Map"
            )
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Uncertainty map not available")
    except Exception as e:
        st.error(f"Error displaying uncertainty map: {e}")

    # ROW 3: Enhanced Interactive Point Selection
    st.markdown("**Interactive Analysis Point Selection:**")

    # Enhanced coordinate controls with +/- buttons (as requested)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        if st.button("X-", key=f"x_minus_{column_id}"):
            if f"x_{column_id}" in st.session_state:
                st.session_state[f"x_{column_id}"] = max(0, st.session_state[f"x_{column_id}"] - 1)

    with col2:
        x_coord = st.number_input("X", min_value=0, max_value=255, value=128, key=f"x_{column_id}")

    with col3:
        if st.button("X+", key=f"x_plus_{column_id}"):
            if f"x_{column_id}" in st.session_state:
                st.session_state[f"x_{column_id}"] = min(255, st.session_state[f"x_{column_id}"] + 1)

    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 2])

    with col1:
        if st.button("Y-", key=f"y_minus_{column_id}"):
            if f"y_{column_id}" in st.session_state:
                st.session_state[f"y_{column_id}"] = max(0, st.session_state[f"y_{column_id}"] - 1)

    with col2:
        y_coord = st.number_input("Y", min_value=0, max_value=255, value=128, key=f"y_{column_id}")

    with col3:
        if st.button("Y+", key=f"y_plus_{column_id}"):
            if f"y_{column_id}" in st.session_state:
                st.session_state[f"y_{column_id}"] = min(255, st.session_state[f"y_{column_id}"] + 1)

    with col5:
        show_ig_button = st.button("Show Integrated Gradients Analysis", key=f"ig_{column_id}")

    # Display original image with crosshair
    try:
        fig = display_image_with_crosshair(original_image, x_coord, y_coord, "Analysis Point Selection")
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Error displaying crosshair image: {e}")

    # Enhanced Integrated Gradients Analysis
    if show_ig_button:
        st.markdown("**Enhanced Integrated Gradients Analysis:**")
        try:
            if 'ig_map' in npz_data:
                # Display enhanced IG map
                fig = display_enhanced_attribution_map(
                    npz_data['ig_map'],
                    x_coord, y_coord,
                    "Integrated Gradients Attribution Map"
                )
                st.pyplot(fig)
                plt.close()

                # Calculate enhanced weighted median distance
                median_dist, sorted_distances, cumulative_weights, total_weight = calculate_weighted_median_distance(
                    npz_data['ig_map'], x_coord, y_coord
                )

                if median_dist is not None:
                    st.metric("Weighted Median Distance", f"{median_dist:.2f} pixels")

                # Enhanced cumulative histogram
                distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(
                    npz_data['ig_map'], x_coord, y_coord
                )

                if distance_bins is not None:
                    # Plot enhanced cumulative histogram
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(distance_bins, cumulative_values, marker='o', linewidth=2, markersize=4, color='blue')

                    if median_dist is not None:
                        ax.axvline(x=median_dist, color='red', linestyle='--', linewidth=2,
                                   label=f'Weighted Median: {median_dist:.2f}px')

                    ax.set_title(f"Enhanced Cumulative Weighted Attribution\nFrom Point ({x_coord}, {y_coord})",
                                 fontsize=14)
                    ax.set_xlabel("Distance from Center (pixels)", fontsize=12)
                    ax.set_ylabel("Normalized Cumulative Weight", fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.05)
                    ax.legend()

                    st.pyplot(fig)
                    plt.close()

            else:
                st.warning("Integrated Gradients data not available")
        except Exception as e:
            st.error(f"Error displaying Integrated Gradients: {e}")

    # ROW 4: Training Curve
    st.markdown("**Training History:**")
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
        distance_bins, cumulative_values = calculate_enhanced_cumulative_histogram(npz_data['ig_map'], x_coord, y_coord)

        return {
            "run_name": f"Col {column_id}: {parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}",
            "hist_data": {"distance_bins": distance_bins, "cumulative_values": cumulative_values},
            "state": selected_state,
            "epoch_used": epoch_used,
            "center_point": (x_coord, y_coord),
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
                    line=dict(color=color, dash='solid'),
                    marker=dict(size=3)
                ))

                # Plot validation curve
                fig.add_trace(go.Scatter(
                    x=training_df['epoch'],
                    y=training_df['val_loss'],
                    mode='lines+markers',
                    name=f"{label} (Val)",
                    line=dict(color=color, dash='dash'),
                    marker=dict(size=3)
                ))

        except Exception as e:
            st.error(f"Error processing {run_name}: {e}")

    fig.update_layout(
        title="Training Progress Comparison",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode='x unified',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


# --- Main App ---
def main():
    st.title("🫁 Enhanced XAI Analysis for Lung Segmentation")
    st.markdown("*Interactive analysis with enhanced attribution mapping and weighted median distance calculation*")
    st.markdown("---")

    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")

    page = st.sidebar.radio(
        "Choose Analysis:",
        ["🔬 Comparative Analysis", "📈 Training Analysis"]
    )

    if page == "🔬 Comparative Analysis":
        st.header("🔬 Enhanced Comparative XAI Analysis")

        available_runs = get_available_runs()
        if not available_runs:
            st.error("No evaluation results found. Please run experiments first.")
            return

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
            st.subheader("🔗 Enhanced Combined Model Comparison")

            try:
                fig = go.Figure()
                colors = px.colors.qualitative.Set1

                for i, item in enumerate(all_column_data):
                    color = colors[i % len(colors)]
                    hist_data = item["hist_data"]

                    if hist_data["distance_bins"] is not None and hist_data["cumulative_values"] is not None:
                        # Main attribution curve
                        fig.add_trace(go.Scatter(
                            x=hist_data["distance_bins"],
                            y=hist_data["cumulative_values"],
                            mode='lines+markers',
                            name=f'{item["run_name"]} ({item["state"]})',
                            line=dict(color=color, width=2),
                            marker=dict(size=4)
                        ))

                fig.update_layout(
                    title="Enhanced Comparison of Cumulative Weighted Attributions",
                    xaxis_title="Distance from Center (pixels)",
                    yaxis_title="Normalized Cumulative Weight",
                    hovermode='x unified',
                    height=500,
                    yaxis=dict(range=[0, 1.05])
                )

                st.plotly_chart(fig, use_container_width=True)

                # Enhanced summary statistics table
                st.subheader("📊 Enhanced Comparison Statistics")
                stats_data = []
                for item in all_column_data:
                    stats_data.append({
                        "Model": item["run_name"],
                        "State": item["state"].replace('_', ' ').title(),
                        "Analysis Point": f"({item['center_point'][0]}, {item['center_point'][1]})",
                        "Image Dice Score": f"{item['dice_score']:.4f}",
                        "Epoch Used": item.get("epoch_used", "N/A")
                    })

                st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

            except Exception as e:
                st.error(f"Error creating comparison plot: {e}")

        elif len(all_column_data) == 1:
            st.info("💡 Select models in multiple columns to see comparative analysis.")

    elif page == "📈 Training Analysis":
        display_training_analysis()


if __name__ == "__main__":
    # Set enhanced page style
    st.markdown("""
    <style>
        .stApp > header {
            background-color: transparent;
        }
        .main > div {
            padding-top: 1rem;
        }
        h1 {
            color: #1f77b4;
        }
        .metric-container {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .stExpander {
            border: 1px solid #e6e6e6;
            border-radius: 0.5rem;
        }
        .stButton > button {
            width: 100%;
            border-radius: 0.5rem;
        }
        .stNumberInput > div > div > input {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    try:
        main()
    except Exception as e:
        st.error(f"Application error: {e}")
        st.markdown("Please check your setup and try again.")
