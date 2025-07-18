import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom utility functions
from utils import calculate_cumulative_histogram, find_best_epoch

# --- Page Configuration ---
st.set_page_config(page_title="XAI Lung Segmentation Analysis", page_icon="🫁", layout="wide")

# --- Constants ---
OUTPUT_DIR = "/home/mohaisen_mohammed/xai/outputs/"
DATASET_ROOT_DIR = "/home/mohaisen_mohammed/Datasets/"


# --- Helper Functions ---
def parse_run_name(run_name):
    parts = run_name.replace("unet_", "").split('_')
    return {"dataset": parts[0].title(), "data_size": parts[1].title(), "epochs": parts[2].title()}


# --- Data Loading Functions (with Caching) ---
@st.cache_data
def get_available_runs():
    if not os.path.isdir(OUTPUT_DIR):
        st.error(f"FATAL: Output directory not found at the specified path: {OUTPUT_DIR}")
        return []
    return sorted([d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))])


@st.cache_data
def load_run_data(run_name, state, split):
    summary_path = os.path.join(OUTPUT_DIR, run_name, "evaluation", state, split, "_evaluation_summary.json")
    if not os.path.exists(summary_path):
        return None
    with open(summary_path, 'r') as f:
        return json.load(f)


@st.cache_data
def load_npz_data(npz_path):
    if not os.path.exists(npz_path):
        st.error(f"NPZ file not found: {npz_path}")
        return None
    try:
        with np.load(npz_path) as data:
            return {key: data[key] for key in data}
    except Exception as e:
        st.error(f"Failed to load NPZ file {npz_path}: {e}")
        return None


def get_original_image_path(image_name, dataset_name):
    base_path = "MontgomeryDataset/CXR_png/" if dataset_name.lower() == 'montgomery' else "JSRT/images/"
    return os.path.join(DATASET_ROOT_DIR, base_path, image_name)


# --- UI Rendering Functions ---
def display_column(column_id, available_runs):
    st.header(f"Analysis Column {column_id}")
    st.sidebar.subheader(f"Column {column_id} Controls")

    selected_run = st.sidebar.selectbox(
        "1. Select Model Run", available_runs, key=f"run_{column_id}",
        format_func=lambda name: " ".join(parse_run_name(name).values())
    )

    if not selected_run:
        st.warning("Please select a model run.")
        return None

    # For the main display, we will show the "good_fitting" state by default
    # The intra-column plot will show all three states.
    selected_state_for_display = "good_fitting"

    selected_split = st.sidebar.selectbox("2. Select Data Subset", ["test", "validation", "training"],
                                          key=f"split_{column_id}")

    # Load reference data to populate image list
    with st.spinner("Loading image list..."):
        reference_data = load_run_data(selected_run, "good_fitting", selected_split)

    if not reference_data:
        st.error(f"Could not load reference data for run '{selected_run}' on split '{selected_split}'.")
        return None

    sorted_results = sorted(reference_data.get("per_sample_results", []), key=lambda x: x.get("dice_score", 0),
                            reverse=True)

    if not sorted_results:
        st.warning("No per-sample results found in the loaded data.")
        return None

    def format_image_name(item):
        return f"{item['image_name']} (Dice: {item['dice_score']:.3f})"

    selected_image_formatted = st.sidebar.selectbox("3. Select Image", sorted_results, key=f"image_{column_id}",
                                                    format_func=format_image_name)
    st.sidebar.divider()

    if not selected_image_formatted:
        st.warning("No images available to select in this subset.")
        return None

    selected_image_name = selected_image_formatted['image_name']

    # --- Display data for the "good_fitting" state as the main view ---
    image_data = selected_image_formatted
    with st.spinner("Loading XAI maps..."):
        npz_data = load_npz_data(image_data["xai_results_path"])

    if not npz_data:
        st.error(f"Failed to load XAI data for {selected_image_name}")
        return None

    parsed_name = parse_run_name(selected_run)
    log_file = os.path.join(OUTPUT_DIR, selected_run, "training_log.csv")
    best_epoch_num = find_best_epoch(log_file) if os.path.exists(log_file) else "N/A"

    st.subheader(f"{parsed_name['dataset']} | {parsed_name['data_size']} | {parsed_name['epochs']}")
    st.write(f"**Displaying State:** `Good Fitting (Epoch {best_epoch_num})` on **{selected_split.upper()}** set")
    st.write(f"**Image:** `{selected_image_name}`")

    c1, c2, c3 = st.columns(3)
    c1.metric("Dice Score (Sample)", f"{image_data['dice_score']:.4f}")
    c2.metric("IoU Score (Sample)", f"{image_data['iou_score']:.4f}")
    c3.metric("Mean Uncertainty", f"{np.mean(npz_data['uncertainty_map']):.4f}")

    original_image_path = get_original_image_path(selected_image_name, parsed_name['dataset'])
    original_image = Image.open(original_image_path) if os.path.exists(original_image_path) else Image.new('L',
                                                                                                           (256, 256),
                                                                                                           color='gray')

    gt_mask, pred_mask = npz_data['ground_truth'], (npz_data['prediction'] > 0.5)

    c1, c2, c3 = st.columns(3)
    c1.image(original_image, caption="Original X-ray", use_container_width=True)
    c2.image(gt_mask, caption="Ground Truth", use_container_width=True, clamp=True)
    c3.image(pred_mask, caption="Model Prediction", use_container_width=True, clamp=True)

    # --- RESTORED: Intra-Column State Comparison Plot ---
    with st.expander("Show Fitting State Comparison Plots"):
        fitting_states = ["underfitting", "good_fitting", "overfitting"]
        state_hist_data = []

        y_coord, x_coord = 128, 128  # Fixed center point for comparison

        for state in fitting_states:
            run_data_state = load_run_data(selected_run, state, selected_split)
            if run_data_state:
                image_data_state = next((item for item in run_data_state["per_sample_results"] if
                                         item["image_name"] == selected_image_name), None)
                if image_data_state:
                    npz_data_state = load_npz_data(image_data_state["xai_results_path"])
                    if npz_data_state:
                        hist_data, median_dist, _ = calculate_cumulative_histogram(npz_data_state['ig_map'],
                                                                                   (y_coord, x_coord))
                        state_hist_data.append(
                            {"state": state.title(), "hist_data": hist_data, "median_dist": median_dist})

        if state_hist_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            palette = sns.color_palette()
            for i, item in enumerate(state_hist_data):
                color = palette[i % len(palette)]
                sns.lineplot(x=item["hist_data"].index, y=item["hist_data"].values, ax=ax,
                             label=f"{item['state']} (Median: {item['median_dist']:.2f})", color=color)
                ax.axvline(x=item["median_dist"], color=color, linestyle='--')

            ax.set_title(f"Fitting State Comparison for {selected_image_name}")
            ax.set_xlabel("Distance from Center (pixels)");
            ax.set_ylabel("Normalized Cumulative Score")
            ax.legend();
            ax.grid(True);
            st.pyplot(fig)

    # Return data for the main combined plot (comparing the good_fitting state across columns)
    hist_data, median_dist, _ = calculate_cumulative_histogram(npz_data['ig_map'], (128, 128))
    return {
        "run_name": f"Col {column_id}: {parsed_name['dataset']} {parsed_name['data_size']} {parsed_name['epochs']}",
        "hist_data": hist_data,
        "median_dist": median_dist
    }


# --- Main App ---
st.title("🫁 Comparative XAI Analysis for Lung Segmentation")
st.sidebar.title("⚙️ Controls")

available_runs = get_available_runs()
if not available_runs:
    st.error(f"No evaluation results found in {OUTPUT_DIR}. Please run the scripts first.")
    st.stop()

c1, c2, c3 = st.columns(3, gap="large")
all_column_data = []
with c1: all_column_data.append(display_column(1, available_runs))
with c2: all_column_data.append(display_column(2, available_runs))
with c3: all_column_data.append(display_column(3, available_runs))
all_column_data = [d for d in all_column_data if d is not None]

st.divider()
st.header("Combined Model Comparison (Good Fitting State)")
if all_column_data:
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = sns.color_palette()
    for i, item in enumerate(all_column_data):
        color = palette[i % len(palette)]
        sns.lineplot(x=item["hist_data"].index, y=item["hist_data"].values, ax=ax, marker='o', label=item["run_name"],
                     color=color)
        ax.axvline(x=item["median_dist"], color=color, linestyle='--', label=f'_nolegend_')
    ax.set_title("Comparison of Cumulative Weighted Attributions")
    ax.set_xlabel("Distance (pixels)")
    ax.set_ylabel("Normalized Score")
    ax.grid(True)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model Run")
    st.pyplot(fig)
