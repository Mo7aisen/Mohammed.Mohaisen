import streamlit as st
import os
import glob
import numpy as np
import json
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Configuration ---
# Set up the page with a title, icon, and wide layout for better data visualization.
st.set_page_config(
    page_title="Lung Segmentation Analysis",
    page_icon="ðŸ«",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Configuration ---
RESULTS_DIR = "./precomputed_results/"
XAI_RESULTS_DIR = os.path.join(RESULTS_DIR, "xai_analysis")
AL_RESULTS_DIR = os.path.join(RESULTS_DIR, "active_learning")
DATASET_PATHS = {
    "montgomery": {
        "images": "Datasets/MontgomeryDataset/CXR_png/",
        "masks": "Datasets/MontgomeryDataset/ManualMask/combined/",
    },
    "jsrt": {
        "images": "Datasets/JSRT/images/",
        "masks": "Datasets/JSRT/masks_png/combined/",
    }
}

# --- Data Loading Functions (Cached for performance) ---
# These functions are cached to avoid reloading data from disk on every interaction.

@st.cache_data
def get_available_models_and_datasets():
    """Scans the results directory to find all available models and their corresponding dataset results."""
    models = {}
    if not os.path.isdir(XAI_RESULTS_DIR):
        st.error(f"XAI results directory not found at: {XAI_RESULTS_DIR}")
        return models
    for model_folder in os.listdir(XAI_RESULTS_DIR):
        model_path = os.path.join(XAI_RESULTS_DIR, model_folder)
        if os.path.isdir(model_path):
            datasets = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if datasets:
                models[model_folder] = sorted(datasets)
    return models

@st.cache_data
def load_xai_data_for_selection(_model_name, _dataset_name):
    """Loads all XAI .npz data for a given model and dataset selection."""
    data_path = os.path.join(XAI_RESULTS_DIR, _model_name, _dataset_name)
    if not os.path.isdir(data_path): return [], []
    all_files = glob.glob(os.path.join(data_path, "*.npz"))
    if not all_files: return [], []
    xai_data, image_names = [], []
    for f_path in sorted(all_files):
        try:
            with np.load(f_path, allow_pickle=True) as data:
                xai_data.append(dict(data))
                image_names.append(os.path.basename(f_path).replace('.npz', ''))
        except Exception as e:
            st.warning(f"Could not load file {os.path.basename(f_path)}: {e}")
    return image_names, xai_data

@st.cache_data
def load_al_data(_model_name, _dataset_name):
    """Loads the active learning JSON data for a given model and dataset."""
    al_filename = f"{_model_name}_{_dataset_name}.json"
    al_path = os.path.join(AL_RESULTS_DIR, al_filename)
    if not os.path.exists(al_path):
        al_filename_alt = f"{_model_name.replace('.pth', '')}_{_dataset_name}.json"
        al_path_alt = os.path.join(AL_RESULTS_DIR, al_filename_alt)
        if not os.path.exists(al_path_alt): return None
        al_path = al_path_alt
    try:
        with open(al_path, 'r') as f:
            data = json.load(f)
            return pd.DataFrame(data.get("ranked_images", []))
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error parsing active learning file '{al_path}': {e}")
        return None

def get_image_path(image_name, dataset_name, image_type='images'):
    """Constructs the full path to an original image or a mask."""
    try:
        base_path = DATASET_PATHS[dataset_name][image_type]
        return os.path.join(base_path, image_name)
    except KeyError:
        return None

# --- UI Rendering Functions ---

def display_xai_analysis(xai_data, image_names, selected_image_name, selected_dataset):
    """Renders the XAI Analysis view."""
    st.header("ðŸ”¬ Explainable AI (XAI) Analysis")
    if not image_names:
        st.warning("No XAI data found for this selection.")
        return

    selected_index = image_names.index(selected_image_name)
    data = xai_data[selected_index]
    st.subheader(f"Displaying: `{selected_image_name}`")

    correlation = data.get('correlation')
    st.metric("Correlation (Spatial vs. Uncertainty)", f"{correlation:.4f}" if correlation is not None and not np.isnan(correlation) else "N/A")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(data.get('original_np', []), caption="Original Image", use_container_width=True)
    with col2:
        st.image(data.get('mean_pred', []), caption="Model Prediction", clamp=True, use_container_width=True)
    with col3:
        mask_path = get_image_path(selected_image_name, selected_dataset, image_type='masks')
        if mask_path and os.path.exists(mask_path):
            mask_img = Image.open(mask_path).convert("L").resize((256, 256))
            st.image(mask_img, caption="Ground Truth Mask", use_container_width=True)
        else:
            st.image(Image.new('L', (256, 256), 0), caption="Mask Not Found", use_container_width=True)

    st.divider()
    col4, col5 = st.columns(2)
    def create_heatmap(data_array, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(data_array, ax=ax, cbar=True, xticklabels=False, yticklabels=False, cmap='viridis', square=True)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig
    with col4:
        st.pyplot(create_heatmap(data.get('spatial_map', np.zeros((256, 256))), "Spatial Attributions"), use_container_width=True)
    with col5:
        st.pyplot(create_heatmap(data.get('uncertainty_map', np.zeros((256, 256))), "Uncertainty Map"), use_container_width=True)

def display_active_learning_insights(al_df, selected_dataset, top_n):
    """Renders the Active Learning Insights view."""
    st.header("ðŸ¤– Active Learning Insights")
    if al_df is None or al_df.empty:
        st.warning("No Active Learning data found for this selection.")
        return

    with st.expander("View Ranked Images Dataframe"):
        st.dataframe(al_df)
    st.divider()

    st.subheader(f"Visualizing Top {top_n} Most Uncertain Images")
    cols = st.columns(5)
    for i in range(top_n):
        if i < len(al_df):
            row = al_df.iloc[i]
            image_name = row['image']
            img_path = get_image_path(image_name, selected_dataset, image_type='images')
            col = cols[i % 5]
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("L")
                col.image(image, caption=f"#{i+1}: {image_name.split('.')[0]}", use_container_width=True)
                col.metric(label="Uncertainty", value=f"{row['uncertainty_score']:.3f}")
            else:
                col.warning(f"Image not found: {image_name}")

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app."""
    st.title("ðŸ« Lung Segmentation Analysis Dashboard")
    st.markdown("Use the sidebar to select a model and dataset, then choose a view to explore the results.")

    # --- Sidebar Controls ---
    st.sidebar.title("âš™ï¸ Controls")
    available_models = get_available_models_and_datasets()
    if not available_models:
        st.sidebar.error("No models found. Please run the pre-computation script.")
        st.stop()

    # --- FIX: Robust State Management using Callbacks ---
    # This ensures that when a primary selection (like model) changes, all dependent selections are reset.
    def on_model_change():
        st.session_state.dataset_key = st.session_state.model_selector
        # When the model changes, we don't need to reset the dataset selector here
        # because its options will be updated automatically in the main logic.

    selected_model = st.sidebar.selectbox(
        "1. Select Model",
        options=sorted(available_models.keys()),
        key='model_selector',
        on_change=on_model_change
    )

    available_datasets = available_models[selected_model]
    selected_dataset = st.sidebar.selectbox(
        "2. Select Dataset",
        options=available_datasets,
        key=f'dataset_selector_{selected_model}' # Key is now dependent on model
    )

    st.sidebar.divider()
    view_mode = st.sidebar.radio("3. Select Analysis View", ('XAI Analysis', 'Active Learning Insights'), horizontal=True)
    st.sidebar.divider()

    # --- Data Loading with Spinner ---
    with st.spinner(f"Loading data for {selected_model} on {selected_dataset}..."):
        image_names, xai_data = load_xai_data_for_selection(selected_model, selected_dataset)
        al_data = load_al_data(selected_model, selected_dataset)

    # --- Main Panel Rendering ---
    if view_mode == 'XAI Analysis':
        if image_names:
            selected_image_name = st.sidebar.selectbox(
                "4. Select Image to Analyze",
                options=image_names,
                key=f"img_selector_{selected_model}_{selected_dataset}" # Key is dependent on both
            )
            display_xai_analysis(xai_data, image_names, selected_image_name, selected_dataset)
        else:
            st.header("ðŸ”¬ Explainable AI (XAI) Analysis")
            st.warning("No XAI data found for this model and dataset combination.")

    elif view_mode == 'Active Learning Insights':
        if al_data is not None and not al_data.empty:
            top_n = st.sidebar.slider(
                "4. Number of Images to Display",
                min_value=1, max_value=min(20, len(al_data)), value=10, step=1,
                key=f"al_slider_{selected_model}_{selected_dataset}" # Key is dependent on both
            )
            display_active_learning_insights(al_data, selected_dataset, top_n)
        else:
            st.header("ðŸ¤– Active Learning Insights")
            st.warning("No Active Learning data found for this model and dataset combination.")

if __name__ == "__main__":
    main()
