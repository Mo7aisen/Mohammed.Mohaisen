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
st.set_page_config(
    page_title="Lung Segmentation Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Path Configuration ---
# Construct absolute paths from the script's location for robustness.
try:
    # This assumes app.py is in the project's root directory (e.g., /XAI/)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for environments where __file__ is not defined
    PROJECT_ROOT = os.getcwd()

RESULTS_DIR = os.path.join(PROJECT_ROOT, "precomputed_results")
XAI_RESULTS_DIR = os.path.join(RESULTS_DIR, "xai_analysis")
AL_RESULTS_DIR = os.path.join(RESULTS_DIR, "active_learning")
DATASET_ROOT_DIR = "/home/mohaisen_mohammed/Datasets/"

# --- FIX: Using the correct, user-provided paths for the ground truth masks ---
DATASET_PATHS = {
    "montgomery": {
        "images": os.path.join(DATASET_ROOT_DIR, "MontgomeryDataset/CXR_png/"),
        "masks": "/home/mohaisen_mohammed/Datasets/montgomery_prepared/masks/",
    },
    "jsrt": {
        "images": os.path.join(DATASET_ROOT_DIR, "JSRT/images/"),
        "masks": "/home/mohaisen_mohammed/Datasets/jsrt_prepared/masks/",
    }
}

# --- Initial Sanity Checks ---
# Perform checks at the start to ensure critical directories exist.
if not os.path.isdir(XAI_RESULTS_DIR):
    st.error(
        f"FATAL ERROR: XAI results directory not found at the expected path: `{XAI_RESULTS_DIR}`. Please check your folder structure.")
    st.stop()
if not os.path.isdir(DATASET_ROOT_DIR):
    st.error(
        f"FATAL ERROR: The main `Datasets` directory not found at the expected path: `{DATASET_ROOT_DIR}`. Please check your folder structure.")
    st.stop()


# --- Data Loading Functions (Cached for performance) ---

@st.cache_data
def get_available_models_and_datasets():
    """Scans the results directory to find all available models and their corresponding dataset results."""
    models = {}
    for model_folder in os.listdir(XAI_RESULTS_DIR):
        model_path = os.path.join(XAI_RESULTS_DIR, model_folder)
        if os.path.isdir(model_path):
            datasets = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            if datasets:
                models[model_folder] = sorted(datasets)
    return models


# --- FIX: Removed @st.cache_data to prevent stale data from being shown ---
def load_xai_data_for_selection(_model_name, _dataset_name):
    """Loads all XAI .npz data and creates clean, extension-less identifiers."""
    data_path = os.path.join(XAI_RESULTS_DIR, _model_name, _dataset_name)
    if not os.path.isdir(data_path): return [], []
    all_files = glob.glob(os.path.join(data_path, "*.npz"))
    if not all_files: return [], []
    xai_data, image_names = [], []
    for f_path in sorted(all_files):
        try:
            with np.load(f_path, allow_pickle=True) as data:
                xai_data.append(dict(data))
                base_name = os.path.basename(f_path)
                clean_identifier = base_name.replace('.png.npz', '')
                image_names.append(clean_identifier)
        except Exception as e:
            st.warning(f"Could not load file {os.path.basename(f_path)}: {e}")
    return image_names, xai_data


# --- FIX: Removed @st.cache_data to prevent stale data from being shown ---
def load_al_data(_model_name, _dataset_name):
    """Loads and cleans the active learning JSON data."""
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
            df = pd.DataFrame(data.get("ranked_images", []))
            if not df.empty and 'image' in df.columns:
                df['image'] = df['image'].str.replace('.png', '', regex=False)
            return df
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"Error parsing active learning file '{al_path}': {e}")
        return None


def get_image_path(image_identifier, dataset_name, image_type='images'):
    """Constructs the full path to an image or mask from a clean base identifier."""
    try:
        file_name = f"{image_identifier}.png"
        base_path = DATASET_PATHS[dataset_name][image_type]
        return os.path.join(base_path, file_name)
    except KeyError:
        st.error(f"Dataset '{dataset_name}' or image type '{image_type}' not found in configuration.")
        return None


# --- UI Rendering Functions ---

def display_xai_analysis(xai_data, image_names, selected_image_name, selected_dataset):
    st.header("🔬 Explainable AI (XAI) Analysis")
    if not image_names:
        st.warning("No XAI data found for this selection.")
        return
    try:
        selected_index = image_names.index(selected_image_name)
    except ValueError:
        st.error("Selected image not found in the loaded data. Please refresh.")
        return
    data = xai_data[selected_index]
    st.subheader(f"Displaying: `{selected_image_name}`")
    correlation = data.get('correlation')
    st.metric("Correlation (Spatial vs. Uncertainty)",
              f"{correlation:.4f}" if correlation is not None and not np.isnan(correlation) else "N/A")
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
            st.image(Image.new('L', (256, 256), 0), caption=f"Mask Not Found", use_container_width=True)
    st.divider()
    col4, col5 = st.columns(2)

    def create_heatmap(data_array, title):
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(data_array, ax=ax, cbar=True, xticklabels=False, yticklabels=False, cmap='viridis', square=True)
        ax.set_title(title, fontsize=14)
        plt.tight_layout()
        return fig

    with col4:
        st.pyplot(create_heatmap(data.get('spatial_map', np.zeros((256, 256))), "Spatial Attributions"),
                  use_container_width=True)
    with col5:
        st.pyplot(create_heatmap(data.get('uncertainty_map', np.zeros((256, 256))), "Uncertainty Map"),
                  use_container_width=True)


def display_active_learning_insights(al_df, selected_dataset, top_n):
    """Renders the Active Learning Insights view with improved UI."""
    st.header("🧠 Active Learning Insights")
    st.info(
        "This view ranks images by an 'uncertainty score'. Images with higher scores are those the model is most unsure about, making them prime candidates for manual review and inclusion in future training rounds.")

    if al_df is None or al_df.empty:
        st.warning("No Active Learning data found for this selection.")
        return

    # --- Improved Dataframe Display ---
    with st.expander("View Ranked Images Dataframe", expanded=True):
        df_display = al_df.copy()
        # Rename columns for clarity
        df_display.rename(columns={
            'image': 'Image ID',
            'uncertainty_score': 'Uncertainty Score',
            'loss_score': 'Dice Loss'
        }, inplace=True)

        # Apply styling: No background color, and container width is not used to keep it compact.
        st.dataframe(df_display.head(top_n).style.format({
            'Uncertainty Score': '{:.4f}',
            'Dice Loss': '{:.4f}'
        }).set_properties(**{'text-align': 'left'}), use_container_width=False)

    st.divider()
    st.subheader(f"Visualizing Top {top_n} Most Uncertain Images")

    # --- Image Gallery Display ---
    cols = st.columns(5)
    for i in range(top_n):
        if i < len(al_df):
            row = al_df.iloc[i]
            image_name = row['image']
            img_path = get_image_path(image_name, selected_dataset, image_type='images')
            col = cols[i % 5]
            if img_path and os.path.exists(img_path):
                image = Image.open(img_path).convert("L")
                with col:
                    st.image(image, caption=f"#{i + 1}: {image_name}", use_container_width=True)
                    st.metric(label="Uncertainty", value=f"{row['uncertainty_score']:.4f}")
            else:
                col.warning(f"Image not found: {image_name}")


# --- Main Application Logic ---
def main():
    st.title("🫁 Lung Segmentation Analysis Dashboard")
    st.markdown("Use the sidebar to select a model and dataset, then choose a view to explore the results.")
    st.sidebar.title("⚙️ Controls")

    available_models = get_available_models_and_datasets()
    if not available_models:
        st.sidebar.error("No models found. Please check the `precomputed_results/xai_analysis` directory.")
        st.stop()

    selected_model = st.sidebar.selectbox(
        "1. Select Model",
        options=sorted(available_models.keys()),
        key='model_selector'
    )

    if selected_model:
        available_datasets = available_models.get(selected_model, [])
        if available_datasets:
            selected_dataset = st.sidebar.selectbox(
                "2. Select Dataset",
                options=available_datasets,
                key=f'dataset_selector_{selected_model}'
            )

            st.sidebar.divider()
            view_mode = st.sidebar.radio("3. Select Analysis View", ('XAI Analysis', 'Active Learning Insights'),
                                         horizontal=True, key=f"view_mode_{selected_model}_{selected_dataset}")
            st.sidebar.divider()

            # --- UI IMPROVEMENT: Conditional controls ---
            if view_mode == 'XAI Analysis':
                # Data loading is no longer cached, so it will re-run every time
                image_names, xai_data = load_xai_data_for_selection(selected_model, selected_dataset)
                st.sidebar.markdown(f"**Images Found:** `{len(image_names)}`")
                st.sidebar.markdown("---")
                if image_names:
                    selected_image_name = st.sidebar.selectbox(
                        "4. Select Image to Analyze",
                        options=image_names,
                        key=f"img_selector_{selected_model}_{selected_dataset}"
                    )
                    display_xai_analysis(xai_data, image_names, selected_image_name, selected_dataset)
                else:
                    st.header("🔬 Explainable AI (XAI) Analysis")
                    st.warning("No XAI data found for this model and dataset combination.")

            elif view_mode == 'Active Learning Insights':
                al_data = load_al_data(selected_model, selected_dataset)
                st.sidebar.markdown(f"**Images Found:** `{len(al_data) if al_data is not None else 0}`")
                st.sidebar.markdown("---")
                if al_data is not None and not al_data.empty:
                    # --- UI IMPROVEMENT: Number input instead of slider ---
                    top_n = st.sidebar.number_input(
                        "4. Number of Images to Display",
                        min_value=1,
                        max_value=min(50, len(al_data)),  # Set a reasonable max
                        value=10,
                        step=1,
                        key=f"al_number_{selected_model}_{selected_dataset}"
                    )
                    display_active_learning_insights(al_data, selected_dataset, top_n)
                else:
                    st.header("🧠 Active Learning Insights")
                    st.warning("No Active Learning data found for this model and dataset combination.")
        else:
            st.sidebar.warning("No datasets found for the selected model.")
    else:
        st.sidebar.info("Please select a model to begin.")


if __name__ == "__main__":
    main()
