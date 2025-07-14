import os
import glob
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from PIL import Image
import json
import traceback

# Assuming these imports are in files in the same directory
# NOTE: Your LungDataset in data_loader.py must return 'image_path'
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor, DATASET_PATHS
from evaluate import generate_spatial_heatmap, run_mc_dropout_inference, calculate_dice_loss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./trained_models/"
IMG_SIZE = (256, 256)
RESULTS_DIR = "./precomputed_results/"
XAI_RESULTS_DIR = os.path.join(RESULTS_DIR, "xai_analysis")
AL_RESULTS_DIR = os.path.join(RESULTS_DIR, "active_learning")
# Increased batch size for better GPU utilization. Adjust if you get out-of-memory errors.
BATCH_SIZE = 8
NUM_WORKERS = 4  # Number of parallel workers for data loading

# Create directories to store results
os.makedirs(XAI_RESULTS_DIR, exist_ok=True)
os.makedirs(AL_RESULTS_DIR, exist_ok=True)


# --- Helper Functions ---
def enable_dropout(model):
    """Set dropout layers to train mode for MC dropout."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def load_model(model_path):
    """Loads a U-Net model and enables dropout."""
    print(f"Loading model from {model_path} onto {DEVICE}...")
    model = UNet(n_channels=1, n_classes=1, dropout_rate=0.2)
    # The 'weights_only=True' argument is best practice for security when loading PyTorch models.
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    enable_dropout(model)  # Set once after loading
    return model


def perform_active_learning_incrementally(model, dataset_name, save_path):
    """
    Performs active learning incrementally using batch processing for significant speedup.
    Resumes from existing progress.
    """
    full_dataset = LungDataset(dataset_name=dataset_name, transform=ResizeAndToTensor())

    # --- Resume Logic ---
    scores = []
    processed_files = set()
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r') as f:
                existing_data = json.load(f)
                scores = existing_data.get('ranked_images', [])
                processed_files = {item['image'] for item in scores}
            print(f"  - Resuming AL: {len(processed_files)}/{len(full_dataset)} images already processed.", flush=True)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  - AL progress file '{save_path}' is corrupt or unreadable. Starting fresh. Error: {e}",
                  flush=True)
            scores = []
            processed_files = set()

    # --- Create a dataset of only the remaining images ---
    indices_to_process = [
        i for i, path in enumerate(full_dataset.image_files)
        if os.path.basename(path) not in processed_files
    ]

    if not indices_to_process:
        print("  - All AL images have been processed.", flush=True)
        return

    print(f"  - Processing {len(indices_to_process)} remaining AL images for {dataset_name}...", flush=True)

    subset_to_process = Subset(full_dataset, indices_to_process)
    dataloader = DataLoader(subset_to_process, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    # --- FIX: Manually track paths to avoid KeyError if 'image_path' is not in batch ---
    paths_to_process = [full_dataset.image_files[i] for i in indices_to_process]
    path_idx = 0

    # --- Batch Processing Loop ---
    for batch in tqdm(dataloader, desc=f"Active Learning for {dataset_name}"):
        image_tensors = batch['image'].to(DEVICE)
        mask_tensors = batch['mask']  # Keep on CPU for numpy conversion

        # Get the paths for the current batch manually
        current_batch_size = image_tensors.size(0)
        image_paths = paths_to_process[path_idx: path_idx + current_batch_size]
        path_idx += current_batch_size

        try:
            # --- Perform all GPU operations on the entire batch at once ---
            sp_maps_batch = generate_spatial_heatmap(model, image_tensors, DEVICE, grid_step=16)
            unc_scores_batch = np.mean(sp_maps_batch, axis=(1, 2))

            with torch.no_grad():
                logits_batch = model(image_tensors)
                preds_batch = (torch.sigmoid(logits_batch) > 0.5).float().cpu().numpy()

            # --- Process results for each item in the batch ---
            for i in range(len(image_paths)):
                image_name = os.path.basename(image_paths[i])
                unc_score = float(unc_scores_batch[i])
                pred_np = np.squeeze(preds_batch[i])
                mask_np = np.squeeze(mask_tensors[i].numpy())
                loss = float(calculate_dice_loss(pred_np, mask_np))
                scores.append({"image": image_name, "uncertainty_score": unc_score, "loss_score": loss})

            # --- Save progress after each batch ---
            ranked_scores = sorted(scores, key=lambda x: x['uncertainty_score'], reverse=True)
            with open(save_path, 'w') as f:
                json.dump({"ranked_images": ranked_scores}, f, indent=4)

        except Exception as e:
            print(f"\n  - Warning: AL batch starting with '{os.path.basename(image_paths[0])}' failed: {e}", flush=True)
            traceback.print_exc()
            continue


# --- Main Pre-computation Logic ---
if __name__ == "__main__":
    print("--- Starting Pre-computation (Fully Optimized) ---", flush=True)
    print(f"Using device: {DEVICE}, Batch Size: {BATCH_SIZE}, Workers: {NUM_WORKERS}", flush=True)
    print("Resumable. Skips existing results.", flush=True)

    models = glob.glob(os.path.join(MODEL_DIR, "*.pth"))
    if not models:
        print("Error: No models found in ./trained_models/", flush=True)
        exit(1)

    for mpath in models:
        mname = os.path.basename(mpath)
        print(f"\nProcessing model: {mname}", flush=True)
        model = load_model(mpath)

        for dname, paths in DATASET_PATHS.items():
            if dname.lower() not in mname.lower():
                print(f" - Skipping {dname} dataset for model {mname} (name mismatch).")
                continue

            # --- XAI Analysis (Optimized with DataLoader) ---
            images_dir = paths.get('images', '')
            if not (images_dir and os.path.isdir(images_dir)):
                print(f" - Skipping XAI for {dname}: Image directory not found or not specified.")
            else:
                out_dir = os.path.join(XAI_RESULTS_DIR, mname, dname)
                os.makedirs(out_dir, exist_ok=True)

                full_dataset = LungDataset(dataset_name=dname, transform=ResizeAndToTensor())

                # Find indices of images that still need to be processed
                processed_files = {os.path.basename(p).replace('.npz', '') for p in
                                   glob.glob(os.path.join(out_dir, "*.npz"))}
                indices_to_process = [i for i, path in enumerate(full_dataset.image_files) if
                                      os.path.basename(path) not in processed_files]

                if not indices_to_process:
                    print(f" - All {len(full_dataset)} XAI images for {dname} already processed.")
                else:
                    print(
                        f" - Resuming XAI for {dname}: {len(processed_files)}/{len(full_dataset)} done, {len(indices_to_process)} remaining.",
                        flush=True)

                    subset_to_process = Subset(full_dataset, indices_to_process)
                    dataloader = DataLoader(subset_to_process, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                            shuffle=False)

                    # --- FIX: Manually track paths to avoid KeyError if 'image_path' is not in batch ---
                    paths_to_process = [full_dataset.image_files[i] for i in indices_to_process]
                    path_idx = 0

                    for batch in tqdm(dataloader, desc=f"XAI for {dname}"):
                        image_tensors = batch['image'].to(DEVICE)

                        # Get the paths for the current batch manually
                        current_batch_size = image_tensors.size(0)
                        image_paths = paths_to_process[path_idx: path_idx + current_batch_size]
                        path_idx += current_batch_size

                        try:
                            # Perform XAI analysis on the batch
                            # --- PERFORMANCE FIX ---
                            # Increased grid_step from 8 to 16. This creates a lower-resolution
                            # heatmap but is 4x faster (16x16=256 steps vs 32x32=1024 steps).
                            # This is the primary bottleneck.
                            spatial_maps = generate_spatial_heatmap(model, image_tensors, DEVICE, grid_step=16)
                            mc_output = run_mc_dropout_inference(model, image_tensors, DEVICE, n_samples=25)

                            if not (isinstance(mc_output, (list, tuple)) and len(mc_output) >= 2):
                                print(
                                    f"\n  - [DEBUG] `run_mc_dropout_inference` returned unexpected value for batch starting with {os.path.basename(image_paths[0])}.")
                                continue

                            mean_preds = mc_output[0]
                            uncertainty_maps = mc_output[1]

                            # Process and save results for each item in the batch
                            for i in range(len(image_paths)):
                                original_np = np.array(Image.open(image_paths[i]).convert("L").resize(IMG_SIZE))
                                s_map_flat = spatial_maps[i].flatten()
                                u_map_flat = uncertainty_maps[i].flatten()

                                correlation = float('nan')
                                if s_map_flat.size > 1 and u_map_flat.size > 1:
                                    try:
                                        correlation = np.corrcoef(s_map_flat, u_map_flat)[0, 1]
                                    except Exception as e:
                                        print(
                                            f"\nCould not compute correlation for image {os.path.basename(image_paths[i])}: {e}")

                                result_data = {
                                    "original_np": original_np,
                                    "mean_pred": mean_preds[i],
                                    "spatial_map": spatial_maps[i],
                                    "uncertainty_map": uncertainty_maps[i],
                                    "correlation": correlation
                                }

                                save_name = os.path.basename(image_paths[i])
                                save_path = os.path.join(out_dir, f"{save_name}.npz")
                                np.savez_compressed(save_path, **result_data)

                        except Exception as e:
                            print(
                                f"\n  - Warning: XAI batch starting with '{os.path.basename(image_paths[0])}' failed: {e}",
                                flush=True)
                            traceback.print_exc()
                            continue

            # --- Active Learning (Optimized with Batching) ---
            savep = os.path.join(AL_RESULTS_DIR, f"{mname}_{dname}.json")
            print(f" - Active Learning for {dname}", flush=True)
            perform_active_learning_incrementally(model, dname, savep)

    print("\n--- Pre-computation Complete ---", flush=True)
