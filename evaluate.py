import torch
import numpy as np
from captum.attr import IntegratedGradients
import scipy.ndimage
from tqdm import tqdm
import torch.nn.functional as F


def get_pixel_explainer(model, target_pixel_y, target_pixel_x):
    """
    Creates a wrapper around the model to explain the output of a single pixel.
    This is necessary for Captum's Integrated Gradients.
    """

    def pixel_forward_func(input_tensor):
        output_logits = model(input_tensor)
        return output_logits[:, 0, target_pixel_y, target_pixel_x]

    return IntegratedGradients(pixel_forward_func)


def calculate_spatial_metric(attributions, center_y, center_x):
    """
    Calculates the weighted median distance.
    """
    # --- FIX ---
    # Squeeze the attributions array to remove any extra single dimensions (like a batch or channel dimension).
    # This makes the function robust to inputs with shapes like (1, 256, 256).
    attributions = np.squeeze(attributions)

    # Take the absolute value of attributions, as their sign is not relevant for weighting.
    weights = np.abs(attributions)

    max_weight = np.max(weights)
    if max_weight == 0:
        return 0.0

    threshold = 0.1 * max_weight
    significant_mask = weights > threshold

    if not np.any(significant_mask):
        return 0.0

    # This line is now safe because `attributions` has been squeezed to 2D.
    h, w = attributions.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]

    distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

    flat_distances = distances[significant_mask]
    flat_weights = weights[significant_mask]

    sort_indices = np.argsort(flat_distances)
    sorted_distances = flat_distances[sort_indices]
    sorted_weights = flat_weights[sort_indices]

    cumulative_weights = np.cumsum(sorted_weights)
    total_weight = cumulative_weights[-1]

    if total_weight > 0:
        median_idx = np.searchsorted(cumulative_weights, total_weight / 2.0)
        median_idx = min(median_idx, len(sorted_distances) - 1)
        return sorted_distances[median_idx]
    else:
        return 0.0


def generate_spatial_heatmap(model, input_tensor_batch, device, grid_step=16):
    """
    Generates the spatial distribution heatmap for an entire BATCH of images efficiently.
    REVISION: This function now correctly handles a batch of images by iterating through them.
    """
    model.eval()
    input_tensor_batch = input_tensor_batch.to(device)
    input_tensor_batch.requires_grad_()

    batch_size, _, h, w = input_tensor_batch.shape
    all_heatmaps = []

    # --- FIX: Loop over each image in the batch ---
    for item_index in range(batch_size):
        # Process one image at a time. Add a dimension to maintain the 4D shape.
        input_tensor_single = input_tensor_batch[item_index].unsqueeze(0)

        h_sparse = (h + grid_step - 1) // grid_step
        w_sparse = (w + grid_step - 1) // grid_step
        sparse_heatmap = np.zeros((h_sparse, w_sparse), dtype=np.float32)
        baseline = torch.zeros_like(input_tensor_single)

        # The tqdm description now shows which image in the batch is being processed
        for r in tqdm(range(h_sparse), desc=f"  Heatmap (Image {item_index + 1}/{batch_size})", leave=False):
            for c in range(w_sparse):
                y = r * grid_step
                x = c * grid_step

                ig = get_pixel_explainer(model, y, x)
                attributions = ig.attribute(input_tensor_single, baselines=baseline, n_steps=25, internal_batch_size=1)

                # Squeeze is correct here because we are processing a single image (batch size 1)
                attr_np = attributions.squeeze().cpu().detach().numpy()

                sparse_heatmap[r, c] = calculate_spatial_metric(attr_np, y, x)

        zoom_factor_h = h / h_sparse
        zoom_factor_w = w / w_sparse
        full_heatmap = scipy.ndimage.zoom(sparse_heatmap, (zoom_factor_h, zoom_factor_w), order=1)
        all_heatmaps.append(full_heatmap)

    # Stack the list of individual heatmaps into a single numpy array for the batch.
    return np.stack(all_heatmaps, axis=0)


def run_mc_dropout_inference(model, input_tensor, device, n_samples=25):
    """
    Performs inference using Monte Carlo Dropout to estimate model uncertainty.
    """
    # In PyTorch, calling model.train() enables dropout layers.
    # If you have a custom `enable_dropout` method, that is also fine.
    model.train()
    input_tensor = input_tensor.to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(input_tensor)
            pred_prob = torch.sigmoid(logits)
            predictions.append(pred_prob.cpu().numpy())

    # Shape: (n_samples, batch, channel, h, w)
    predictions = np.stack(predictions)

    # Shape: (batch, channel, h, w)
    mean_prediction = np.mean(predictions, axis=0)
    uncertainty_map = np.var(predictions, axis=0)

    # --- FIX ---
    # Safely squeeze only the channel dimension (axis=1), which is usually 1.
    # This avoids accidentally removing the batch dimension.
    if mean_prediction.shape[1] == 1:
        mean_prediction = mean_prediction.squeeze(axis=1)
    if uncertainty_map.shape[1] == 1:
        uncertainty_map = uncertainty_map.squeeze(axis=1)

    return mean_prediction, uncertainty_map


def calculate_dice_loss(pred_mask, true_mask):
    """Calculates the Dice Loss between a prediction and a ground truth mask."""
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()
    intersection = np.sum(pred_flat * true_flat)
    dice_coefficient = (2. * intersection) / (np.sum(pred_flat) + np.sum(true_flat) + 1e-8)
    return 1 - dice_coefficient
