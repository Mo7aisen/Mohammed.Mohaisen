import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def find_best_epoch(log_file_path):
    """
    Parses a training log CSV to find the epoch with the lowest validation loss
    and the final epoch number.

    Args:
        log_file_path (str): The full path to the training_log.csv file.

    Returns:
        tuple: A tuple containing (best_epoch, final_epoch).
               Returns (0, 0) if the file cannot be found or parsed.
    """
    if not os.path.exists(log_file_path):
        return 0, 0
    try:
        # Load the training log
        df = pd.read_csv(log_file_path)

        # Check for the required columns to avoid errors
        if 'val_loss' not in df.columns or 'epoch' not in df.columns:
            return 0, df['epoch'].max() if 'epoch' in df.columns else 0

        # Find the index of the minimum validation loss
        best_epoch_idx = df['val_loss'].idxmin()
        # Get the epoch number at that index
        best_epoch = int(df.loc[best_epoch_idx, 'epoch'])
        # Get the last epoch number
        final_epoch = int(df['epoch'].max())

        return best_epoch, final_epoch
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"Could not process log file {log_file_path}: {e}")
        return 0, 0

def plot_loss_curve(df_log, best_epoch):
    """
    Generates a Matplotlib figure of training and validation loss curves.

    Args:
        df_log (pd.DataFrame): DataFrame with 'epoch', 'train_loss', 'val_loss'.
        best_epoch (int): The epoch number to highlight with a vertical line.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(x='epoch', y='train_loss', data=df_log, label='Training Loss', ax=ax, color='royalblue')
    sns.lineplot(x='epoch', y='val_loss', data=df_log, label='Validation Loss', ax=ax, color='darkorange')

    # Highlight the best epoch
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=2, label=f'Best Epoch: {best_epoch}')

    ax.set_title("Training & Validation Loss Curve", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(bottom=0) # Loss should not be negative
    plt.tight_layout()
    return fig


def calculate_cumulative_histogram(attribution_map, center_point):
    """
    Calculates the cumulative histogram of attribution scores based on distance
    from a specified center point and returns the data and a plot.

    Args:
        attribution_map (np.array): 2D numpy array of attribution scores (e.g., IG map).
        center_point (tuple): A tuple (y, x) for the center coordinate.

    Returns:
        tuple: A tuple containing:
               - pd.Series: The cumulative histogram data (index=distance, values=score).
               - float: The weighted median distance.
               - matplotlib.figure.Figure: The figure object for the plot.
    """
    # Create coordinate grids for distance calculation
    y_coords, x_coords = np.mgrid[0:attribution_map.shape[0], 0:attribution_map.shape[1]]

    # Calculate Euclidean distance from the center_point to all other points
    distances = np.sqrt((y_coords - center_point[0])**2 + (x_coords - center_point[1])**2)

    # Use absolute attribution values as weights
    weights = np.abs(attribution_map)

    # Create a DataFrame for easier manipulation
    df = pd.DataFrame({
        'distance': distances.flatten(),
        'weight': weights.flatten()
    })

    # Sort by distance to ensure correct cumulative sum
    df = df.sort_values(by='distance').reset_index(drop=True)

    # Calculate the cumulative sum of weights (attributions)
    df['cumulative_weight'] = df['weight'].cumsum()

    # Normalize the cumulative sum to a [0, 1] range
    total_weight = df['cumulative_weight'].max()
    if total_weight > 0:
        df['normalized_cumulative'] = df['cumulative_weight'] / total_weight
    else:
        df['normalized_cumulative'] = 0

    # Group by integer distance and take the max cumulative value for each distance bin
    # This creates a smooth curve for plotting
    hist_data = df.groupby(df['distance'].round().astype(int))['normalized_cumulative'].max()

    # Calculate the weighted median distance (where cumulative score is 0.5)
    try:
        # Find the first distance where the cumulative score is >= 0.5
        median_dist = df[df['normalized_cumulative'] >= 0.5]['distance'].iloc[0]
    except IndexError:
        # If 0.5 is never reached (e.g., all weights are zero), default to max distance
        median_dist = df['distance'].max()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=hist_data.index, y=hist_data.values, ax=ax, marker='.', markersize=5, linestyle='-')
    ax.axvline(x=median_dist, color='r', linestyle='--', label=f'Median Distance: {median_dist:.2f} px')
    ax.set_title(f"Cumulative Attribution from Point {center_point}")
    ax.set_xlabel("Distance from Center (pixels)")
    ax.set_ylabel("Normalized Cumulative Score")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, hist_data.index.max())
    plt.tight_layout()

    return hist_data, median_dist, fig
