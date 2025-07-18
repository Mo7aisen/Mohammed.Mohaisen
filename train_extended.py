import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import time
import yaml
import shutil
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor

# Temporary fix - define the function here if import fails
try:
    from utils import plot_and_save_loss_curve
except ImportError:
    def plot_and_save_loss_curve(log_file_path, output_path):
        """Plot training and validation loss curves from a CSV log file"""
        try:
            log_df = pd.read_csv(log_file_path)
        except FileNotFoundError:
            return

        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid")
        plt.plot(log_df['epoch'], log_df['train_loss'], 'b-o', label='Training Loss', markersize=4)
        plt.plot(log_df['epoch'], log_df['val_loss'], 'r-o', label='Validation Loss', markersize=4)
        plt.title(f'Training & Validation Loss', fontsize=16)
        plt.xlabel('Epoch');
        plt.ylabel('Loss')

        if not log_df['val_loss'].empty and log_df['val_loss'].notna().any():
            best_epoch = log_df.loc[log_df['val_loss'].idxmin()]
            plt.axvline(x=best_epoch['epoch'], color='g', linestyle='--',
                        label=f"Best Model (Epoch {int(best_epoch['epoch'])})")

        plt.legend();
        plt.tight_layout();
        plt.savefig(output_path, dpi=300);
        plt.close()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def continue_training(run_name, target_epochs, config):
    """
    Continue training from existing checkpoint to reach target epochs
    """
    # Extract base information from run_name
    parts = run_name.split('_')
    dataset_name = parts[1]  # montgomery or jsrt
    data_fraction = 1.0 if parts[2] == 'full' else 0.5
    current_epochs = int(parts[3])

    # Setup paths
    output_base_dir = config['output_base_dir']

    # Source directory (where current model is)
    source_run_name = run_name.replace(f"_{target_epochs}", f"_{current_epochs}")
    source_dir = os.path.join(output_base_dir, source_run_name)

    # Target directory (where extended model will be saved)
    target_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(target_dir, exist_ok=True)

    # Copy existing files
    print(f"Copying existing results from {source_dir} to {target_dir}")
    for item in ['training_log.csv', 'loss_curve.png', 'snapshots']:
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        if os.path.exists(source_path):
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_path)

    # Load existing training log
    log_file = os.path.join(target_dir, "training_log.csv")
    if os.path.exists(log_file):
        training_log = pd.read_csv(log_file).to_dict('records')
        start_epoch = len(training_log)
    else:
        training_log = []
        start_epoch = 0

    # Load model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    checkpoint_path = os.path.join(source_dir, "final_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("Warning: No checkpoint found, starting from scratch")

    # Create dataset and dataloaders
    dataset = LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())

    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    if data_fraction < 1.0:
        train_set_size = int(len(train_set) * data_fraction)
        train_set = Subset(train_set, range(train_set_size))
        n_train = len(train_set)

    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                            batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Setup training
    learning_rate = config['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Extended save frequency
    save_checkpoint_freq = 10  # Save every 10 epochs for extended training

    # Training loop
    print(f"Continuing training from epoch {start_epoch + 1} to {target_epochs}")
    snapshots_dir = os.path.join(target_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    for epoch in range(start_epoch, target_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{target_epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar.update(images.size(0))

        avg_train_loss = train_loss / n_train if n_train > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)
                masks_pred = model(images)
                val_loss += criterion(masks_pred, true_masks).item() * images.size(0)

        avg_val_loss = val_loss / n_val if n_val > 0 else 0

        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Update training log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        # Save checkpoint
        if (epoch + 1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(snapshots_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: epoch_{epoch + 1}.pth")

    # Save final model and update logs
    final_model_path = os.path.join(target_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Save updated training log
    pd.DataFrame(training_log).to_csv(log_file, index=False)

    # Update loss curve plot
    plot_file = os.path.join(target_dir, "loss_curve.png")
    plot_and_save_loss_curve(log_file, plot_file)

    print(f"Extended training completed! Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Continue training to demonstrate overfitting")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['montgomery_full', 'montgomery_half', 'jsrt_half'])
    parser.add_argument('--target_epochs', type=int, required=True)
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Map dataset to run name
    dataset_mapping = {
        'montgomery_full': f'unet_montgomery_full_{args.target_epochs}',
        'montgomery_half': f'unet_montgomery_half_{args.target_epochs}',
        'jsrt_half': f'unet_jsrt_half_{args.target_epochs}'
    }

    run_name = dataset_mapping[args.dataset]
    continue_training(run_name, args.target_epochs, config)


if __name__ == "__main__":
    main()
import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import time
import yaml
import shutil

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor
from utils import plot_and_save_loss_curve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def continue_training(run_name, target_epochs, config):
    """
    Continue training from existing checkpoint to reach target epochs
    """
    # Extract base information from run_name
    parts = run_name.split('_')
    dataset_name = parts[1]  # montgomery or jsrt
    data_fraction = 1.0 if parts[2] == 'full' else 0.5
    current_epochs = int(parts[3])

    # Setup paths
    output_base_dir = config['output_base_dir']

    # Source directory (where current model is)
    source_run_name = run_name.replace(f"_{target_epochs}", f"_{current_epochs}")
    source_dir = os.path.join(output_base_dir, source_run_name)

    # Target directory (where extended model will be saved)
    target_dir = os.path.join(output_base_dir, run_name)
    os.makedirs(target_dir, exist_ok=True)

    # Copy existing files
    print(f"Copying existing results from {source_dir} to {target_dir}")
    for item in ['training_log.csv', 'loss_curve.png', 'snapshots']:
        source_path = os.path.join(source_dir, item)
        target_path = os.path.join(target_dir, item)

        if os.path.exists(source_path):
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, target_path)

    # Load existing training log
    log_file = os.path.join(target_dir, "training_log.csv")
    if os.path.exists(log_file):
        training_log = pd.read_csv(log_file).to_dict('records')
        start_epoch = len(training_log)
    else:
        training_log = []
        start_epoch = 0

    # Load model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    checkpoint_path = os.path.join(source_dir, "final_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    else:
        print("Warning: No checkpoint found, starting from scratch")

    # Create dataset and dataloaders
    dataset = LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())

    n_val = int(len(dataset) * 0.15)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    if data_fraction < 1.0:
        train_set_size = int(len(train_set) * data_fraction)
        train_set = Subset(train_set, range(train_set_size))
        n_train = len(train_set)

    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False,
                            batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    # Setup training
    learning_rate = config['learning_rate']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Extended save frequency
    save_checkpoint_freq = 10  # Save every 10 epochs for extended training

    # Training loop
    print(f"Continuing training from epoch {start_epoch + 1} to {target_epochs}")
    snapshots_dir = os.path.join(target_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    for epoch in range(start_epoch, target_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{target_epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                pbar.update(images.size(0))

        avg_train_loss = train_loss / n_train if n_train > 0 else 0

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch['image'].to(DEVICE), batch['mask'].to(DEVICE)
                masks_pred = model(images)
                val_loss += criterion(masks_pred, true_masks).item() * images.size(0)

        avg_val_loss = val_loss / n_val if n_val > 0 else 0

        print(f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

        # Update training log
        training_log.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        # Save checkpoint
        if (epoch + 1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(snapshots_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: epoch_{epoch + 1}.pth")

    # Save final model and update logs
    final_model_path = os.path.join(target_dir, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)

    # Save updated training log
    pd.DataFrame(training_log).to_csv(log_file, index=False)

    # Update loss curve plot
    plot_file = os.path.join(target_dir, "loss_curve.png")
    plot_and_save_loss_curve(log_file, plot_file)

    print(f"Extended training completed! Final model saved to {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Continue training to demonstrate overfitting")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['montgomery_full', 'montgomery_half', 'jsrt_half'])
    parser.add_argument('--target_epochs', type=int, required=True)
    args = parser.parse_args()

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Map dataset to run name
    dataset_mapping = {
        'montgomery_full': f'unet_montgomery_full_{args.target_epochs}',
        'montgomery_half': f'unet_montgomery_half_{args.target_epochs}',
        'jsrt_half': f'unet_jsrt_half_{args.target_epochs}'
    }

    run_name = dataset_mapping[args.dataset]
    continue_training(run_name, args.target_epochs, config)


if __name__ == "__main__":
    main()