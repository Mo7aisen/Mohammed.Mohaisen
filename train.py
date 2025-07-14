import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.nn as nn

from model import UNet
from data_loader import LungDataset, ResizeAndToTensor

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_DIR = "./trained_models/"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


def train_model(
        dataset_name,
        epochs,
        batch_size,
        learning_rate,
        data_fraction,
        model_name,
        val_percent=0.1
):
    """
    Main training loop for the U-Net model.
    This script allows you to train models with different configurations,
    which is essential for creating the "less trained" models your supervisors requested.
    """
    print(f"--- Starting Training ---")
    print(f"Device: {DEVICE}")
    print(f"Dataset: {dataset_name}, Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}")
    print(f"Data Fraction: {data_fraction}, Validation Split: {val_percent}")
    print(f"Model will be saved as: {model_name}")

    # 1. Create Dataset and Dataloaders
    try:
        dataset = LungDataset(
            dataset_name=dataset_name,
            transform=ResizeAndToTensor(),
            data_fraction=data_fraction
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Please check your dataset paths in data_loader.py and ensure you have access.")
        return

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    # FIX: Set drop_last=False for validation loader to ensure it's never empty if val_set has samples.
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=4,
                            pin_memory=True)

    # 2. Initialize Model, Optimizer, and Loss Function
    model = UNet(n_channels=1, n_classes=1, dropout_rate=0.2).to(DEVICE)
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Training Loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device=DEVICE, dtype=torch.float32)
                true_masks = batch['mask'].to(device=DEVICE, dtype=torch.float32)

                masks_pred = model(images)
                loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # 4. Validation
        # FIX: Check if the validation loader has any batches before trying to calculate validation loss.
        if len(val_loader) > 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device=DEVICE, dtype=torch.float32)
                    true_masks = batch['mask'].to(device=DEVICE, dtype=torch.float32)
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            print(f'Epoch {epoch + 1} - Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(MODEL_SAVE_DIR, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path} (Val Loss: {best_val_loss:.4f})")
        else:
            # If there's no validation set, just print train loss and save the model at the end
            print(
                f'Epoch {epoch + 1} - Train Loss: {epoch_loss / len(train_loader):.4f}. No validation set to evaluate.')
            # Save the model from the last epoch if no validation is done
            if epoch == epochs - 1:
                model_path = os.path.join(MODEL_SAVE_DIR, model_name)
                torch.save(model.state_dict(), model_path)
                print(f"No validation set. Model from final epoch saved to {model_path}")

    print("--- Training Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net model for lung segmentation.")
    parser.add_argument('--dataset', type=str, required=True, choices=['jsrt', 'montgomery'],
                        help='Dataset to use for training.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_targument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--data_fraction', type=float, default=1.0, help='Fraction of data to use (0.0 to 1.0).')
    parser.add_argument('--model_name', type=str, default='unet_model.pth', help='Filename for the saved model.')
    args = parser.parse_args()

    train_model(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        data_fraction=args.data_fraction,
        model_name=args.model_name
    )
