import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
import csv
from captum.attr import IntegratedGradients
import yaml

# Import our custom modules
from model import UNet
from data_loader import LungDataset, ResizeAndToTensor
from utils import get_data_splits, dice_score, iou_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout): m.train()

def generate_ig_map(model, input_tensor, device):
    model.eval()
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()
    def model_forward_wrapper(inp): return model(inp).sum().unsqueeze(0)
    ig = IntegratedGradients(model_forward_wrapper)
    baseline = torch.zeros_like(input_tensor)
    attributions = ig.attribute(input_tensor, baselines=baseline, n_steps=25)
    return attributions.squeeze().cpu().detach().numpy()

def generate_uncertainty_map(model, input_tensor, device, n_samples=25):
    enable_dropout(model)
    input_tensor = input_tensor.to(device)
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            predictions.append(torch.sigmoid(model(input_tensor)).cpu())
    return torch.var(torch.stack(predictions), dim=0).squeeze().numpy()

def evaluate_model(run_name, state, split, config):
    print(f"--- Starting Evaluation on '{split}' split for state '{state}' ---")
    
    # 1. Get parameters from config
    params = config['experiments'][run_name]
    dataset_name = params['dataset']
    run_dir = os.path.join(config['output_base_dir'], run_name)
    
    # 2. Determine model path based on state
    states_map = {
        "underfitting": "snapshots/epoch_10.pth",
        "good_fitting": "best_model.pth",
        "overfitting": "final_model.pth"
    }
    model_file = states_map[state]
    model_path = os.path.join(run_dir, model_file)

    if not os.path.exists(model_path):
        print(f"Model for '{state}' state not found: {model_path}. SKIPPING."); return

    # 3. Load Model
    model = UNet(n_channels=1, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    # 4. Prepare Dataloader
    full_dataset = LungDataset(dataset_name=dataset_name, config=config, transform=ResizeAndToTensor())
    train_set, val_set, test_set = get_data_splits(full_dataset)
    
    split_map = {"training": train_set, "validation": val_set, "test": test_set}
    data_to_evaluate = split_map[split]
    data_loader = DataLoader(data_to_evaluate, batch_size=1, shuffle=False)
    
    # 5. Setup Output
    split_output_dir = os.path.join(run_dir, "evaluation", state, split)
    maps_dir = os.path.join(split_output_dir, "maps")
    os.makedirs(maps_dir, exist_ok=True)
    
    csv_file = open(os.path.join(split_output_dir, "_results_log.csv"), 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_name", "dice_score", "iou_score"])
    
    # 6. Evaluation Loop
    results = []
    for i, batch in enumerate(tqdm(data_loader, desc=f"Evaluating on {dataset_name} '{split}' set")):
        image_tensor, mask_tensor = batch['image'].to(DEVICE), batch['mask']
        image_name = os.path.basename(full_dataset.image_files[data_to_evaluate.indices[i]])
        
        with torch.no_grad(): pred_probs = torch.sigmoid(model(image_tensor))
        
        dice = dice_score((pred_probs > 0.5).cpu(), mask_tensor)
        iou = iou_score((pred_probs > 0.5).cpu(), mask_tensor)
        csv_writer.writerow([image_name, f"{dice:.6f}", f"{iou:.6f}"])
        
        ig_map = generate_ig_map(model, image_tensor, DEVICE)
        uncertainty_map = generate_uncertainty_map(model, image_tensor, DEVICE)
        
        npz_path = os.path.join(maps_dir, f"{image_name}.npz")
        np.savez_compressed(npz_path, ig_map=ig_map, uncertainty_map=uncertainty_map,
                            prediction=pred_probs.cpu().squeeze().numpy(),
                            ground_truth=mask_tensor.squeeze().numpy())
        
        results.append({"image_name": image_name, "dice_score": dice, "iou_score": iou, "xai_results_path": npz_path})
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    csv_file.close()
        
    # 7. Save Summary
    summary = {
        "model_path": model_path, "dataset_name": dataset_name, "split": split, "state": state,
        "num_samples": len(data_to_evaluate), "average_dice_score": np.mean([r['dice_score'] for r in results]),
        "average_iou_score": np.mean([r['iou_score'] for r in results]),
        "per_sample_results": results
    }
    with open(os.path.join(split_output_dir, "_evaluation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n--- Evaluation for '{state}' on '{split}' split Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, required=True)
    # --- FIXED: Added the missing --state argument ---
    parser.add_argument('--state', type=str, required=True, choices=['underfitting', 'good_fitting', 'overfitting'])
    parser.add_argument('--split', type=str, required=True, choices=['training', 'validation', 'test'])
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    evaluate_model(args.run_name, args.state, args.split, config)
