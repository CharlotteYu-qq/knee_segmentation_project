import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataset import read_xray, read_mask
from model import UNetLext
from utils import dice_score_from_logits

def evaluate(model_path, csv_path, device="cpu"):

    # Load test CSV
    df = pd.read_csv(csv_path)

    # Load model
    model = UNetLext(input_channels=1, output_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    dice_scores = []

    for idx, row in df.iterrows():
        img_path = row["xrays"]
        mask_path = row["masks"]

        # Load XRAY
        image = read_xray(img_path)
        image_tensor = torch.tensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)

        pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0,0]


        # If mask exists (not null), compute Dice
        if pd.notna(mask_path):
            mask = read_mask(mask_path)[0]  # shape (H, W)
            mask_tensor = torch.tensor(mask).unsqueeze(0).to(device)  # shape (1, H, W)
            dice = dice_score_from_logits(output, mask_tensor)
            dice_scores.append(dice)

        # Show visualization
        if idx < 3:  # show first 3 images
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.title("Original X-ray")
            plt.imshow(image[0], cmap="gray")

            plt.subplot(1,2,2)
            plt.title("Predicted Mask")
            plt.imshow(pred_mask, cmap="gray")
            plt.show()

    if len(dice_scores) > 0:
        print("Average Dice:", sum(dice_scores)/len(dice_scores))
    else:
        print("Dice cannot be computed (test set masks unavailable)")


if __name__ == "__main__":
    evaluate(
        model_path="session/best_model.pth",
        csv_path="data/CSVs/test.csv",
        device="cpu"
    )