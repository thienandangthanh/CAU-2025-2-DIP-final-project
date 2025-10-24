import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# --- Import local modules ---
from model import DCENet, enhance
from loss import TotalLoss
from dataset import LowLightDataset

def train(args):
    # --- Create directories if they don't exist ---
    if not os.path.exists(args.weights_dir):
        os.makedirs(args.weights_dir)

    # --- Setup device (CPU or GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Initialize model, loss, and optimizer ---
    model = DCENet().to(device)
    criterion = TotalLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # --- Prepare dataset and dataloader ---
    train_dataset = LowLightDataset(data_dir=args.data_dir, image_size=args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print("\n--- Starting Training ---")
    model.train()
    
    for epoch in range(args.num_epochs):
        # Use tqdm for a progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{args.num_epochs}]")
        total_loss_epoch = 0

        for low_light_images in loop:
            # Move data to the selected device
            low_light_images = low_light_images.to(device)

            # --- Forward pass ---
            # 1. Get the curve parameters from the network
            curve_params = model(low_light_images)
            # 2. Apply the enhancement
            enhanced_images = enhance(low_light_images, curve_params)
            
            # --- Calculate loss ---
            # The loss function requires the original image, the enhanced image, and the curve parameters
            loss, spa, exp, col, tvA = criterion(low_light_images, enhanced_images, curve_params)

            # --- Backward pass and optimization ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- Update progress bar ---
            total_loss_epoch += loss.item()
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                spa=f"{spa.item():.4f}",
                exp=f"{exp.item():.4f}",
                col=f"{col.item():.4f}",
                tvA=f"{tvA.item():.4f}"
            )
            
        # --- Log average loss for the epoch ---
        avg_loss = total_loss_epoch / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.num_epochs}] - Average Loss: {avg_loss:.4f}")

        # --- Save model weights ---
        if (epoch + 1) % args.save_interval == 0:
            save_path = os.path.join(args.weights_dir, f"dce_net_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    print("--- Training Finished ---")

if __name__ == '__main__':
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Train the Zero-DCE model.")
    
    parser.add_argument('--data_dir', type=str, default='data/train', help='Directory for training data.')
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save model weights.')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--image_size', type=int, default=256, help='Image size for training.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model weights every N epochs.')
    
    args = parser.parse_args()
    
    # --- Start training ---
    train(args)