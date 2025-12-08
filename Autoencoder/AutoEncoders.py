import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, random_split
import os
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
load_dotenv()

# Configuration
DEVICE = None
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("CUDA device found.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("MPS device found.")
else:
    DEVICE = torch.device("cpu")
    print("MPS/CUDA device not found, falling back to CPU.")

x = torch.ones(1, device=DEVICE)
print(x)
print(f"The available device is {DEVICE}")


    
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
LOG_DIR = "./ae_checkpoints_fcn"
os.makedirs(LOG_DIR, exist_ok=True)
SAVE_RECON_EVERY = 1

# Safe Loader
def safe_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (OSError, UnidentifiedImageError):
        print(f"Warning: Skipping corrupt or missing file: {path}")
        return Image.new('RGB', (224, 224))

# Data Loading
print("Loading data...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_folder = os.getenv("IMAGE_FOLDER") # Path to your image folder
dataset = datasets.ImageFolder(image_folder, transform=transform, loader=safe_loader)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Data loaded. Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# Fully Convolutional Architecture
class Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        # Input: 3 x 224 x 224
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, stride=1, padding=1),  # -> 32 x 224 x 224
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),          # -> 64 x 224 x 224
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),         # -> 128 x 224 x 224
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),        # -> 256 x 224 x 224
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),        # -> 512 x 224 x 224
            nn.ReLU(True),
            # Bottleneck convolution to compress channels
            # Output will be 32 x 224 x 224 (No downsampling)
            nn.Conv2d(512, 32, 3, padding=1),                   
            nn.ReLU(True)
        )

    def forward(self, x):
        z = self.conv(x)
        return z

class Decoder(nn.Module):
    def __init__(self, out_channel=3):
        super().__init__()
        
        self.deconv = nn.Sequential(
            # Expand channels back from 32 to 512
            nn.ConvTranspose2d(32, 512, 3, padding=1),          # -> 512 x 224 x 224
            nn.ReLU(True),
            # Upsampling path
            nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1), # -> 256 x 224 x 224
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1), # -> 128 x 224 x 224
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # -> 64 x 224 x 224
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),   # -> 32 x 224 x 224
            nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channel, 3, stride=1, padding=1), # -> 3 x 224 x 224
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.deconv(z)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()
        self.encoder = Encoder(in_channel)
        self.decoder = Decoder(in_channel)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def save_image_grid(x_tensor, filename, nrow=8):
    grid = utils.make_grid(x_tensor.cpu(), nrow=nrow, padding=2)
    ndarr = grid.mul(255).byte().permute(1, 2, 0).numpy()
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.axis('off')
    plt.imshow(ndarr)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for xb, _ in dataloader:
        xb = xb.to(device)
        optimizer.zero_grad()
        out, z = model(xb)
        loss = criterion(out, xb)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * xb.size(0)
        total_samples += xb.size(0)

    return running_loss / total_samples

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for xb, _ in dataloader:
            xb = xb.to(device)
            recon, z = model(xb)
            loss = criterion(recon, xb)
            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)
    return total_loss / total_samples

# Training Loop
print(f"Initializing Fully Convolutional model on {DEVICE}...")
model = AutoEncoder(in_channel=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

best_val = float("inf")

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
    val_loss = evaluate(model, test_loader, criterion, DEVICE)
    
    print(f"[Epoch {epoch:03d}] train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

    # Save checkpoint
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(LOG_DIR, "ae_best_fcn.pt"))

    # Save reconstructions
    if epoch % SAVE_RECON_EVERY == 0:
        model.eval()
        xb, _ = next(iter(test_loader))
        xb = xb.to(DEVICE) # Ensure input is on the correct device
        with torch.no_grad():
            recon, _ = model(xb)
        
        # Concatenate original and recon
        both = torch.cat([xb.cpu(), recon.cpu()], dim=0)
        save_image_grid(both, os.path.join(LOG_DIR, f"recon_epoch{epoch:03d}.png"), nrow=8)
        print(f"Saved reconstruction to {LOG_DIR}/recon_epoch{epoch:03d}.png")
