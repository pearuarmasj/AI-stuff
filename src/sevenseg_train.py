import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# segments: a b c d e f g
#
#    aaaa
#   f    b
#   f    b
#    gggg
#   e    c
#   e    c
#    dddd
#
# All displayable 7-segment characters (digits + letters)
CHAR_TO_SEG = {
    # Digits
    '0': [1,1,1,1,1,1,0],
    '1': [0,1,1,0,0,0,0],
    '2': [1,1,0,1,1,0,1],
    '3': [1,1,1,1,0,0,1],
    '4': [0,1,1,0,0,1,1],
    '5': [1,0,1,1,0,1,1],
    '6': [1,0,1,1,1,1,1],
    '7': [1,1,1,0,0,0,0],
    '8': [1,1,1,1,1,1,1],
    '9': [1,1,1,1,0,1,1],
    # Letters (uppercase-ish where possible, lowercase where clearer)
    'A': [1,1,1,0,1,1,1],
    'b': [0,0,1,1,1,1,1],
    'C': [1,0,0,1,1,1,0],
    'c': [0,0,0,1,1,0,1],
    'd': [0,1,1,1,1,0,1],
    'E': [1,0,0,1,1,1,1],
    'F': [1,0,0,0,1,1,1],
    'G': [1,0,1,1,1,1,0],
    'H': [0,1,1,0,1,1,1],
    'h': [0,0,1,0,1,1,1],
    # 'I': [0,1,1,0,0,0,0],  # same as 1 - REMOVED (use '1' instead)
    'J': [0,1,1,1,0,0,0],
    'L': [0,0,0,1,1,1,0],
    'n': [0,0,1,0,1,0,1],
    # 'O': [1,1,1,1,1,1,0],  # same as 0 - REMOVED (use '0' instead)
    'o': [0,0,1,1,1,0,1],
    'P': [1,1,0,0,1,1,1],
    'q': [1,1,1,0,0,1,1],
    'r': [0,0,0,0,1,0,1],
    # 'S': [1,0,1,1,0,1,1],  # same as 5 - REMOVED (use '5' instead)
    't': [0,0,0,1,1,1,1],
    'U': [0,1,1,1,1,1,0],
    'u': [0,0,1,1,1,0,0],
    'Y': [0,1,1,1,0,1,1],
    '-': [0,0,0,0,0,0,1],
    '_': [0,0,0,1,0,0,0],
    ' ': [0,0,0,0,0,0,0],
}

# Build lookup tables for training
CHARS = list(CHAR_TO_SEG.keys())
NUM_CLASSES = len(CHARS)
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}

def draw_seven_seg(segments, H=24, W=16, thickness=2, brightness=1.0, shift=(0,0), noise=0.03):
    img = torch.zeros((H, W), dtype=torch.float32)
    dy, dx = shift

    def rect(y0, y1, x0, x1, val):
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        y0 = max(0, y0); y1 = min(H, y1)
        x0 = max(0, x0); x1 = min(W, x1)
        if y1 > y0 and x1 > x0:
            img[y0:y1, x0:x1] = torch.maximum(img[y0:y1, x0:x1], torch.tensor(val))

    t = thickness
    pad = 2

    a = (pad+dy, pad+dy+t, pad+dx+t, W-pad+dx-t)
    d = (H-pad+dy-t, H-pad+dy, pad+dx+t, W-pad+dx-t)
    g = (H//2+dy - t//2, H//2+dy + math.ceil(t/2), pad+dx+t, W-pad+dx-t)

    f = (pad+dy+t, H//2+dy - t//2, pad+dx, pad+dx+t)
    b = (pad+dy+t, H//2+dy - t//2, W-pad+dx-t, W-pad+dx)
    e = (H//2+dy + t//2, H-pad+dy-t, pad+dx, pad+dx+t)
    c = (H//2+dy + t//2, H-pad+dy-t, W-pad+dx-t, W-pad+dx)

    boxes = [a,b,c,d,e,f,g]

    for on, (y0,y1,x0,x1) in zip(segments, boxes):
        if on:
            rect(y0, y1, x0, x1, brightness)

    if noise > 0:
        img = torch.clamp(img + noise * torch.randn_like(img), 0.0, 1.0)

    return img.unsqueeze(0)  # (1,H,W)

class SevenSegDataset(Dataset):
    def __init__(self, n=50000, H=24, W=16):
        self.n = n
        self.H, self.W = H, W

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        char_idx = random.randint(0, NUM_CLASSES - 1)
        char = CHARS[char_idx]
        seg = CHAR_TO_SEG[char].copy()

        # tiny corruption sometimes
        if random.random() < 0.15:
            j = random.randint(0, 6)
            seg[j] = 1 - seg[j]

        thickness = random.choice([1,2,2,3])
        brightness = random.uniform(0.5, 1.0)
        shift = (random.randint(-2,2), random.randint(-2,2))
        noise = random.uniform(0.00, 0.06)

        img = draw_seven_seg(seg, H=self.H, W=self.W,
                             thickness=thickness, brightness=brightness,
                             shift=shift, noise=noise)
        y = torch.tensor(char_idx, dtype=torch.long)
        return img, y

class TinyCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 24x16 -> 12x8
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 12x8 -> 6x4
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def load_model(path="sevenseg_cnn.pth", device="cpu"):
    """Load trained model and return (model, idx_to_char) for prediction."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle old format (just state_dict) vs new format (full checkpoint)
    if 'model_state_dict' in checkpoint:
        # New format
        num_classes = checkpoint['num_classes']
        state_dict = checkpoint['model_state_dict']
        idx_to_char = checkpoint['idx_to_char']
    else:
        # Old format - assume 10 digit classes (0-9)
        num_classes = 10
        state_dict = checkpoint
        idx_to_char = {i: str(i) for i in range(10)}

    model = TinyCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, idx_to_char


def predict(model, img_tensor, idx_to_char):
    """
    Predict character from image tensor.
    img_tensor: (1, H, W) or (B, 1, H, W)
    Returns: list of (char, confidence) tuples
    """
    with torch.no_grad():
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        confs, idxs = probs.max(dim=1)
        results = [(idx_to_char[i.item()], c.item()) for i, c in zip(idxs, confs)]
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print(f"Training on {NUM_CLASSES} classes: {CHARS}")

    train_ds = SevenSegDataset(n=60000)  # more samples for more classes
    test_ds  = SevenSegDataset(n=8000)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = TinyCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 6):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                n += y.numel()

        print(f"epoch {epoch} | loss {total_loss/len(train_ds):.4f} | acc {correct/n:.4f}")

    # Save model and class mappings together
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'chars': CHARS,
        'num_classes': NUM_CLASSES,
        'char_to_idx': CHAR_TO_IDX,
        'idx_to_char': IDX_TO_CHAR,
    }
    torch.save(checkpoint, "sevenseg_cnn.pth")
    print(f"Saved: sevenseg_cnn.pth ({NUM_CLASSES} classes)")

if __name__ == "__main__":
    main()
