import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import sys

# Import model loader from training script
from sevenseg_train import load_model


def load_and_preprocess(path, crop=None, invert=False):
    """
    crop = (left, top, right, bottom) in pixels, optional.
    Returns tensor shape (1,1,24,16) float in [0,1]
    """
    img = Image.open(path)

    # Handle transparency - composite onto white background
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
        img = background
    else:
        img = img.convert("RGB")

    if crop is not None:
        img = img.crop(crop)

    # Convert to "segment detection" BEFORE resizing (at full resolution)
    # This handles red, blue, green, or any colored segments on white background
    arr = np.array(img, dtype=np.float32)  # (H, W, 3)

    # Distance from white (255,255,255) - higher = more "segment-like"
    white = np.array([255.0, 255.0, 255.0])
    dist_from_white = np.linalg.norm(arr - white, axis=2) / np.sqrt(3 * 255**2)

    # Threshold to binary - this prevents blur from turning off-segments on
    binary = (dist_from_white > 0.3).astype(np.float32)

    # Now resize the binary mask using NEAREST to avoid bleeding
    binary_img = Image.fromarray((binary * 255).astype(np.uint8), mode='L')
    binary_img = binary_img.resize((16, 24), Image.NEAREST)

    gray = np.array(binary_img, dtype=np.float32) / 255.0

    if invert:
        gray = 1.0 - gray

    x = torch.tensor(gray).unsqueeze(0).unsqueeze(0)  # (1,1,24,16)
    return x


def debug_show_preprocessed(img_path, crop=None, invert=False):
    """Show what the model actually sees after preprocessing."""
    x = load_and_preprocess(img_path, crop=crop, invert=invert)
    arr = x.squeeze().numpy()  # (24, 16)

    print(f"Preprocessed image ({arr.shape[0]}x{arr.shape[1]}):")
    print(f"Min: {arr.min():.2f}, Max: {arr.max():.2f}")

    # ASCII visualization
    for row in arr:
        line = ""
        for val in row:
            if val > 0.7:
                line += "██"
            elif val > 0.3:
                line += "▒▒"
            else:
                line += "  "
        print(line)


def main():
    if len(sys.argv) < 2:
        print("Usage: python sevenseg_predict.py <image_path> [left top right bottom] [invert0or1] [debug]")
        print("Example: python sevenseg_predict.py sevenseg.png 100 200 260 420 1")
        print("Add 'debug' as last arg to see preprocessed image")
        sys.exit(1)

    img_path = sys.argv[1]

    crop = None
    invert = False

    # Optional crop box
    if len(sys.argv) >= 6:
        l = int(sys.argv[2]); t = int(sys.argv[3]); r = int(sys.argv[4]); b = int(sys.argv[5])
        crop = (l, t, r, b)

    # Optional invert flag
    if len(sys.argv) >= 7:
        invert = (int(sys.argv[6]) != 0)

    # Debug mode - show what the model sees
    if 'debug' in sys.argv:
        debug_show_preprocessed(img_path, crop=crop, invert=invert)
        print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model with character mappings
    model, idx_to_char = load_model("sevenseg_cnn.pth", device=device)

    x = load_and_preprocess(img_path, crop=crop, invert=invert).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        pred_char = idx_to_char[pred_idx]

    topk = torch.topk(probs, k=5)
    print(f"Prediction: '{pred_char}' (confidence: {probs[pred_idx]:.2%})")
    print("Top 5:")
    for val, idx in zip(topk.values, topk.indices):
        char = idx_to_char[int(idx)]
        print(f"  '{char}': {float(val):.2%}")

if __name__ == "__main__":
    main()
