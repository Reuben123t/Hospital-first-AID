from __future__ import annotations

import argparse
from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms
from train import BrainCNN


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', type=Path, required=True); p.add_argument('--image', type=Path, required=True); args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = BrainCNN(len(ckpt['classes'])).to(device); model.load_state_dict(ckpt['model']); model.eval()
    tfm = transforms.Compose([transforms.Grayscale(1), transforms.Resize((ckpt['image_size'], ckpt['image_size'])), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    with Image.open(args.image) as im, torch.inference_mode():
        probs = model(tfm(im).unsqueeze(0).to(device)).softmax(1)[0]; idx = int(probs.argmax())
    print({'label': ckpt['classes'][idx], 'confidence': round(float(probs[idx]), 4), 'probabilities': {c: round(float(x), 4) for c, x in zip(ckpt['classes'], probs)}})


if __name__ == '__main__': main()
