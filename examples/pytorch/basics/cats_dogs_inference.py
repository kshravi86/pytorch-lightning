# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference script for the cats and dogs classifier."""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from cats_dogs_classifier import LitCatsDogs


def predict(image_path: str, checkpoint_path: str = "cats_dogs_classifier.ckpt") -> str:
    model = LitCatsDogs.load_from_checkpoint(checkpoint_path)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return "dog" if pred == 1 else "cat"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="cats_dogs_classifier.ckpt")
    args = parser.parse_args()
    label = predict(args.image, args.checkpoint)
    print(label)


if __name__ == "__main__":
    main()
