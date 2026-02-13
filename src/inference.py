import torch
from PIL import Image
from torchvision import transforms
import argparse
import utils
import model as m


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cats vs Dogs inference")
    # Arguments used for inference (To be written in terminal)
    p.add_argument("--config", type=str, default=None, required=True,
                   help="Path to YAML config")
    p.add_argument("--model", type=str, default=None, required=True,
                   help="Path to trained model weights")
    p.add_argument("--input", type=str, default=None, required=True,
                   help="Path to image file")
    return p.parse_args()


def main():
    args = parse_args()

    # Load config and device
    config = utils.load_config(args.config)
    device = utils.get_device(config)

    # Build model and load weights
    model = m.build_model()
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Preprocess image
    size = config["dataset"]["image_size"]

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    img = Image.open(args.input).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.inference_mode():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = probs.argmax(dim=1).item()

    label = "dog" if pred == 1 else "cat"

    print(f"Prediction: {label}")
    print(f"Probabilities: {probs.cpu().tolist()}")


if __name__ == "__main__":
    main()
