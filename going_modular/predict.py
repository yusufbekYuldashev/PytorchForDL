
import torch
import torchvision
from torchvision import transforms
import model_builder
import argparse
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, help='Target image address')
parser.add_argument("--model_path",
                    default="models/05_going_modular_tinyvgg.pth",
                    type=str,
                    help="target model to use for prediction filepath")

args = parser.parse_args()

image_path = args.image
class_names = ['pizza', 'steak', 'sushi']

image = torchvision.io.read_image(str(image_path)).type(torch.float32)
image = image / 255.

transform = transforms.Compose([
    transforms.Resize((64, 64))
])
image = transform(image)

model = model_builder.TinyVGG(3, 10, 3)
model.load_state_dict(torch.load(f=args.model_path))
model.to(device)
model.eval()
with torch.inference_mode():
    image = image.unsqueeze(dim=0)
    logits = model(image)
    probs = torch.softmax(logits, dim=1)
    label = torch.argmax(probs, dim=1)
print(f"Pred: {class_names[label.cpu()]} Prob: {probs.max().cpu():.3f}")
