"""
Contains utility functions
"""
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

def save_model(model, target_dir, model_name):
    MODEL_PATH = Path(target_dir)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODEL_PATH/model_name
    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def pred_and_plot_image(model, img_path, class_names, img_size=(224, 224), transform=None, device='cpu'):
    image = Image.open(img_path)
    if transform:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    model.to(device)
    model.eval()
    with torch.inference_mode():
        img = image_transform(image)
        img = img.unsqueeze(dim=0)
        pred = model(img.to(device))
    probs = torch.softmax(pred, dim=1)
    label = torch.argmax(probs, dim=1)
    plt.figure()
    plt.imshow(image)
    plt.title(f"Pred: {class_names[label]} Prob: {probs.max():.3f}")
    plt.axis(False)