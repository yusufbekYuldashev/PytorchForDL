
import requests
import zipfile
from pathlib import Path

data_path = Path('data/')
image_path = data_path/'pizza_steak_sushi'

if image_path.is_dir():
    print('Skipping creating')
else:
    print('Creating one')
    image_path.mkdir(parents=True, exist_ok=True)
    
with open(data_path/'pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print('Downloading')
    f.write(request.content)
    
with zipfile.ZipFile(data_path/'pizza_steak_sushi.zip', 'r') as zip_ref:
    print('Unzipping')
    zip_ref.extractall(image_path)
