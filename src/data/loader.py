import yaml
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from data.transforms import transform
from pathlib import Path


def load_config(config_path="src/configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Config settings from the config file  
config = load_config()
train_path = Path(config['data']['train_dir'])
batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']


# Check the dataset folder and retrive the images from their respective folders ...
# and aasign to them corresponding labels
dataset = datasets.ImageFolder(root=train_path, transform=transform)

# A dataloader to select batches of images from the dataset, randomly
train_dataloader = DataLoader(
    dataset=dataset, 
    batch_size=batch_size, 
    num_workers=num_workers, 
    shuffle=True
)
