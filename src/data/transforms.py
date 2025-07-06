import yaml
from torchvision import transforms

def load_config(config_path="src/configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
    
config = load_config()
img_size = config['data']['img_size']


transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])