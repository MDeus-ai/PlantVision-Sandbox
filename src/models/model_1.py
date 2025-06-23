import yaml
import torch.nn as nn

def load_config(config_path="src/configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Config settings from the config file  
config = load_config()
img_size = config['data']['img_size']

# How the model would be
class MLP(nn.Module):
    def __init__(self, input_size=img_size*img_size, hidden_size=128, output_size=38):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.relu_1 = nn.ReLU()
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu_1(self.input_layer(x))
        x = self.output_layer(x)
        return x



