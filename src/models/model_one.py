import yaml
import torch.nn as nn

def load_config(config_path="src/configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Config settings from the config file  
config = load_config()
img_size = config['data']['img_size']

# How the model would be
class Cnn_model_one(nn.Module):
    def __init__(self, input_size=img_size*img_size, hidden_size=128, output_size=38):
        super().__init__()
        # Convolutional Layers
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3) # output -> (5, img_size-kernel_size+1, img_size-kernel_size+1)
        self.activation1 = nn.ReLU()
        self.conv_layer_2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3) # output -> (10, img_size-kernel_size+1, img_size-kernel_size+1)
        self.activation2 = nn.ReLU()

        self.max_pool_layer = nn.MaxPool2d(kernel_size=2, stride=2) # output -> (10, (img_size-kernel_size+1)/2, (img_size-kernel_size+1)/2)

        print("Debugging Shapes")
        conv_1_shape = (img_size - 3 + 1)
        print(f"conv_layer_1 output shape: 5 x {conv_1_shape} x {conv_1_shape}")
        conv_2_shape = (conv_1_shape - 3 + 1)
        print(f"conv_layer_2 output shape: 10 x {conv_2_shape} x {conv_2_shape}")
        pool_shape = int((conv_1_shape - 3 + 1)/2)
        print(f"maxpooling_layer output shape: 10 x {pool_shape} x {pool_shape}")

        input_size = int(10 * (conv_1_shape - 3 + 1)/2 * (conv_1_shape - 3 + 1)/2)
        print(f"Required size into FC: {input_size}")

        # Fully Connected Layers
        self.fc_layer_1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.activation3 = nn.ReLU()
        self.fc_layer_2 = nn.Linear(in_features=hidden_size, out_features=output_size)


    def forward(self, x):
        # Conv Layers
        x = self.activation1(self.conv_layer_1(x))
        x = self.activation2(self.conv_layer_2(x))
        # Max pooling layer
        x = self.max_pool_layer(x)
        # Flatten features
        x = x.view(x.size(0), -1)
        # Fully connected layers
        x = self.activation3(self.fc_layer_1(x))
        x = self.fc_layer_2(x)
        return x



