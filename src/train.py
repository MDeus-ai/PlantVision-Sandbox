import yaml
import tqdm
import torch
import torch.nn as nn
from models.model_one import Cnn_model_one
from  data.loader import train_dataloader


def load_config(config_path="src/configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    
# Config settings from the config file  
config = load_config()
num_epochs = config['train']['num_epochs']
learning_rate = config['train']['learning_rate']


if __name__ == "__main__":    

    # Model Instance
    model = Cnn_model_one()

    # Loss function and the optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") # For showing the progress bar
        for imgs, labels in progress_bar:
            outputs = model(imgs) # The forward pass, data ingestion
            loss = criterion(outputs, labels) # Calculate the loss

            optimizer.zero_grad() # Zero out accumulated gradients from previous epochs
            loss.backward() # perform backpropagation
            optimizer.step() # Update model weights

            progress_bar.set_postfix(loss=loss.item())

            running_loss += loss.item()
        epoch_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, - Avg Loss: {epoch_loss:.4f}") # Track loss for each epoch



