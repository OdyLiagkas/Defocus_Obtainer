import yaml
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataloaders import get_dataloader
from classifier import Classifier
import wandb

import torchvision.utils as vutils  # Import make_grid
import numpy as np
import matplotlib.pyplot as plt

import torch.nn.init as init

# Function to load config from YAML file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Main function
def main(config):

    wandb.init(
        project=config['wandb']['project'],
        config=config,
        name=config['wandb']['run_id'],  
        entity="cryo_team_di"
    )    
    
    # Load device from config or default to CPU
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    # Load training parameters from config
    batch_size = config['batch_size']

    lr = config['lr']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    betas = (beta_1, beta_2)

    epochs = config['epochs']
    classes_num = config['classes_num']


    # Load data paths from config and select the first one (if needed, you can modify this to select multiple)
    data_paths = config['data_paths']  # List of data paths from config

    train_loader, test_loader = get_dataloader(paths_to_data=data_paths, batch_size=batch_size, standarization=config['standarization'])

    def initialize_weights(model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    classifier = Classifier(num_classes=classes_num)
    initialize_weights(classifier)



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):  

        print("Epoch:",epoch+1)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            particle, defocus = data
            particle=particle.float()
            defocus = defocus.long()
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = classifier(particle)
            loss = criterion(outputs, defocus)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:    # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50    :.3f}')
                wandb.log({"loss": running_loss/50})
                running_loss = 0.0

    print('Finished Training')

    dataiter = iter(test_loader)
    particle, defocus = next(dataiter)
    particle=particle.float()
    print('GroundTruth: ', ' '.join(f'{defocus[j]:5d}' for j in range(32)))

    outputs = classifier(particle)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.long()
    print('Predicted: ', ' '.join(f'{predicted[j]:5d}' for j in range(32)))

    PATH = './classifier_CNN.pth'
    torch.save(classifier.state_dict(), PATH)

if __name__ == "__main__":
    # Argument parser to pass the path of the YAML config file
    parser = argparse.ArgumentParser(description="Train Classifier with config")
    parser.add_argument('-a', '--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Run the main function with the loaded config
    main(config)
