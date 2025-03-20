import yaml
import argparse
import os
import torch
import torch.optim as optim
import torch.nn as nn
from dataloaders import get_dataloader
from regressor import Regressor
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

    train_loader, test_loader = get_dataloader(paths_to_data=data_paths, type_='sequential' ,batch_size=batch_size, standarization=config['standarization'])

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

    regressor = Regressor()
    initialize_weights(regressor)



    criterion = nn.MSELoss()
    optimizer = optim.Adam(regressor.parameters(), lr=lr, betas=betas)

    for epoch in range(epochs):  

        print("Epoch:",epoch+1)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            particle, defocus = data
            particle=particle.float()
            defocus = defocus.long()
            defocus = (defocus - 7599) / (45410 - 7599)  # NORMALIZATION FORMULA #min-max scaling to bring values to 0-1
            defocus = defocus.unsqueeze(1)  # Shape: [bs, 1]  FOR MSE LOSS!

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = regressor(particle)
            loss = criterion(output, defocus)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.3f}')

        #SAVE THE MODEL EACH EPOCH!
        PATH = f'../Defocus_Classifier_with_pth_NOT_ON_GIT/cryoSPIN_regressor_NO_DECIMALS_CNN_EPOCH_{epoch}.pth'
        torch.save(regressor.state_dict(), PATH)        
        
        # WILL ADD WANDB LOG OF DIVERGENCE FROM VALIDATION TEST BELOW:
        total1 = 0
        counter1 = 0
        with torch.no_grad():
            print('TRAIN SET ACCURACY:')
            for data in train_loader:
                particle, defocus = data
                defocus = (defocus - 7599) / (45410 - 7599)  # NORMALIZATION FORMULA
                # calculate outputs by running images through the network
                predicted = regressor(particle)
                #counter += torch.sum(torch.abs(defocus - predicted)).item()           
                total1 += defocus.size()[0]
                for i in range(defocus.size()[0]):
                    counter1 += (abs(defocus[i].item() - predicted[i].item()))
                    #print(defocus[i].item(), predicted[i].item())
                    #print('COUNTER:',counter1,'TOTAL:',total1)  
            print("Average divergence from actual value FOR THE TRAIN SET(!!!):", counter1/total1)
            
            total2=0
            counter2=0

            print('NOW FOR THE TEST SET:')
            for data in test_loader:
                particle, defocus = data
                defocus = (defocus - 7599) / (45410 - 7599)  # NORMALIZATION FORMULA
                # calculate outputs by running images through the network
                predicted = regressor(particle)
                #counter += torch.sum(torch.abs(defocus - predicted)).item()           
                total2 += defocus.size()[0]
                for i in range(defocus.size()[0]):
                    counter2 += (abs(defocus[i].item() - predicted[i].item()))
                    #print(defocus[i].item(), predicted[i].item())
                    #print('COUNTER:',counter2,'TOTAL:',total2)
            print("Average divergence from actual value for the test set:", counter2/total2)
            wandb.log({"Average divergence - train_set": counter1/total1, "Average divergence - test_set": counter2/total2, "loss": avg_loss})
            running_loss = 0.0
        
    
    print('Finished Training')
        
    '''
    dataiter = iter(test_loader)
    particle, defocus = next(dataiter)
    particle=particle.float()
    print('GroundTruth: ', ' '.join(f'{defocus[j]:5d}' for j in range(32)))

    outputs = regressor(particle)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.long()
    print('Predicted: ', ' '.join(f'{predicted[j]:5d}' for j in range(32)))
    '''
    


    
    total = 0
    counter = 0
    with torch.no_grad():
        print('TRAIN SET ACCURACY:')
        for data in train_loader:
            particle, defocus = data
            defocus = (defocus - 7599) / (45410 - 7599)  # NORMALIZATION FORMULA
            # calculate outputs by running images through the network
            predicted = regressor(particle)
            #counter += torch.sum(torch.abs(defocus - predicted)).item()           
            total += defocus.size()[0]
            for i in range(defocus.size()[0]):
                counter += (abs(defocus[i].item() - predicted[i].item()))
                #print(defocus[i].item(), predicted[i].item())
                #print('COUNTER:',counter,'TOTAL:',total)  
        print("Average divergence from actual value FOR THE TRAIN SET(!!!):", counter/total)
        total=0
        counter=0

        print('NOW FOR THE TEST SET:')
        for data in test_loader:
            particle, defocus = data
            defocus = (defocus - 7599) / (45410 - 7599)  # NORMALIZATION FORMULA
            # calculate outputs by running images through the network
            predicted = regressor(particle)
            #counter += torch.sum(torch.abs(defocus - predicted)).item()           
            total += defocus.size()[0]
            for i in range(defocus.size()[0]):
                counter += (abs(defocus[i].item() - predicted[i].item()))
                #print(defocus[i].item(), predicted[i].item())
                #print('COUNTER:',counter,'TOTAL:',total)
            

    print("Average divergence from actual value for the test set:", counter/total)

if __name__ == "__main__":
    # Argument parser to pass the path of the YAML config file
    parser = argparse.ArgumentParser(description="Train Regressor with config")
    parser.add_argument('-a', '--config', type=str, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(args.config)

    # Run the main function with the loaded config
    main(config)
