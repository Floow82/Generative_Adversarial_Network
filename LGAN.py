
"""Generative Adversarial Network with Linear layers -- Models and Training"""

###Importations
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import FRFDataset

#--------------------------------------------------------MODELS----------------------------------------------------------

###GENERATEUR
class Generator (nn.Module):
  def __init__(self, frf_length, input_size):
    super(Generator, self).__init__()
    self.frf_length = frf_length
    self.input_size = input_size

    #Repetitive block of layer, normalization and activation
    def GBlock(self, inp, out):
      return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True))
    
    #Structure of the layers of the model
    self.linear = nn.Sequential(
        nn.Linear(self.input_size, 64), 
        GBlock(64, 128),
        GBlock(128, 512),
        GBlock(512, 1024),
        GBlock(64, 128),
        nn.Linear(1024, 3001),
        nn.Tanh())
    
    #Initialization of layer weights for faster, more stable convergence during the training
    self._initialize_weights()

  def forward (self, coordinates):
    x = self.linear(coordinates)
    return x

  def _initialize_weights(self):
    for m in self.modules():                                             #Iterations through the layers
      if isinstance(m,nn.ConvTranspose2d) or isinstance(m, nn.Linear):   #Selecting linear or convolutionnal layers
        nn.init.kaiming_uniform_(m.weight)                               #Kaiming initialisation for the weigths
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)                                   #If layers have biais-> biais to zero


#DISCRIMINATEUR
class Discriminator (nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.frf_length = 3001
    def DBlock(self, inp, out):
      return nn.Sequential(
        nn.Linear(inp, out),
        nn.BatchNorm1d(out),
        nn.LeakyReLU(0.2))
    
    self.linear = nn.Sequential(
        DBlock(self.frf_length, 512),
        DBlock(512, 128),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    #Initialization of layer weights for faster, more stable convergence during the training
    self._initialize_weights()

  def forward(self, frf):
    x = self.model(frf)
    return x

  def _initialize_weights(self):
    for m in self.modules():                                    #Iterations through the layers
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):  #Selecting linear or convolutionnal layers
        nn.init.kaiming_uniform_(m.weight)                      #Kaiming initialisation for the weigths
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)                          #If layer has biais-> biais to zero


#-------------------------------------------------------TRAINING----------------------------------------------------------

#PARAMETERS
data_path = "/content/data/step5/freq"                #location of the frf data
excel_file = "/content/data/step5/coordinates.xlsx"   #location of the coordinates
frf_length = 3001
input_size = 6
batch_size = 15
lr_D = 0.00001
lr_G = 0.0001
num_epochs = 1000

#DATASET
dataset=FRFDataset(excel_file, data_path, batch_size)


#TOOLS
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')       #definition of the device to use

generator = Generator(frf_length).to(device)                                  #initilisation of the generator
discriminator = Discriminator().to(device)                                    #initilisation of the discriminator

Loss = nn.BCELoss().to(device)                                                #Loss function
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_G)               #Generator Optimizer
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D)           #Discriminator Optimizer

D_loss_history = []                                                           #initialisation of lists for loss history
G_loss_history = []                                                           


def plot_generated_frf(frf_generated, data):
  #function for ploting the results
  plt.subplot(2,1,1)
  plt.plot(frf_generated[0,0,:].cpu().detach().numpy(), label='Generated FRF')
  plt.xlabel('Frequency')
  plt.ylabel('Magnitude')
  plt.legend()
  plt.subplot(2,1,2)
  plt.plot(data[0,0,:].cpu().detach().numpy(), label='Real FRF')
  plt.xlabel('Frequency')
  plt.ylabel('Magnitude')
  plt.legend()
  plt.show()


#TRAINING LOOP
for epochs in range (num_epochs):
  for i, (data, coordinates) in enumerate(dataset):
    data = data.unsqueeze(1)
    data = data.to(device)
    coordinates = coordinates.to(device)

    ###Discriminator's training
    optimizer_D.zero_grad()                                                 #Making the gradiant to zero
    generated_data = generator(coordinates)                                 
    discriminator_output = discriminator(generated_data)                    
    real_labels = torch.ones_like(discriminator_output).to(device)          #Perfect result of the discriminator for real data
    fake_labels = torch.zeros_like(discriminator_output).to(device)         #Perfect result of the discriminator for generated data

    disc_output = discriminator(data)                                       
    real_loss = Loss(disc_output, real_labels)                              #Difference between Disc and perfect result for real data
    fake_loss = Loss(discriminator(generated_data.detach()), fake_labels)   #Difference between Disc and perfect result for generated data
    d_loss = (real_loss + fake_loss)/2

    d_loss.backward()                                                       #Backpropagating the result of the loss
    optimizer_D.step()

    ###Generator's training
    optimizer_G.zero_grad()                                                 #Making the gradiant to zero
    generated_data_2 = generator(coordinates)                               
    discriminator_output_2 = discriminator(generated_data_2)                
    g_loss = Loss(discriminator_output_2, real_labels)                      #Difference between Disc result for generated data and perfect result for real data
    g_loss.backward()
    optimizer_G.step()                                                      #Backpropagating the result of the loss
    
    
    G_loss_history.append(g_loss.item())
    D_loss_history.append(d_loss.item())

    if i%100 == 0:
      plot_generated_frf(generated_data, data)
      print(f"Epochs [{epochs+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}],"
      f"G Loss: {g_loss.item(): .4f}, D Loss: {d_loss.item(): .4f}")
      plt.plot(D_loss_history, label='Disc', color='green')
      plt.legend()
      plt.plot(G_loss_history, label='Gen')
      plt.legend()
      plt.show()
