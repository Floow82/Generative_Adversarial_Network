
""" WASSERSTEIN GENERATIVE ADVERSARIAL NETWORK -- Models and training """

###Importations
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#--------------------------------------MODELS---------------------------------------------------------

#GENERATOR
class Generator (nn.Module):

  def __init__(self, frf_length, input_size):
    super(Generator, self).__init__()
    self.frf_length = frf_length
    self.input_size = input_size

    def GBlock(self, inp, out, a, b, c):
      return nn.Sequential(
        nn.Convtranspose1d(inp, out, a, b, c),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True))

    self.fc = nn.Linear(input_size ,64)
    self.deconv_layers = nn.Sequential(
        GBlock(64, 32, 4, 2, 1),
        GBlock(32, 16, 4, 2, 1),
        GBlock(16, 8, 4, 2, 1),
        GBlock(8, 4, 4, 2, 1),
        GBlock(4, 1, 4, 2, 1),
        nn.Linear(in_features=self._get_length, self.frf_length),
        nn.Tanh())

    self._initialize_weights()

  def forward (self, coordinates):
    x = self.fc(coordinates)
    x = x.view(-1,64,1).repeat(1,1, self.frf_length)  #making tensor at the good size
    x = self.deconv_layers(x)
    return x

  def _get_length(self):
    out_length = self.frf_length
    for layer in self.conv_layers:
      if isinstance(layer, nn.Conv1d):
        out_length = (out_length + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    return out_length

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m,nn.ConvTranspose2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

#DISCRIMINATOR
class Discriminator (nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()
    def DBlock(self, inp, out, a, b, c):
      return nn.Sequential(
        nn.Conv1d(inp, out, a, b, c),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True))

    self.conv_layers = nn.Sequential(
        DBlock(1, 4, 4,2,1),
        DBlock(4, 8,4, 2,1),
        DBlock(8, 16,4,2,1),
        DBlock(16, 32, 4,2,1),
        nn.Conv1d(32, 1, 4,2,1),
        nn.Tanh()
    )

    self._initialize_weights()

  def forward(self, frf):
    x = self.conv_layers(frf)
    x = x.view(x.size(0),-1)  #(batch_size, automatic_calculation)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
          nn.init.constant_(m.bias, 0)

#-------------------------------------------TRAINING--------------------------------------------

#PARAMETERS
data_path = "/content/data/step5/freq"                #location of the frf data
excel_file = "/content/data/step5/coordinates.xlsx"   #location of the coordinates
frf_length = 3001
input_size = 6
batch_size = 20
lr_D = 0.00001
lr_G = 0.0001
num_epochs = 1000
crit_cycles = 5
lambda_gp = 10

#DATASET
dataset=FRFDataset(excel_file, data_path, batch_size)

#TOOLS
generator = Generator(frf_length).to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

D_loss_history = []
G_loss_history = []

def plot_generated_frf(frf_generated, data):
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

#GRADIENT PENALTY CALCULATION
def compute_gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand((real_data.size(0), 1, 1), device=real_data.device,requires_grad=True)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size(), device=real_data.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


#TRAINING LOOP


for epochs in range (num_epochs):
  for i, (data, coordinates) in enumerate(dataloader):
    data = data.unsqueeze(1)
    data = data.to(device)
    #print(data.shape)
    coordinates = coordinates.to(device)

    ###Discriminator
    mean_disc_loss = 0
    for _ in range(crit_cycles):
      optimizer_D.zero_grad()
      generated_data = generator(coordinates).detach()
      real_loss = - torch.mean(discriminator(data))
      fake_loss = torch.mean(discriminator(generated_data))
      gradient_penalty = compute_gradient_penalty(discriminator, data, generated_data)
      d_loss = real_loss + fake_loss + gradient_penalty * lambda_gp
      d_loss.backward()
      optimizer_D.step()

    D_loss_history+=[d_loss.item()]

    ###Generator
    optimizer_G.zero_grad()
    generated_data_2 = generator(coordinates)
    discriminator_output_2 = discriminator(generated_data_2)
    g_loss = -torch.mean(discriminator_output_2)
    g_loss.backward()
    optimizer_G.step()
    G_loss_history+=[g_loss.item()]

    if i%100 == 0:
      plot_generated_frf(generated_data, data)
      gen_mean=sum(G_loss_history[-100:]) / 100
      crit_mean = sum(D_loss_history[-100:]) / 100
      print(f"Epoch [{epochs+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], G Loss: {g_loss.item(): .4f}, D Loss: {d_loss.item(): .4f}")
      plt.plot(D_loss_history, label='Discriminator_loss', color='green')
      plt.legend()
      plt.plot(G_loss_history, label='Generator_loss', color='red')
      plt.legend()
      plt.show()