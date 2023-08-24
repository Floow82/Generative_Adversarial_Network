
"""Generative Adversarial Network with Linear layers -- Dataloader"""
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import gdown, zipfile
from PIL import Image
import numpy as np
import pandas as pd

#DOWNLOADING THE DATA FROM DROP BOX
def Get_data(url, path):
    url='https://www.dl.dropboxusercontent.com/scl/fi/2mnsa9c3xzbkg3h6cvawb/step5.zip?rlkey=svrqtvt891h8gszx3yabt5e8a&dl=0'
    path='data'
    download_path = f'{path}/step'
    if not os.path.exists(path):
        os.makedirs(path)
    gdown.download(url, download_path, quiet=False)           #Downloading .zip file in the path

    with zipfile.ZipFile(download_path, 'r') as ziphandler:
        ziphandler.extractall(path)                           #Extracting the files from .zip


#DEFINITION OF THE DATASET
class FRFDataset(Dataset):

  def __init__(self, excel_file, path, width=256, height=256, lim=1000):
    super().__init__()
    self.coord = pd.read_excel(excel_file)
    self.frf_folder = path

  def __len__(self):
    return len(self.coord)

  def __getitem__(self, idx):
    #FRF Data preparation
    frf_file = os.path.join(self.frf_folder, self.coord.iloc[idx,0])              #Joining frf data and coordinates
    frf_data = pd.read_excel(frf_file, header=None, skiprows = 1)                 
    frf_data = frf_data.iloc[:, 1]                                                #Keeping the second column (not the frequency)
    frf_data = np.asarray(frf_data).astype(np.float32)                            #Making array and changing its type
    frf_data = torch.from_numpy(frf_data)                                         #Creating the tensor

    #data checking
    frf_data[torch.isnan(frf_data)]=0                                             #Checking the 'NaN' values

    #Cordinates preparation
    coordinate = self.coord.loc[idx, ['x_e','y_e','z_e','x_s','y_s','z_s']]       #Reading the coordinates
    coordinates = coordinates.values.astype(np.float32)                           #Changinf the type

    self.dataset = frf_data, coordinate
  
  def __create__(self, batch_size):
    dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle = True)  #Creating the dataloader with the mini batch and shuffle
    return dataloader
     

