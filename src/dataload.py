from torch_geometric.data import Dataset, DataLoader
from torch_geometric.utils import smiles
import json
import os
import torch

import os
import json
import math

def into_dataset2(file = None, with_hydrogen=False, kekulize=False, output_file='smiles_numbers.txt'):
    if file is None or not os.path.exists(file):
        file = '../dataset/ld50_smiles.json'  # Fallback path if file not found
    
    # Open and load the JSON file
    with open(file, 'r+') as f:
        j = json.load(f)
    
    # Open the output file in write mode
    with open(output_file, 'w') as f_out:
        # Loop through each element in the JSON data
        for part in j:
            smiles = part[1]
            number = 1 if part[2] > 2000 else 0
            
            # Write the SMILES and the number to the output file with space in between
            f_out.write(f"{smiles} {number}\n")
    
    print(f"Data has been written to {output_file}")
    
   


def into_dataset(file:str, with_hydrogen=False, kekulize=False):
    if not os.path.exists(file):
        file = '../dataset/ld50_smiles.json'

    with open(file, 'r+') as f:
        j = json.load(f)

    x = [smiles.from_smiles(part[1], with_hydrogen=with_hydrogen, kekulize=kekulize) for part in j]
    for idx,_ in enumerate(x):
        x[idx].x.float()
        x[idx].edge_attr.float()

    y = [part[2] for part in j]    
    return x, y

class Data(Dataset):
    '''
        dataset format for SMILES dataset.

        Look for model:
            https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
            that can fit the given format
    '''
    def __init__(self, x, y, standard=True):
        import statistics

        self.x = x
        #self.y = y
        self.y = torch.FloatTensor(y).log10()
        
    def __getitem__(self, idx, return_name=False):
        #if return_name:
         #   return self.data[idx]['x'], self.data[idx]['ld50'], self.data[idx]['x_smile']
        return self.x[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)