from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import smiles
import json


def into_dataset(file:str, with_hydrogen=False, kekulize=False):
    with open(file, 'r+') as f:
        j = json.load(f)


    xy = [[part[1], part[2]] for part in j]

    for idx, _ in enumerate(xy):

        xy[idx] = {
            'x_smile': xy[idx][0],
            'x': smiles.from_smiles(xy[idx][0], with_hydrogen=with_hydrogen, kekulize=kekulize),
            'ld50': xy[idx][1]
        }
    
    return xy

class Data(Dataset):
    '''
        dataset format for SMILES dataset.

        Look for model:
            https://pytorch-geometric.readthedocs.io/en/latest/notes/cheatsheet.html
            that can fit the given format
    '''
    def __init__(self, data):
        self.data = data
        # each index : x_smile: str, x: tensor, ld50: float

    def __getitem__(self, index, return_name=False):
        if return_name:
            return self.data.x, self.data.ld50, self.data.x_smile
        return self.data.x, self.data.ld50
    
    def __len__(self):
        return len(self.data)