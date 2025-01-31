import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
try:
    import wandb
except ImportError: 
    raise Exception("Note that wandb is required to track the progress.")
from datetime import datetime
from dataload import into_dataset, Data
from model import GNN1
config = {
    
    'dataset': 'dataset/ld50_smiles.json',
    'in_chn': 9,
    'out_chn': 128,
    'in_edge': 3,
    'batch_size': 32,
    'num_workers': 4,
    'epoch': 100
}

class trainer:
    '''
        trainer module
            train
            eval
            visualize
    '''


    def __init__(self, config):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = f"{datetime.now()}"
        wandb.init(
            config={
                
                **config,
                'device': "cuda" if torch.cuda.is_available() else "cpu",
                'gpu_type': "None" if not torch.cuda.is_available() else torch.cuda.get_device_name(device),
            },
            project="LD50-Prediction",
            name=name
        )

        self.config = config
        self.model = GNN1(in_chn=config.in_chn, out_chn=config.out_chn, in_edge=config.in_edge)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.5e-4)
        self.loss = nn.MSELoss()
        self.dataset = Data(into_dataset(file=config.dataset))
        dlen = len(self.dataset)
        self.train_dataset = Subset(self.data, [i for i in range(dlen - 300)])
        self.test_dataset = Subset(self.data, [i for i in range(dlen - 300, dlen - 1, 1)])
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers,
            shuffle=True)
        
        self.test_loader = DataLoader(
            self.test_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers,
            shuffle=False)
        

    def train(self):
        epoch = self.config.epoch
        for e in range(epoch):
            for idx, (data, y) in self.train_loader:
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = self.loss(out, y)
                loss.backward()
                self.optimizer.step()
                wandb.log({
                    "train/loss": loss.detach()
                })

        

        



    def eval(self, ):
        pass


    def visualize(self):
        pass




if __name__ == "__main__":
    t = trainer(config)
    t.train()