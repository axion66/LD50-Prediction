import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset, random_split
import wandb
from datetime import datetime
from dataload import into_dataset, Data
from model import GNN1, GCN,WeightedGNN

config = {
    'dataset': 'dataset/ld50_smiles.json',
    'in_chn': 9,
    'out_chn': 128,
    'in_edge': 3,
    'batch_size': 1,
    'num_workers': 0,
    'epoch': 100
}

class Trainer:
    def __init__(self, config=config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        name = f"{datetime.now()}"
        
        wandb.init(
            config={**config, 'device': device, 'gpu_type': "None" if not torch.cuda.is_available() else torch.cuda.get_device_name(device)},
            project="LD50-Prediction",
            name=name,
        )

        self.config = config
        self.model = WeightedGNN(in_channels=config['in_chn'], hidden_channels=config['out_chn'], out_channels=config['in_edge']).float()
        #self.model = GNN1(in_chn=config['in_chn'], out_chn=config['out_chn'], in_edge=config['in_edge']).float()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2.5e-4)
        self.loss = nn.MSELoss()
        
        x, y = into_dataset(file=config['dataset'])
        self.dataset = Data(x, y)
        
        train_size = int(0.8 * len(self.dataset))  # 80% train, 20% test
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])

        self.train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'], shuffle=False)

    def train(self):
        epoch = self.config['epoch']
        for e in range(epoch):
            self.model.train()  # Ensure the model is in training mode
            total_loss = 0
            correct_train = 0
            total_train = 0

            for idx, batch in enumerate(self.train_loader):
                y = batch[1]
                batch = batch[0]
                print(batch.x.shape)
                                
                
                out = self.model(batch)
                
                # Loss and backward pass
                loss = self.loss(out.squeeze(-1), y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update total loss and accuracy
                total_loss += loss.item()
                _, predicted = torch.sigmoid(out).max(1)  # Apply sigmoid and get predicted class
                correct_train += (predicted == y).sum().item()
                total_train += y.size(0)

            # Calculate average loss and accuracy for this epoch
            avg_loss = total_loss / len(self.train_loader)
            train_accuracy = 100 * correct_train / total_train

            # Log metrics to WandB for training
            wandb.log({"train/loss": avg_loss, "train/accuracy": train_accuracy, "epoch": e + 1})
            print(f"Epoch [{e+1}/{epoch}], Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

    def eval(self):
        self.model.eval()  # Ensure the model is in evaluation mode
        correct = 0
        total = 0
        total_loss = 0

        with torch.no_grad():
            for batch in self.test_loader:
                x, edge_index, edge_attr, batch_ids = batch.x, batch.edge_index, batch.edge_attr, batch.batch
                y = batch.y
                
                out = self.model(batch)
                loss = self.loss(out.squeeze(-1), y)
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.sigmoid(out).max(1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)

        # Log evaluation metrics to WandB
        wandb.log({"eval/loss": avg_loss, "eval/accuracy": accuracy})
        print(f"Evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def visualize(self):
        pass

if __name__ == "__main__":
    t = Trainer(config)
    t.train()  # Train the model
    t.eval()   # Evaluate after training
