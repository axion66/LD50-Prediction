import torch_geometric.nn as nng
import torch
import torch.nn as nn



class GNN1(nn.Module):
    def __init__(self, in_chn, out_chn, in_edge):
        self.in_chn = in_chn
        self.hidden_chn = in_chn * 4
        self.out_chn = out_chn
        self.in_edge = in_edge

        self.c1 = nng.GeneralConv(in_channels=in_chn, out_channels=in_chn * 4, in_edge_channels=in_edge)
        self.a1 = nn.GELU(approximate='tanh')
        self.c2 = nng.GATv2Conv(in_channels=in_chn * 4, out_channels=out_chn, heads=1, edge_dim=in_edge)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_features=out_chn, out_features=out_chn*4),
            nn.GELU(approximate='tanh'),
            nn.Linear(in_features=out_chn*4, out_features=1)
        )

    def forward(self, data):
        # has to be batch-form
        x, edge_index, edge_attr, batch = [data.x, data.edge_index, data.edge_attr, data.batch]
        
        x = self.c1(x, edge_index, edge_attr)
        x = self.a1(x)
        x = self.c2(x, edge_index, edge_attr)
        x = nng.global_mean_pool(x, batch)
        x = self.ffn(x)
        return x
