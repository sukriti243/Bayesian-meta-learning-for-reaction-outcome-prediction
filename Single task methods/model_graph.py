# https://github.com/seokhokang/reaction_yield_nn/blob/main/model.py

import torch
import gpytorch
import torch.nn as nn
from dgl.nn.pytorch import NNConv, Set2Set


class MPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats, hidden_feats = 64,
                 num_step_message_passing = 3, num_step_set2set = 3, num_layer_set2set = 1,
                 readout_feats = 512):
        
        super(MPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, hidden_feats), nn.ReLU()
        )
        
        self.num_step_message_passing = num_step_message_passing
        
        edge_network = nn.Linear(edge_in_feats, hidden_feats * hidden_feats)
        
        self.gnn_layer = NNConv(
            in_feats = hidden_feats,
            out_feats = hidden_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.activation = nn.ReLU()
        
        self.gru = nn.GRU(hidden_feats, hidden_feats)

        self.readout = Set2Set(input_dim = hidden_feats * 2,
                               n_iters = num_step_set2set,
                               n_layers = num_layer_set2set)

        self.sparsify = nn.Sequential(
            nn.Linear(hidden_feats * 4, readout_feats), nn.PReLU()
        )
             
    def forward(self, g):
        node_feats = g.ndata['attr']
        edge_feats = g.edata['edge_attr']
        
        node_feats = self.project_node_feats(node_feats)
        hidden_feats = node_feats.unsqueeze(0)

        node_aggr = [node_feats]        
        for _ in range(self.num_step_message_passing):
            node_feats = self.activation(self.gnn_layer(g, node_feats, edge_feats)).unsqueeze(0)
            node_feats, hidden_feats = self.gru(node_feats, hidden_feats)
            node_feats = node_feats.squeeze(0)
        
        node_aggr.append(node_feats)
        node_aggr = torch.cat(node_aggr, 1)
        
        readout = self.readout(g, node_aggr)
        graph_feats = self.sparsify(readout)
        
        return graph_feats

class GraphFeatureExtractor(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats):
        
        super(GraphFeatureExtractor, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)
    
    def forward(self, rmols):
        r_graph_feats = self.mpnn(rmols) 

        return r_graph_feats   
