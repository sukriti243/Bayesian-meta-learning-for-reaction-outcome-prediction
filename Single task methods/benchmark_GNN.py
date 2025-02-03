import os
import time
import torch
import dgl
import random
import logging
import numpy as np
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from get_tasks_graph import load_tasks_graph
from sklearn.metrics import average_precision_score, roc_auc_score
from model_graph import MPNN
from sklearn.preprocessing import StandardScaler


class reactionMPNN(nn.Module):

    def __init__(self, node_in_feats, edge_in_feats,
                 readout_feats = 1544,
                 predict_hidden_feats1 = 1024, predict_hidden_feats2 = 512, prob_dropout = 0.1):
        
        super(reactionMPNN, self).__init__()

        self.mpnn = MPNN(node_in_feats, edge_in_feats)

        self.predict = nn.Sequential(
            nn.Linear(readout_feats, predict_hidden_feats1), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats1, predict_hidden_feats2), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(predict_hidden_feats2, 2)
        )
    
    def forward(self, batch, train:bool):
        scaler = StandardScaler()
        self.predict = self.predict.double()
        support_features, y_support, query_features, y_query = batch   

        # SUPPORT FEATURES
        support_reactant = support_features['reactant_graph'].values.tolist()
        support_ligand = support_features['ligand_graph'].values.tolist()
        support_solvent = support_features['solvent_graph'].values.tolist()
        
        gs_react = dgl.batch(support_reactant)
        gs_lig = dgl.batch(support_ligand)
        gs_solv = dgl.batch(support_solvent)
        
        support_reactant_graph = self.mpnn(gs_react.to(device))
        support_ligand_graph = self.mpnn(gs_lig.to(device))
        support_solvent_graph = self.mpnn(gs_solv.to(device))
        support_condition = scaler.fit_transform(support_features[['pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2']].values.tolist())
        support_condition = torch.tensor(support_condition).to(device)
    
        # QUERY FEATURES
        query_reactant = query_features['reactant_graph'].values.tolist()
        query_ligand = query_features['ligand_graph'].values.tolist()
        query_solvent = query_features['solvent_graph'].values.tolist()
        
        gs_react_query = dgl.batch(query_reactant)
        gs_lig_query = dgl.batch(query_ligand)
        gs_solv_query = dgl.batch(query_solvent)
        
        query_reactant_graph = self.mpnn(gs_react_query.to(device))
        query_ligand_graph = self.mpnn(gs_lig_query.to(device))
        query_solvent_graph = self.mpnn(gs_solv_query.to(device))
        query_condition = scaler.fit_transform(query_features[['pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2']].values.tolist())
        query_condition = torch.tensor(query_condition).to(device)
        
        if train:
            graph_features = torch.cat([support_reactant_graph, support_ligand_graph, support_solvent_graph, support_condition], axis=1)
            out = self.predict(graph_features)
            
        else:
            graph_features = torch.cat([query_reactant_graph, query_ligand_graph, query_solvent_graph, query_condition], axis=1)
            out = self.predict(graph_features)
        
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_mpnn_model():
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks_graph()
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []

    loss_fn = nn.CrossEntropyLoss()
    net = reactionMPNN(52, 8).to(device)
    n_epochs = 300
    optimizer = Adam(net.parameters(), lr = 1e-3, weight_decay = 1e-5)
    lr_scheduler = None 

    n_support = 8
    n_query = 128
    n_trials = 1
    for i in range(0, n_trials):
        
        print(f'Starting trial {i}')

        for task in test_tasks:
            # sample a task
            print(f'Task {task}')
            data_support = train_dfs[task]
            data_query = test_dfs[task]

            indices = list(np.arange(len(data_support)))
            np.random.shuffle(indices)
            support_indices = np.sort(indices[0:n_support])

            X_support = data_support[['reactant_graph','ligand_graph','solvent_graph','pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2']]
            X_support = X_support.iloc[support_indices.astype(int)]
            y_support = data_support['ee_class'].values.tolist()
            y_support = np.array(y_support)[support_indices.astype(int)]
            y_support = torch.tensor(np.array(y_support))
            y_support =  y_support.type(torch.LongTensor)  
            y_support = y_support.to(device)

            indices = list(np.arange(len(data_query)))
            np.random.shuffle(indices)
            query_indices = np.sort(indices[0:n_query])

            X_query = data_query[['reactant_graph','ligand_graph','solvent_graph','pressure','temperature','S/C','metal_1','metal_2','metal_3','add_1','add_2']]
            X_query = X_query.iloc[query_indices.astype(int)]
            y_query = data_query['ee_class'].values.tolist()
            y_query = np.array(y_query)[query_indices.astype(int)]
            y_query = torch.tensor(np.array(y_query))
            y_query = y_query.type(torch.LongTensor)  
            y_query = y_query.to(device)

            batch = (X_support, y_support, X_query, y_query)

            for epoch in range(n_epochs):
                # training
                net.train()
                
                pred = net(batch, train=True)
                loss = loss_fn(pred, y_support)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss = loss.detach().item()

            with torch.no_grad():
                outputs = net(batch, train=False)
                _, y_pred = torch.max(outputs, 1)
                
           
            task_labels.append(y_query.cpu().numpy())
            task_preds.append(y_pred.cpu().numpy())

        predictions = np.concatenate(task_preds, axis=0)
        labels=np.concatenate(task_labels, axis=0)  

        if np.sum(labels) == len(labels) or np.sum(labels) == 0:
            roc_auc = 0.0
        else:
            roc_auc = roc_auc_score(labels, predictions)
        
        prec=average_precision_score(labels, predictions)

        task_roc.append(roc_auc)
        task_prec.append(prec)

    task_roc = np.array(task_roc)
    task_prec = np.array(task_prec)
        
    print("ROC_score: {:.4f} +- {:.4f}\n".format(np.mean(task_roc), np.std(task_roc)))
    print("Avg_prec: {:.4f} +- {:.4f}\n".format(np.mean(task_prec), np.std(task_prec)))  

    return None

evaluate_mpnn_model()