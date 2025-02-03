# adapted from https://github.com/Wenlin-Chen/ADKF-IFT/blob/main/fs_mol

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time
import numpy as np
import pandas as pd
import torch
import random

from proto_meta import PrototypicalNetwork
from get_tasks import load_tasks
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score


@dataclass(frozen=True)
class PrototypicalNetworkTrainerConfig():
    tasks_per_batch: int = 5
    num_support: int = 512
    num_query: int = 256
    
    num_train_steps: int = 1000
    validate_every_num_steps: int = 50
    batch_size: int = 256

    learning_rate: float = 0.0001
    clip_value: Optional[float] = 1.0
    distance_metric: str = "mahalanobis"
    

def run_on_batches(model, batch, train: bool = False):
    if train:
        model.train()
    else:
        model.eval()

    task_preds: List[np.ndarray] = []
    task_labels: List[np.ndarray] = []
    support_features, support_target, query_features, query_target = batch

    # Compute task loss
    batch_logits = model(batch)
    
    if train:
        batch_loss = model.compute_loss(batch_logits, query_target)
        batch_loss.backward()
    
    # compute metric at test time
    else:
        with torch.no_grad():
            batch_preds = torch.nn.functional.softmax(batch_logits, dim=1).detach().cpu().numpy()

    if train:
        sample_loss = batch_loss.detach().cpu().numpy()
        metrics = None
    else:
        sample_loss = None
        metrics = batch_preds[:,1]

    return sample_loss, metrics
   

def evaluate_protonet_model(model, device):
    task_labels = []
    task_preds = []
    n_trials = 5

    # Load the dataset
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()

    print('\nBeginning training loop...')
    for i in range(0, n_trials):
        print(f'Starting trial {i}')
        for task in val_tasks:
            # sample a task
            data_support = train_dfs[task]
            data_query = val_dfs[task]
            scaler = StandardScaler()

            X_support = data_support.iloc[:,0:1544]
            X_scaled_support = scaler.fit_transform(X_support)
            X_support = torch.tensor(np.array(X_scaled_support), dtype=torch.float32).to(device)
            y_support = (data_support['ee_class']).values.tolist()
            y_support = torch.tensor(np.array(y_support), dtype=torch.float32).flatten().to(device)

            X_query = data_query.iloc[:,0:1544]
            X_scaled_query = scaler.fit_transform(X_query)
            X_query = torch.tensor(np.array(X_scaled_query), dtype=torch.float32).to(device)
            y_query = (data_query['ee_class']).values.tolist()
            y_query = torch.tensor(np.array(y_query), dtype=torch.float32).flatten().to(device)
            
            batch = (X_support, y_support, X_query, y_query)

            _, batch_preds = run_on_batches(model, batch, train=False)
            task_labels.append(y_query.detach().cpu().numpy())
            task_preds.append(batch_preds)
   
    predictions = np.concatenate(task_preds, axis=0)
    labels=np.concatenate(task_labels, axis=0)

    if np.sum(labels) == len(labels) or np.sum(labels) == 0:
        roc_auc = 0.0
    else:
        roc_auc = roc_auc_score(labels, predictions)
    
    prec=average_precision_score(labels, predictions)

    print("ROC-AUC: ", roc_auc)
    print("Precision: ", prec) 

    return prec

def test_protonet_model(model, device):
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []
    
    n_trials = 10
    n_query = 128

    # Load the dataset
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()

    print('\nBeginning training loop...')
    for i in range(0, n_trials):
        print(f'Starting trial {i}')
        for task in test_tasks:
            # sample a task
            data_support = train_dfs[task]
            data_query = test_dfs[task]
            scaler = StandardScaler()

            X_support = data_support.iloc[:,0:1544]
            X_scaled_support = scaler.fit_transform(X_support)
            X_support = torch.tensor(np.array(X_scaled_support), dtype=torch.float32).to(device)
            y_support = (data_support['ee_class']).values.tolist()
            y_support = torch.tensor(np.array(y_support), dtype=torch.float32).flatten().to(device)

            indices = list(np.arange(len(data_query)))
            np.random.shuffle(indices)
            query_indices = np.sort(indices[0:n_query])

            # query set
            X_query = data_query.iloc[:,0:1544]
            X_scaled_query = scaler.fit_transform(X_query)
            X_query = X_scaled_query[query_indices.astype(int)]
            X_query = torch.tensor(np.array(X_query), dtype=torch.float32).to(device)
            y_query = (data_query['ee_class']).values.tolist()
            y_query = np.array(y_query)[query_indices.astype(int)]
            y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
            
            batch = (X_support, y_support, X_query, y_query)

            _, batch_preds = run_on_batches(model, batch, train=False)
            
            task_labels.append(y_query.detach().cpu().numpy())
            task_preds.append(batch_preds)
            
        predictions = np.concatenate(task_preds, axis=0)
        labels=np.concatenate(task_labels, axis=0)
        
        roc_auc = roc_auc_score(labels, predictions)
        prec=average_precision_score(labels, predictions)

        task_roc.append(roc_auc)
        task_prec.append(prec)
   
    task_roc = np.array(task_roc)
    task_prec = np.array(task_prec)
        
    print("ROC_score: {:.4f} +- {:.4f}\n".format(np.mean(task_roc), np.std(task_roc)))
    print("Avg_prec: {:.4f} +- {:.4f}\n".format(np.mean(task_prec), np.std(task_prec)))  

    return prec


class PrototypicalNetworkTrainer(PrototypicalNetwork):
    def __init__(self, config: PrototypicalNetworkTrainerConfig):
        super().__init__(config)
        self.config = config
        self.optimizer = torch.optim.Adam(self.parameters(), config.learning_rate, weight_decay=1e-5)
        self.lr_scheduler = None 

    def get_model_state(self) -> Dict[str, Any]:
        return {
            "model_config": self.config,
            "model_state_dict": self.state_dict(),
        }

    def save_model(
        self,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
    ):
        data = self.get_model_state()

        if optimizer is not None:
            data["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            data["epoch"] = epoch

        torch.save(data, path)

    def load_model_weights(
        self,
        path: str,
        load_task_specific_weights: bool,
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ):
        pretrained_state_dict = torch.load(path, map_location=device)

        for name, param in pretrained_state_dict["model_state_dict"].items():
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            self.state_dict()[name].copy_(param)

        optimizer_weights = pretrained_state_dict.get("optimizer_state_dict")
        if optimizer_weights is not None:
            for name, param in optimizer_weights.items():
                self.optimizer.state_dict()[name].copy_(param)

    @classmethod
    def build_from_model_file(
        cls,
        model_file: str,
        config_overrides: Dict[str, Any] = {},
        quiet: bool = False,
        device: Optional[torch.device] = None,
    ) -> "PrototypicalNetworkTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        model = PrototypicalNetworkTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            load_task_specific_weights=True,
            device=device,
        )
        return model

    def train_loop(self, out_dir, device: torch.device):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))
        best_validation_avg_prec = 0.0

        data = pd.read_csv("/homes/ss2971/Documents/AHO/AHO_FP/train_task_identity.csv")
        train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()
        n_support = self.config.num_support
        n_query = self.config.num_query
        start_time = time.time()
        for step in range(1, self.config.num_train_steps + 1):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            task_batch_losses: List[float] = []
            # RANDOMISE ORDER OF TASKS PER EPISODE
            shuffled_train_tasks = random.sample(train_tasks, len(train_tasks))
            
            for task in shuffled_train_tasks[:self.config.tasks_per_batch]:
                X = data.iloc[:,0:1544]
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                y = (data['ee_class']).values.tolist()
                
                indices = list(np.arange(len(X)))
                np.random.shuffle(indices)
                support_indices = np.sort(indices[0:n_support])
                query_indices = np.sort(indices[n_support:n_support+n_query])

                # support set
                X_support = X_scaled[support_indices.astype(int)]
                X_support = torch.tensor(X_support, dtype=torch.float32).to(device)
                y_support = np.array(y)[support_indices.astype(int)]
                y_support = torch.tensor(y_support, dtype=torch.float32).flatten().to(device)

                while (y_support!=0).all() or (y_support!=1).all():
                    np.random.shuffle(indices)
                    support_indices = np.sort(indices[0:n_support])
                    query_indices = np.sort(indices[n_support:n_support+n_query])
                    X_support = X_scaled[support_indices.astype(int)]
                    X_support = torch.tensor(X_support, dtype=torch.float32).to(device)
                    y_support = torch.tensor(np.array(y)[support_indices.astype(int)]).to(device)

                # query set
                X_query = X_scaled[query_indices.astype(int)]
                X_query = torch.tensor(X_query, dtype=torch.float32).to(device)
                y_query = np.array(y)[query_indices.astype(int)]
                y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
                
                batch = (X_support, y_support, X_query, y_query)
                task_loss, _ = run_on_batches(self, batch=batch, train=True)
                task_batch_losses.append(task_loss)

            # Now do a training step 
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_value)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            if (step%1==0):
                print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(step, self.optimizer.param_groups[-1]['lr'], task_batch_mean_loss, (time.time()-start_time)/60))

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = evaluate_protonet_model(self, device)

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_avg_prec:
                    best_validation_avg_prec = valid_metric
                    model_path = os.path.join(out_dir, "best_validation_proto.pt")
                    self.save_model(model_path)
                    print('model updated at train step: ', step)

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained_proto.pt"))