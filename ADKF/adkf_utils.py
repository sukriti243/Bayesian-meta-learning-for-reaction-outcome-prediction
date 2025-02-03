import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time
import numpy as np
import pandas as pd
import torch
import random
import gpytorch

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from adkf import ADKFModel 
from get_tasks import load_tasks

from botorch.optim.fit import fit_gpytorch_scipy
from sklearn.metrics import roc_auc_score, average_precision_score

from cauchy_hypergradient import cauchy_hypergradient
from _stateless import functional_call


@dataclass(frozen=True)
class ADKFModelTrainerConfig():
    num_support: int = 512
    num_query: int = 64
    tasks_per_batch: int = 5
    
    num_train_steps: int = 1000
    validate_every_num_steps: int = 50
    
    learning_rate: float = 0.001
    clip_value: Optional[float] = None

    use_ard: bool = False
    gp_kernel: str = "matern"
    use_lengthscale_prior: bool = False
    ignore_grad_correction: bool = False
    

def run_on_batches(model: ADKFModel, batch, train: bool = False):

    # Compute task loss
    model.train()
    _ = model(batch, train_loss=True)
    fit_gpytorch_scipy(model.mll)

    # Compute train loss on the support set at training time
    if not train:
        model.eval()
        with torch.no_grad():
            batch_logits = model(batch, train_loss=None)
            batch_preds = torch.sigmoid(batch_logits.mean).detach().cpu().numpy()

    if train:
        metrics = None
    else:
        metrics = batch_preds
            
    return metrics


def evaluate_adkf_model(model: ADKFModel, device):
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []

    n_trials = 10
    n_support = 8
    n_query = 128

    # Load the dataset
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()

    print('\nBeginning training loop...')
    for i in range(0, n_trials):
        print(f'Starting trial {i}')
        for task in val_tasks:
            # sample a task
            data = val_dfs[task]
            X = data.iloc[:,0:1543]
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

            # query set
            X_query = X_scaled[query_indices.astype(int)]
            X_query = torch.tensor(X_query, dtype=torch.float32).to(device)
            y_query = np.array(y)[query_indices.astype(int)]
            y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
            
            batch = (X_support, y_support, X_query, y_query)
            batch_preds = run_on_batches(model, batch, train=False)
            
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

    return np.mean(task_roc)

def test_adkf_model(model, device):
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []

    n_trials = 10
    n_support = 8
    n_query = 128

    # Load the dataset
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()

    print('\nBeginning training loop...')
    for i in range(0, n_trials):
        print(f'Starting trial {i}')
        for task in test_tasks:
            # sample a task
            data = test_dfs[task]
            X = data.iloc[:,0:1543]
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

            # query set
            X_query = X_scaled[query_indices.astype(int)]
            X_query = torch.tensor(X_query, dtype=torch.float32).to(device)
            y_query = np.array(y)[query_indices.astype(int)]
            y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
            
            batch = (X_support, y_support, X_query, y_query)
            batch_preds = run_on_batches(model, batch, train=False)
            
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

    return None


class ADKFModelTrainer(ADKFModel):
    def __init__(self, config: ADKFModelTrainerConfig):
        super().__init__(config)
        self.config = config
        self.optimizer = torch.optim.Adam(self.feature_extractor_params(), config.learning_rate, weight_decay=1e-5)
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
        #load_task_specific_weights: bool,
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
    ) -> "ADKFModelTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        model = ADKFModelTrainer(config)
        model.load_model_weights(
            path=model_file,
            quiet=quiet,
            #load_task_specific_weights=True,
            device=device,
        )
        return model

    def train_loop(self, out_dir, device: torch.device):
        self.save_model(os.path.join(out_dir, "best_validation.pt"))
        best_validation_avg_prec = 0.0

        train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()
        n_support = self.config.num_support
        n_query = self.config.num_query
        start_time = time.time()
        for step in range(1, self.config.num_train_steps + 1):
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()

            grad_accum = [0.0 for p in self.feature_extractor_params()]

            task_batch_losses: List[float] = []
            # RANDOMISE ORDER OF TASKS PER EPISODE
            shuffled_train_tasks = random.sample(train_tasks, len(train_tasks))
        
            # find the best GP parameters given the current GNN parameters
            for task in shuffled_train_tasks[:self.config.tasks_per_batch]:
                data = train_dfs[task]
                X = data.iloc[:,0:1543]
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

                # query set
                X_query = X_scaled[query_indices.astype(int)]
                X_query = torch.tensor(X_query, dtype=torch.float32).to(device)
                y_query = np.array(y)[query_indices.astype(int)]
                y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
                
                batch = (X_support, y_support, X_query, y_query)
                _ = run_on_batches(self, batch=batch, train=True)

                # compute gradient w.r.t. GNN parameters with GP parameters set to the best using the validation loss
                self.train()

                feature_extractor_params_names = [n for n, _ in self.named_parameters() if not n.startswith("gp_")]
                gp_params_names = [n for n, _ in self.named_parameters() if n.startswith("gp_")]

                def f_inner(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(self, self_params_dict, (batch,), kwargs={"train_loss": True, "is_functional_call": True})
                    return batch_loss

                def f_outer(params_outer, params_inner):
                    feature_extractor_params_dict = {n: p for n, p in zip(feature_extractor_params_names, params_outer)}
                    gp_params_dict = {n: p for n, p in zip(gp_params_names, params_inner)}
                    self_params_dict = {**feature_extractor_params_dict, **gp_params_dict}
                    batch_loss = functional_call(self, self_params_dict, (batch,), kwargs={"train_loss": False, "predictive_val_loss": True, "is_functional_call": True})
                    return batch_loss

                batch_loss = cauchy_hypergradient(f_outer, f_inner, tuple(self.feature_extractor_params()), tuple(self.gp_params()), self.device, ignore_grad_correction=self.config.ignore_grad_correction)
                task_loss = batch_loss 
                task_loss = task_loss.cpu().item()
                task_batch_losses.append(task_loss)

                for i, param in enumerate(self.feature_extractor_params()):
                    grad_accum[i] += param.grad.data.clone() / self.config.tasks_per_batch

            for i, param in enumerate(self.feature_extractor_params()):
                param.grad = grad_accum[i]
            # Now do a training step - run_on_batches will have accumulated gradients
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.feature_extractor_params(), self.config.clip_value)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            if (step%1==0):
                print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(step, self.optimizer.param_groups[-1]['lr'], task_batch_mean_loss, (time.time()-start_time)/60))

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = evaluate_adkf_model(self, device)

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_avg_prec:
                    best_validation_avg_prec = valid_metric
                    model_path = os.path.join(out_dir, "best_validation_adkf.pt")
                    self.save_model(model_path)
                    print('model updated at train step: ', step)

        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained_adkf.pt"))