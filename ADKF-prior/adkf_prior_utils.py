import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import time
import numpy as np
import pandas as pd
import torch
import random
import gpytorch
import higher

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from adkf_prior import ADKFPriorMetaModel
from get_tasks import load_tasks

from botorch.optim.fit import fit_gpytorch_scipy
from sklearn.metrics import roc_auc_score, average_precision_score

from hypergradient import hypergradient, apply_grad, mix_grad
from _stateless import functional_call


@dataclass(frozen=True)
class ADKFPriorMetaModelTrainerConfig():
    num_support: int = 512
    num_query: int = 64
    tasks_per_batch: int = 5
    
    num_inner_iters: int = 100
    num_train_steps: int = 1000
    validate_every_num_steps: int = 50
    
    learning_rate: float = 0.001
    clip_value: Optional[float] = None

    use_ard: bool = False
    gp_kernel: str = "matern"
    use_lengthscale_prior: bool = False
    ignore_grad_correction: bool = False
    

def run_on_batches(model: ADKFPriorMetaModel, batch, train: bool = False):

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


def evaluate_ADKFPrior_meta_model(model: ADKFPriorMetaModel, device):
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
            model.reinit_feature_extractor_params()
            with higher.innerloop_ctx(model, model.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt): 

                for step_i in range(model.config.num_inner_iters):
                    support_loss = fmodel(batch, train_loss=True, is_functional_call=True)
                    diffopt.step(support_loss)
                
                batch_preds = run_on_batches(fmodel, batch, train=False)
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

def test_ADKFPrior_model(model, device):
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
            model.reinit_feature_extractor_params()
            with higher.innerloop_ctx(model, model.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt): 

                for step_i in range(model.config.num_inner_iters):
                    support_loss = fmodel(batch, train_loss=True, is_functional_call=True)
                    diffopt.step(support_loss)
                
                batch_preds = run_on_batches(fmodel, batch, train=False)
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

class ADKFPriorMetaModelTrainer(ADKFPriorMetaModel):
    def __init__(self, config: ADKFPriorMetaModelTrainerConfig):
        super().__init__(config)
        # print(config)
        self.config = config
        self.inner_optimizer = torch.optim.Adam([{'params': self.feature_extractor_params(), 'lr': config.learning_rate}, 
                                                 {'params': self.gp_params(), 'lr': config.learning_rate/0.1}])
        self.outer_optimizer = torch.optim.Adam(self.prior_params(), config.learning_rate*0.1)
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
    ) -> "ADKFPriorMetaModelTrainer":
        """Build the model architecture based on a saved checkpoint."""
        checkpoint = torch.load(model_file, map_location=device)
        config = checkpoint["model_config"]

        model = ADKFPriorMetaModelTrainer(config)
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
            self.reinit_feature_extractor_params()
            grad_list = []
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
                
                with higher.innerloop_ctx(self, self.inner_optimizer, track_higher_grads=False) as (fmodel, diffopt):  

                    for i in range(self.config.num_inner_iters):                    
                        support_loss = fmodel(batch, train_loss=True, is_functional_call=True)
                        diffopt.step(support_loss)
                    
                    support_loss = fmodel(batch, train_loss=True, is_functional_call=True) 
                    query_loss = fmodel(batch, train_loss=False, predictive_val_loss=True, is_functional_call=True)
                    
                    task_loss = query_loss / y_query.shape[0] #  report per-sample loss
                    task_loss = task_loss.cpu().item()
                    task_batch_losses.append(task_loss)
                    
                    params = fmodel.feature_extractor_params()
                    hparams = fmodel.prior_params()

                    implicit_grad = hypergradient(query_loss, support_loss, hparams, params) 
                    grad_list.append(implicit_grad)
                
            self.outer_optimizer.zero_grad()
            weight = torch.ones(len(grad_list))
            weight = weight/torch.sum(weight)
            grad = mix_grad(grad_list, weight)
            _ = apply_grad(self, grad)
            
            # Now do a training step - run_on_batches will have accumulated gradients
            if self.config.clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.prior_params(), self.config.clip_value)
            self.outer_optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            task_batch_mean_loss = np.mean(task_batch_losses)
            if (step%1==0):
                print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(step, self.outer_optimizer.param_groups[-1]['lr'], task_batch_mean_loss, (time.time()-start_time)/60))

            if step % self.config.validate_every_num_steps == 0:
                valid_metric = evaluate_ADKFPrior_meta_model(self, device)

                # save model if validation avg prec is the best so far
                if valid_metric > best_validation_avg_prec:
                    best_validation_avg_prec = valid_metric
                    model_path = os.path.join(out_dir, "best_validation_adkfprior.pt")
                    self.save_model(model_path)
                    print('model updated at train step: ', step)
           
        # save the fully trained model
        self.save_model(os.path.join(out_dir, "fully_trained_adkfprior.pt"))