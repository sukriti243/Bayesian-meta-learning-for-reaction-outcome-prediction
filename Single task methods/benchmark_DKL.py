import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

import gpytorch
from gpytorch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from get_tasks import load_tasks
from model import ExactGPLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

class DKL_Model(nn.Module):
    def __init__(self, gp_kernel, use_ard, use_lengthscale_prior):
        super().__init__()

        self.gp_kernel = gp_kernel
        self.use_ard = use_ard
        self.use_lengthscale_prior = use_lengthscale_prior

        # Create MLP if needed:
        self.fc_out_dim = 512
        # Determine dimension:
        fc_in_dim = 1543

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(1024, self.fc_out_dim)
        )
        
        if self.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None
        self.__create_tail_GP(kernel_type=self.gp_kernel, ard_num_dims=ard_num_dims, use_lengthscale_prior=self.use_lengthscale_prior)

    def __create_tail_GP(self, kernel_type, ard_num_dims=None, use_lengthscale_prior=False):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp_model = ExactGPLayer(
            train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, 
            kernel=kernel_type, ard_num_dims=ard_num_dims
        )

        if use_lengthscale_prior:
            scale = 0.25
            loc = 0.0
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
            self.gp_model.covar_module.base_kernel.register_prior(
                "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
            )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_prior.mean

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def save_gp_params(self):
        self.gp_model_params = deepcopy(self.gp_model.state_dict())
        self.gp_likelihood_params = deepcopy(self.gp_likelihood.state_dict())

    def load_gp_params(self):
        self.gp_model.load_state_dict(self.gp_model_params)
        self.gp_likelihood.load_state_dict(self.gp_likelihood_params)

    def forward(self, batch, train: bool):
        # self.fc = self.fc.double()
        support_features, support_target, query_features, query_target = batch
        
        support_features_flat = self.fc(support_features)
        query_features_flat = self.fc(query_features)

        if self.training and train:
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_target, strict=False)
            logits = self.gp_model(support_features_flat)
            
        else:
            assert self.training == False and train == False
            self.gp_model.train()

            self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_target, strict=False)
            
            self.gp_model.eval()
            self.gp_likelihood.eval()
            
            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))
            
        return logits

    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)
 
def run_on_batches():

    n_trials = 10
    n_support = 8
    n_query = 128

    # Load the dataset
    task_labels = []
    task_preds = []
    task_roc = []
    task_prec = []
    
    train_tasks, train_dfs, val_tasks, val_dfs, test_tasks, test_dfs = load_tasks()

    print('\nBeginning training loop...')

    for i in range(0, n_trials):
        
        print(f'Starting trial {i}')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

        for task in test_tasks:

            # Set parameters
            model = DKL_Model(gp_kernel='matern', use_ard=False, use_lengthscale_prior=False).to(device)
            n_epochs = 400
            optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3, weight_decay = 1e-5)
            lr_scheduler = None 
            clip_value = None
            scaler = StandardScaler()

            # sample a task
            data_train = pd.read_csv(f"/homes/ss2971/Documents/AHO/AHO_FP/train_tasks_wo_cluster.csv")
            data_test = test_dfs[task]

            indices = list(np.arange(len(data_test)))
            np.random.shuffle(indices)
            support_indices = np.sort(indices[0:n_support])
            query_indices = np.sort(indices[n_support:n_support+n_query])

            # support set
            X_support = data_train.iloc[:,0:1543]
            X_scaled_support = scaler.fit_transform(X_support)
            X_support = torch.tensor(np.array(X_scaled_support), dtype=torch.float32).to(device)
            y_support = (data_train['ee_class']).values.tolist()
            y_support = torch.tensor(np.array(y_support), dtype=torch.float32).flatten().to(device)

            # query set
            X_query = data_test.iloc[:,0:1543]
            X_scaled_query = scaler.fit_transform(X_query)
            X_query = X_scaled_query[query_indices.astype(int)]
            X_query = torch.tensor(np.array(X_query), dtype=torch.float32).to(device)
            y_query = (data_test['ee_class']).values.tolist()
            y_query = np.array(y_query)[query_indices.astype(int)]
            y_query = torch.tensor(y_query, dtype=torch.float32).flatten().to(device)
            
            batch = (X_support, y_support, X_query, y_query)

            start_time = time.time()
            for i in range(n_epochs):
                optimizer.zero_grad()
                # Compute task loss
                model.train()
                batch_logits = model(batch, train=True)
                batch_loss = model.compute_loss(batch_logits)
                batch_loss.backward()
                if clip_value is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                optimizer.step()

                train_loss = batch_loss.detach().item()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                # print('--- training epoch %d, lr %f, loss %.3f, time elapsed(min) %.2f'%(i, optimizer.param_groups[-1]['lr'], train_loss, (time.time()-start_time)/60))

            # Compute metric at test time
            model.eval()
            batch_logits = model(batch, train=False)

            with torch.no_grad():
                batch_preds = torch.sigmoid(batch_logits.mean).detach().cpu().numpy()

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


# For running the experiment
run_on_batches()
