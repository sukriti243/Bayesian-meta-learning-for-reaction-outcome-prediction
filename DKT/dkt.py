# adapted from https://github.com/Wenlin-Chen/ADKF-IFT/blob/main/fs_mol/models/dkt.py

import torch
import numpy as np
import torch.nn as nn
from copy import deepcopy
from typing import List, Tuple
from typing_extensions import Literal

import gpytorch
from gpytorch.distributions import MultivariateNormal
from botorch.optim.fit import fit_gpytorch_scipy

from model import ExactGPLayer


class DKTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.test_time_adaptation = False

        # Create MLP if needed:
        self.fc_out_dim = 512
        fc_in_dim = 1543

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(1024, self.fc_out_dim)
        )

        kernel_type = self.config.gp_kernel
        if self.config.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None
        self.__create_tail_GP(kernel_type=kernel_type, ard_num_dims=ard_num_dims, use_lengthscale_prior=self.config.use_lengthscale_prior)

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

    def forward(self, batch, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        support_features, support_target, query_features, query_target = batch
       
        support_features_flat = self.fc(support_features)
        query_features_flat = self.fc(query_features)

        if self.training:
            combined_features_flat = torch.cat([support_features_flat, query_features_flat], dim=0)
            combined_labels_converted = torch.cat([support_target, query_target])

            self.gp_model.set_train_data(inputs=combined_features_flat, targets=combined_labels_converted, strict=False)
            logits = self.gp_model(combined_features_flat)
        else:
            self.gp_model.train()
            if self.test_time_adaptation:
                self.load_gp_params()

            self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_target, strict=False)
            
            if self.test_time_adaptation:
                self.gp_likelihood.train()
                fit_gpytorch_scipy(self.mll)
            
            self.gp_model.eval()
            self.gp_likelihood.eval()
            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits

    def compute_loss(self, logits: MultivariateNormal) -> torch.Tensor:
        assert self.training == True
        return -self.mll(logits, self.gp_model.train_targets)