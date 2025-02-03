import torch
import torch.nn as nn
import numpy as np
import gpytorch

from model import ExactGPLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ADKFModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create MLP if needed:
        self.fc_out_dim = 512
        fc_in_dim = 1543

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(1024, self.fc_out_dim)
        )

        self.__create_tail_GP(kernel_type=self.config.gp_kernel)

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params

    def reinit_gp_params(self, gp_input, use_lengthscale_prior=True):

        self.__create_tail_GP(kernel_type=self.config.gp_kernel)

        if self.config.gp_kernel == 'matern' or self.config.gp_kernel == 'rbf' or self.config.gp_kernel == 'RBF':
            median_lengthscale_init = self.compute_median_lengthscale_init(gp_input)
            if use_lengthscale_prior:
                scale = 0.25
                loc = torch.log(median_lengthscale_init).item() + scale**2 # make sure that mode=median_lengthscale_init
                lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
                self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * median_lengthscale_init

    def __create_tail_GP(self, kernel_type):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        if self.config.use_ard:
            ard_num_dims = self.fc_out_dim
        else:
            ard_num_dims = None
        
        scale = 0.25
        loc = np.log(0.01) + scale**2 # make sure that mode=0.01
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
        
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        # self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior).to(self.device)
        self.gp_model = ExactGPLayer(train_x=dummy_train_x, train_y=dummy_train_y, likelihood=self.gp_likelihood, kernel=kernel_type, ard_num_dims=ard_num_dims).to(self.device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(self.device)

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input) ** 2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5 * torch.median(dist_squared[dist_squared>0.0]))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_batch, train_loss: bool, predictive_val_loss: bool=False, is_functional_call: bool=False):
        support_features, support_labels, query_features, query_labels = input_batch

        support_features_flat = self.fc(support_features)
        query_features_flat = self.fc(query_features)

        # compute train/val loss if the model is in the training mode
        if self.training:
            assert train_loss is not None
            if train_loss: # compute train loss (on the support set)
                if is_functional_call: # return loss directly
                    self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels, strict=False)
                    logits = self.gp_model(support_features_flat)
                    logits = -self.mll(logits, self.gp_model.train_targets)
                else:
                    self.reinit_gp_params(support_features_flat.detach(), self.config.use_lengthscale_prior)
                    self.gp_model.set_train_data(inputs=support_features_flat.detach(), targets=support_labels.detach(), strict=False)
                    logits = None
            else: # compute val loss (on the query set)
                assert is_functional_call == True
                if predictive_val_loss:
                    self.gp_model.eval()
                    self.gp_likelihood.eval()
                    with gpytorch.settings.detach_test_caches(False):
                        self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels, strict=False)
                        # return sum of the log predictive losses for all data points, which converges better than averaged loss
                        logits = -self.gp_likelihood(self.gp_model(query_features_flat)).log_prob(query_labels).sum() #/ self.predictive_targets.shape[0]
                    self.gp_model.train()
                    self.gp_likelihood.train()
                else:
                    self.gp_model.set_train_data(inputs=query_features_flat, targets=query_labels, strict=False)
                    logits = self.gp_model(query_features_flat)
                    logits = -self.mll(logits, self.gp_model.train_targets)

        # do GP posterior inference if the model is in the evaluation mode
        else:
            assert train_loss is None
            self.gp_model.set_train_data(inputs=support_features_flat, targets=support_labels, strict=False)

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_flat))

        return logits
    

