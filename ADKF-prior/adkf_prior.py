import torch
import torch.nn as nn
import numpy as np
import gpytorch
import math
from torch.nn import Parameter

from model import ExactGPLayer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ADKFPriorMetaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create MLP if needed:
        self.fc_out_dim = 512
        fc_in_dim = 1543

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024), 
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, self.fc_out_dim)
        )

        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                if "weight" in name:
                    setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(nn.init.xavier_uniform_(torch.Tensor(*tuple(param.shape)))))
                    setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.01))))
                    
                elif "bias" in name:
                    setattr(self, name.replace(".", "_")+"_mu_nn", Parameter(torch.zeros(param.shape)))
                    setattr(self, name.replace(".", "_")+"_logsigma_nn", Parameter(torch.full(tuple(param.shape), math.log(0.01))))

                else:
                    raise ValueError("Unexpected parameter with name {}.".format(name))

        self.__create_tail_GP()

    def gp_params(self):
        gp_params = []
        for name, param in self.named_parameters():
            if name.startswith("gp_"):
                gp_params.append(param)
        return gp_params

    def feature_extractor_params(self):
        fe_params = []
        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                fe_params.append(param)
        return fe_params
    
    def prior_params(self):
        prior_params = []
        for name, param in self.named_parameters():
            if name.endswith("_nn"):
                prior_params.append(param)
        return prior_params

    def reinit_gp_params(self, use_lengthscale_prior=True):
        self.__create_tail_GP()

        if use_lengthscale_prior:
            scale = 0.25
            loc = 0.0
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
            self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_prior.mean

    def reinit_feature_extractor_params(self):  

        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                param.data = getattr(self, name.replace(".", "_")+"_mu_nn") + torch.exp(getattr(self, name.replace(".", "_")+"_logsigma_nn")) * torch.randn_like(getattr(self, name.replace(".", "_")+"_logsigma_nn")).to(device)
        
    def __create_tail_GP(self, use_lengthscale_prior=True):
        dummy_train_x = torch.ones(64, self.fc_out_dim)
        dummy_train_y = torch.ones(64)

        if self.config.use_ard:
            ard_num_dims = 10
        else:
            ard_num_dims = None

        scale = 0.25
        loc = np.log(0.01) + scale**2
        noise_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)

        # self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior) 
        self.gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.gp_model = ExactGPLayer(train_x=None, train_y=dummy_train_y, likelihood=self.gp_likelihood, kernel=self.config.gp_kernel, ard_num_dims=ard_num_dims).to(device)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model).to(device)

        if use_lengthscale_prior:
            scale = 0.25
            loc = 0.0
            lengthscale_prior = gpytorch.priors.LogNormalPrior(loc=loc, scale=scale)
            self.gp_model.covar_module.base_kernel.register_prior(
                    "lengthscale_prior", lengthscale_prior, lambda m: m.lengthscale, lambda m, v: m._set_lengthscale(v)
                )
            self.gp_model.covar_module.base_kernel.lengthscale = torch.ones_like(self.gp_model.covar_module.base_kernel.lengthscale) * lengthscale_prior.mean

    def compute_median_lengthscale_init(self, gp_input):
        dist_squared = torch.cdist(gp_input, gp_input)**2
        dist_squared = torch.triu(dist_squared, diagonal=1)
        return torch.sqrt(0.5*torch.median(dist_squared[dist_squared>0.0]))

    def log_prob(self, loc, logscale, value):
        # compute the variance
        var = (torch.exp(logscale) ** 2)
        return -((value - loc) ** 2) / (2 * var) - logscale - math.log(math.sqrt(2 * math.pi))
        
    def log_prior(self):
        logprob_prior = torch.tensor(0.0).to(device)
        
        for name, param in self.named_parameters():
            if not name.endswith("_nn") and not name.startswith("gp_"):
                logprob_prior += self.log_prob(getattr(self, name.replace(".", "_")+"_mu_nn"), getattr(self, name.replace(".", "_")+"_logsigma_nn"), param).sum()
    
        return logprob_prior

    def forward(self, input_batch, train_loss:bool, predictive_val_loss:bool=False, is_functional_call:bool=False):
        support_features, support_labels, query_features, query_labels = input_batch
        
        support_features_z = self.fc(support_features)
        query_features_z = self.fc(query_features)
        
        # compute train/val loss if model is in training mode
        if self.training:
            assert train_loss is not None
            if train_loss: #compute train loss on support set
                if is_functional_call: #return loss directly
                    self.gp_model.set_train_data(inputs=support_features_z, targets=support_labels, strict=False)
                    logits = self.gp_model(support_features_z)
                    logits = -self.mll(logits, self.gp_model.train_targets)  - self.log_prior() 
                else:
                    self.reinit_gp_params(use_lengthscale_prior=True)
                    self.gp_model.set_train_data(inputs=support_features_z.detach(), targets=support_labels.detach(), strict=False)
                    
                    logits = None
            
            else: # compute val loss on the query set
                if predictive_val_loss:
                    self.gp_model.eval()
                    self.gp_likelihood.eval()
                    with gpytorch.settings.detach_test_caches(False):
                        self.gp_model.set_train_data(inputs=support_features_z, targets=support_labels, strict=False)
                        # return sum of the log predictive losses for all data points, which converges better than averaged loss
                        logits = -self.gp_likelihood(self.gp_model(query_features_z)).log_prob(query_labels)

                    self.gp_model.train()
                    self.gp_likelihood.train()
                else:
                    self.gp_model.set_train_data(inputs=query_features_z, targets=query_labels, strict=False)
                    logits = self.gp_model(query_features_z)
                    logits = -self.mll(logits, self.gp_model.train_targets)

        else:
            assert train_loss is None
            self.gp_model.set_train_data(inputs=support_features_z, targets=support_labels, strict=False)

            with torch.no_grad():
                logits = self.gp_likelihood(self.gp_model(query_features_z))

        return logits