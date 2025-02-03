# adapted from https://github.com/Wenlin-Chen/ADKF-IFT/blob/main/fs_mol/models/protonet.py

from dataclasses import dataclass
from typing import List, Tuple
from typing_extensions import Literal

import torch
import torch.nn as nn
import numpy as np


class PrototypicalNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Create MLP if needed:
        self.fc_out_dim = 512
        fc_in_dim = 1544

        self.fc = nn.Sequential(
            nn.Linear(fc_in_dim, 1024), 
            nn.PReLU(), nn.Dropout(0.1),
            nn.Linear(1024, self.fc_out_dim)
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, batch, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        
        support_features, support_target, query_features, query_target = batch
        
        # SUPPORT_FEATURES
        support_features_flat = self.fc(support_features)
       
        # QUERY FEATURES
        query_features_flat = self.fc(query_features)
        
        if self.config.distance_metric == "mahalanobis":
            class_means, class_precision_matrices = self.compute_class_means_and_precisions(
                support_features_flat, support_target
            )

            # grabbing the number of classes and query examples for easier use later
            number_of_classes = class_means.size(0)
            number_of_targets = query_features_flat.size(0)

            """
            Calculating the Mahalanobis distance between query examples and the class means
            including the class precision estimates in the calculations, reshaping the distances
            and multiplying by -1 to produce the sample logits
            """
            repeated_target = query_features_flat.repeat(1, number_of_classes).view(
                -1, class_means.size(1)
            )
            repeated_class_means = class_means.repeat(number_of_targets, 1)
            repeated_difference = repeated_class_means - repeated_target
            repeated_difference = repeated_difference.view(
                number_of_targets, number_of_classes, repeated_difference.size(1)
            ).permute(1, 0, 2)
            first_half = torch.matmul(repeated_difference, class_precision_matrices)
            logits = torch.mul(first_half, repeated_difference).sum(dim=2).transpose(1, 0) * -1
        else:  # euclidean
            logits = self._protonets_euclidean_classifier(
                support_features_flat,
                query_features_flat,
                support_target,
            )
        return logits

    def compute_class_means_and_precisions(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        means = []
        precisions = []
        task_covariance_estimate = self._estimate_cov(features)
        for c in torch.unique(labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(features, 0, self._extract_class_indices(labels, c))
            # mean pooling examples to form class means
            means.append(torch.mean(class_features, dim=0, keepdim=True).squeeze())
            lambda_k_tau = class_features.size(0) / (class_features.size(0) + 1)
            lambda_k_tau = min(lambda_k_tau, 0.1)
            precisions.append(
                torch.inverse(
                    (lambda_k_tau * self._estimate_cov(class_features))
                    + ((1 - lambda_k_tau) * task_covariance_estimate)
                    + 0.1
                    * torch.eye(class_features.size(1), class_features.size(1)).to(self.device)
                )
            )

        means = torch.stack(means)
        precisions = torch.stack(precisions)

        return means, precisions

    @staticmethod
    def _estimate_cov(
        examples: torch.Tensor, rowvar: bool = False, inplace: bool = False
    ) -> torch.Tensor:
        """
        SCM: Function based on the suggested implementation of Modar Tensai
        and his answer as noted in:
        https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/5

        Estimate a covariance matrix given data.

        Covariance indicates the level to which two variables vary together.
        If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
        then the covariance matrix element `C_{ij}` is the covariance of
        `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

        Args:
            examples: A 1-D or 2-D array containing multiple variables and observations.
                Each row of `m` represents a variable, and each column a single
                observation of all those variables.
            rowvar: If `rowvar` is True, then each row represents a
                variable, with observations in the columns. Otherwise, the
                relationship is transposed: each column represents a variable,
                while the rows contain observations.

        Returns:
            The covariance matrix of the variables.
        """
        if examples.dim() > 2:
            raise ValueError("m has more than 2 dimensions")
        if examples.dim() < 2:
            examples = examples.view(1, -1)
        if not rowvar and examples.size(0) != 1:
            examples = examples.t()
        factor = 1.0 / (examples.size(1) - 1)
        if inplace:
            examples -= torch.mean(examples, dim=1, keepdim=True)
        else:
            examples = examples - torch.mean(examples, dim=1, keepdim=True)
        examples_t = examples.t()
        return factor * examples.matmul(examples_t).squeeze()

    @staticmethod
    def _extract_class_indices(labels: torch.Tensor, which_class: torch.Tensor) -> torch.Tensor:
        class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
        return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector

    @staticmethod
    def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits, labels.long())

    def _protonets_euclidean_classifier(
        self,
        support_features: torch.Tensor,
        query_features: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        class_prototypes = self._compute_class_prototypes(support_features, support_labels)
        logits = self._euclidean_distances(query_features, class_prototypes)
        return logits

    def _compute_class_prototypes(
        self, support_features: torch.Tensor, support_labels: torch.Tensor
    ) -> torch.Tensor:
        means = []
        for c in torch.unique(support_labels):
            # filter out feature vectors which have class c
            class_features = torch.index_select(
                support_features, 0, self._extract_class_indices(support_labels, c)
            )
            means.append(torch.mean(class_features, dim=0))
        return torch.stack(means)

    def _euclidean_distances(
        self, query_features: torch.Tensor, class_prototypes: torch.Tensor
    ) -> torch.Tensor:
        num_query_features = query_features.shape[0]
        num_prototypes = class_prototypes.shape[0]
        distances = (
            (
                query_features.unsqueeze(1).expand(num_query_features, num_prototypes, -1)
                - class_prototypes.unsqueeze(0).expand(num_query_features, num_prototypes, -1)
            )
            .pow(2)
            .sum(dim=2)
        )
        return -distances