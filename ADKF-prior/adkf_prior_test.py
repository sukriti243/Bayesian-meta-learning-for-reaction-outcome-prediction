import argparse
import os

import torch
import numpy as np
from adkf_prior_utils import ADKFPriorMetaModelTrainer, test_ADKFPrior_model


def main():
    
    out_dir = '/homes/ss2971/Documents/AHO'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_weights_file = '/homes/ss2971/Documents/AHO/fully_trained_adkfprior.pt'

    model = ADKFPriorMetaModelTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )

    model.to(device)
    test_ADKFPrior_model(model, device)


if __name__ == "__main__":
    main()
    