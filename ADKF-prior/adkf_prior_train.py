import argparse
import os

import torch
import numpy as np
from adkf_prior_utils import ADKFPriorMetaModelTrainerConfig, ADKFPriorMetaModelTrainer



def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train an Adaptive DKT model on reactions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num_support", type=int, default=512, help="Number of samples in the training support set")
   
    parser.add_argument("--num_query", type=int, default=64, help="Number of samples in the training query set")

    parser.add_argument("--tasks_per_batch", type=int, default=5, help="Number of tasks to accumulate gradients for.")

    parser.add_argument("--num_inner_iters", type=int, default=40, help="Number of training steps in inner loop")

    parser.add_argument("--num_train_steps", type=int, default=10, help="Number of training steps.")

    parser.add_argument("--validate_every", type=int, default=5, help="Number of training steps between model validations.")
    
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    
    parser.add_argument("--clip_value", type=float, default=1.0, help="Gradient norm clipping value")

    parser.add_argument("--use-ard", action="store_true", help="Use a different lengthscale for each input dimension to the GP.")

    parser.add_argument("--gp-kernel", type=str, default="matern", help="The GP kernel.")
    
    parser.add_argument("--use-lengthscale-prior", action="store_true", help="Put a logNormal prior over the lengthscale(s).")
    
    parser.add_argument("--ignore-grad-correction", action="store_true", help="Ignore the second order term in the hypergradient. Default: False.")
    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> ADKFPriorMetaModelTrainerConfig:
    return ADKFPriorMetaModelTrainerConfig(
        num_support=args.num_support,
        num_query=args.num_query,
        tasks_per_batch=args.tasks_per_batch,
        validate_every_num_steps=args.validate_every,
        num_inner_iters=args.num_inner_iters,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
        use_ard=args.use_ard,
        gp_kernel=args.gp_kernel,
        use_lengthscale_prior=args.use_lengthscale_prior,
        ignore_grad_correction=args.ignore_grad_correction,
    )

def main():
    args = parse_command_line()
    config = make_trainer_config(args)

    model_path = '/homes/ss2971/Documents/AHO'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = ADKFPriorMetaModelTrainer(config=config).to(device)

    model_trainer.train_loop(model_path, device)


if __name__ == "__main__":
    main()
    