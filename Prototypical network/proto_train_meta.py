import argparse
import sys, os
import torch
import numpy as np

from proto_meta_utils import PrototypicalNetworkTrainerConfig, PrototypicalNetworkTrainer


def parse_command_line():
    parser = argparse.ArgumentParser(
        description="Train a Prototypical model on AHO reaction dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--num_support", type=int, default=512, help="Number of samples in the training support set"
    )
    parser.add_argument(
        "--num_query", type=int, default=64, help="Number of samples in the training query set"
    )
    parser.add_argument(
        "--validate_every_num_steps", type=int, default=50, help="Number of training steps between model validations"
    )
    parser.add_argument(
        "--tasks_per_batch", type=int, default=5, help="Number of tasks for gradient accumulation"
    )
    parser.add_argument(
        "--num_train_steps", type=int, default=1000, help="Number of training steps."
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--clip_value", type=float, default=1.0, help="Gradient norm clipping value"
    )
    parser.add_argument(
        "--distance_metric", type=str, choices=["mahalanobis", "euclidean"], default="mahalanobis", help="Choice of distance to use."
    )
    args = parser.parse_args()
    return args


def make_trainer_config(args: argparse.Namespace) -> PrototypicalNetworkTrainerConfig:
    return PrototypicalNetworkTrainerConfig(
        num_support=args.num_support,
        num_query=args.num_query,
        validate_every_num_steps=args.validate_every_num_steps,
        num_train_steps=args.num_train_steps,
        learning_rate=args.lr,
        clip_value=args.clip_value,
        distance_metric=args.distance_metric,
    )


def main():

    args = parse_command_line()
    config = make_trainer_config(args)

    model_path = '/homes/ss2971/Documents/AHO/AHO_FP'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = PrototypicalNetworkTrainer(config=config).to(device)
    model_trainer.train_loop(model_path, device)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)