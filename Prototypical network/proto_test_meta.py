import argparse
import sys, os
import torch
import numpy as np

from proto_meta_utils import PrototypicalNetworkTrainer, test_protonet_model


def main():
    
    out_dir = '/homes/ss2971/Documents/AHO'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model_weights_file = '/homes/ss2971/Documents/AHO/AHO_FP/best_validation_proto.pt'
    model_weights_file = '/homes/ss2971/Documents/AHO/AHO_FP/fully_trained_proto.pt'

    model = PrototypicalNetworkTrainer.build_from_model_file(
        model_weights_file,
        device=device,
    )

    model.to(device)
    test_protonet_model(model, device)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        import pdb

        _, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)