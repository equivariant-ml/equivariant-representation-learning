import argparse
import torch
from utils.nn_utils import entropy_loss, infoNCE
import torch.nn.functional as F


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help="Set seed for training")

    # Training details
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--save-interval', default=1, type=int, help="Epoch to save model")

    # Dataset
    parser.add_argument('--dataset', default='sprites', type=str, help="Dataset")
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size")


    parser.add_argument('--extra-dim', default=8, type=int, help="Number of classes")
    parser.add_argument('--model-name', required=True, type=str, help="Name of model")
    parser.add_argument('--decoder', action='store_true', default=False, help="Do you need a decoder?")
    parser.add_argument('--model', default='resnet_lie', type=str, help="Model to use")

    # Rotation specific arguments
    parser.add_argument('--method', type=str, default=None, help="What loss to use for rotations")

    parser.add_argument('--checkpoints-dir', default='checkpoints', type=str)

    # Optimization
    parser.add_argument('--lr-scheduler', action='store_true', default=False, help="Use a lr scheduler")
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--data-dir', default='data', type=str)

    parser.add_argument('--novel-class', type=bool, default=False)
    parser.add_argument('--novel-pose', type=bool, default=False)
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--use-comet', default=False, action='store_true')

    parser.add_argument('--regularization', type=str, default='None', help="Hinge or Info-NCE")
    parser.add_argument('--tau', type=float, default=0.8, help="Temperature on Info-NCE")

    return parser
