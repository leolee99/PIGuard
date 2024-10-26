import os
import torch
import random
import numpy as np
import logging
import torch.backends.cudnn as cudnn

def compute_accuracy(predictions, labels):
    correct = predictions == labels

    label_0_indices = labels == 0
    label_1_indices = labels == 1

    correct_0 = correct[label_0_indices].sum().item()
    correct_1 = correct[label_1_indices].sum().item()

    total_0 = label_0_indices.sum().item()
    total_1 = label_1_indices.sum().item()

    total_samples = labels.size(0)
    total_correct = correct.sum().item()

    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_1 = correct_1 / total_1 if total_1 > 0 else 0

    total_accuracy = total_correct / total_samples if total_samples > 0 else 0

    return accuracy_0, accuracy_1, total_accuracy

def set_seed(args):
    seed=args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return args.seed

def get_logger(filename=None):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(asctime)s - %(levelname)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)
    return logger