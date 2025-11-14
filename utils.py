import logging
import os
import torch
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # formatter = logging.Formatter(
    #     "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    # )
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level_dict[verbosity])
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    
    logger.addHandler(sh)

    return logger


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def graph_tv(node_features, edge_index, norm="l2sq", reduce="mean", unique=False):
    """
    Graph Total Variation for node embeddings.
    node_features: [N, F]
    edge_index:    [2, E]
    norm:  'l1' | 'l2' | 'l2sq' (default)
    reduce: 'mean' (default) | 'sum'
    unique: if True, use only one direction of undirected edges (row < col)
    """
    row, col = edge_index
    if unique:
        mask = row < col
        row, col = row[mask], col[mask]

    diff = node_features[row] - node_features[col]  # [E, F]
    if norm == "l1":
        per_edge = diff.abs().sum(dim=1)
    elif norm == "l2":
        per_edge = diff.norm(p=2, dim=1)
    else:  # 'l2sq'
        per_edge = (diff * diff).sum(dim=1)

    if reduce == "sum":
        return per_edge.sum() / max(1, per_edge.numel())
    return per_edge.mean()