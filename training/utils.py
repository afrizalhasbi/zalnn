from ast import literal_eval as literal
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import gc
import shutil

def set_seed(seed):    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    
    gc.collect()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except Exception as e:
            pass

def get_loss_fn(loss_fn):
    avail_losses = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'bce': nn.BCELoss, 
        'ce': nn.CrossEntropyLoss,
        'huber': nn.HuberLoss,
        'l1': nn.L1Loss,
        'smooth_l1': nn.SmoothL1Loss,
        'kl': nn.KLDivLoss,
        'nll': nn.NLLLoss,
        'poisson': nn.PoissonNLLLoss,
        'margin': nn.MarginRankingLoss,
        'hinge': nn.HingeEmbeddingLoss,
        'cosine': nn.CosineEmbeddingLoss,
        'triplet': nn.TripletMarginLoss
    }
    loss_fn = avail_losses[loss_fn.lower()]
    return loss_fn()

def save_ckpt(model, optimizer, scheduler, logs, save_dir, step, max_saves):
    model.eval()
    ckpt_dir = f"{save_dir}/ckpt_{step}"
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save(ckpt_dir)
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step
        }
    torch.save(checkpoint, f"{ckpt_dir}/states.pt")
    save_logs(ckpt_dir,logs)
    # with open(f"{ckpt_dir}/logs.json", "w") as f:
        # json.dump(logs, f, indent=2)
    saves = [f for f in os.listdir(save_dir) if "." not in f]
    saves.sort(key=lambda x: int(x.split('_')[-1]))
    if len(saves) > max_saves:
        shutil.rmtree(f"{save_dir}/{saves[0]}")

def load_optimizer_state(ckpt_path):
    checkpoint = torch.load(f"{ckpt_path}/states.pt")
    return checkpoint

def save_logs(savedir, logs):
    with open(f"{savedir}/logs.json", "w") as f:
        json.dump(logs, f, indent=2)

def load_logs(log_dir):
    with open(f"{log_dir}/logs.json", "r") as f:
        log = json.load(f)
    return log

def train_type(dataset):
    match value:
        case InterpolatedImageDataset() as dataset:
            return f"{x} is an integer"
        case ClassificationImage() as datasetx:
            return f"{dataset} is a string"
        case list() as x:
            return f"{dataset} is a list"
        case _:
            raise TypeError("Unknown dataset class, cannot determine training type")

# def avg_loss(log, current_step, window_size=10):
    # assert isinstance(log, list)
    # losses = [step['train'] for step in log]
    # start = max(0, current_step - window_size)
    # end = current_step + 1
    # window = losses[start : end]
    # avg = sum(losses) / len(losses)
    # return avg, len(window)

def avg_loss(log, current_step, window_size=10):
    assert isinstance(log, list)
    losses = [step['train'] for step in log]
    start = max(0, current_step - (window_size - 1))
    window = losses[start:current_step + 1]
    avg = sum(window) / len(window)
    return avg, len(window)
