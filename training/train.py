import sys
import os
import glob
from tqdm import tqdm
import torch
import torch.nn as nn
import gc
import os
import random
import json
import shutil
import numpy as np
from .utils import *
from torch.utils.data import Subset, DataLoader
from data.datasets import InterpolatedImageDataset
from datetime import datetime, timedelta;
import time

def Train(
    model,
    dataset,
    run_name, 
    batch_size, 
    grad_accum_steps,
    lr,
    loss_fn,
    save_steps,
    num_workers=8,
    seed = 111,
    warmup_steps = 25,
    validation = False, max_saves = 1, max_steps=None, save_dir="results", resume_from=None):

    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = f"{save_dir}/{run_name}"
    os.makedirs(save_dir, exist_ok=True)
    start_step = 0

    if validation:
        raise NotImplementedError("Eval steps not yet implemented!")
    else:
        train_set = dataset
    step_size = batch_size * grad_accum_steps
    num_steps = int(np.ceil(len(dataset) / step_size)) - 1

    if max_steps:
        num_steps = min(max_steps, num_steps)

    ### ------------------- Load model & optimizer & scheduler & loss ------------------- ###
    if resume_from:
        if len(os.listdir(save_dir)) > 0:
            if resume_from == "latest":
                saves = [f"{save_dir}/{f}" for f in os.listdir(save_dir) if "ckpt" in f]
                saves.sort(key=lambda x: int(x.split('_')[-1]))
                resume_from = saves[-1] if len(saves) > 1 else saves[0]
                print(f"Resuming training from the latest checkpoint, {resume_from}")

            print(f"Loading model from checkpoint: {resume_from}...")
            model.load_safetensors(resume_from)
            model.num_params()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda current_step: max(0.0, 1 - current_step / float(num_steps)),
                last_epoch=-1
            )
            if 'states.pt' in os.listdir(resume_from):
                checkpoint = load_optimizer_state(resume_from)
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_step = checkpoint['step']
                del(checkpoint)
            else:
                print("Optimizer states not found in the checkpoint directory. Will proceed as if training from zero")
            if 'logs.json' in os.listdir(resume_from):
                logs = load_logs(resume_from)
                for step in logs:
                    print(f"Step {step['step']}: train loss {step['train']:.6f}")
                start_step = logs[-1]['step'] + 1
            clear_memory()
        else:
            print("resume_from is set to latest, but no checkpoint found in the save directory. Starting a new run from scratch.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                last_epoch=-1,
                lr_lambda=lambda current_step: min(current_step / warmup_steps, 
                                                 max(0.0, 1 - current_step / float(num_steps))))
            logs = []
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            last_epoch=-1,
            lr_lambda=lambda current_step: min(current_step / warmup_steps, 
                                             max(0.0, 1 - current_step / float(num_steps))),   
        )
        logs = []


    criterion = get_loss_fn(loss_fn)
    pbar_global = tqdm(range(start_step, num_steps),  desc=f'Global step')

    os.makedirs(save_dir, exist_ok=True)
    model.to(torch.bfloat16)
    model.train()
    try:
        for step in pbar_global:
            clear_memory()
            train_subset = Subset(train_set, range(step * step_size, (step + 1) * step_size))
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,
                shuffle=False,
                prefetch_factor=2,
            )
            dloader_load_time = 0
            train_time = 0

            # -------------------- Gradient accumulation steps -------------------- #
            for grad_step in tqdm(range(grad_accum_steps), total=grad_accum_steps, leave=False, desc="Gradient accumulation"):
                dloader_grad_step_load_time = time.time()
                train_iter = iter(train_loader)
                inputs, preprocessed_inputs, targets = next(train_iter)
                dloader_load_time += (time.time() - dloader_grad_step_load_time)

                train_step_start_time = time.time()
                "Whether to use preprocessed inputs or original inputs"
                # placeholder if function, dont mind
                if 1 > 0:
                    inputs = preprocessed_inputs.to(device, dtype=torch.bfloat16)
                    outputs = model(inputs)
                else:
                    inputs = inputs.to(device).to(dtype=torch.bfloat16)
                    outputs = model(inputs)

                "Whether to use targets like labels or to treat the input as the target"
                if 1 < 0:
                    targets = targets.to(device, dtype=torch.bfloat16)
                else:
                    targets = inputs.to(device, dtype=torch.bfloat16)

                assert outputs.size() == targets.size(), f"Input size: {inputs.size()}, Output size: {outputs.size()}, Target size: {targets.size()}"

                loss = criterion(outputs, targets) / grad_accum_steps                
                loss.backward()
                train_time += (time.time() - train_step_start_time)
                
                del inputs
                del preprocessed_inputs
                del targets
                clear_memory()
                
            # After accumulating gradients for grad_accum_steps steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            clear_memory()
            finish = (datetime.utcnow() + timedelta(hours=7)).strftime('%d-%m-%Y %H:%M')
            
            losses = {
                "step": step, 
                "train": loss.item() * grad_accum_steps, # multiply back for reporting
                "time": finish
                
                }
            logs.append(losses)
            running_avg_train_loss, running_avg_window = avg_loss(logs, step)
            clear_memory()

            if validation:
                raise NotImplementedError("Validation logic not implemented yet!")
                with torch.no_grad():
                    model.eval()
                    eval_loss = 0
                    
                    for grad_step in range(grad_accum_steps):
                        eval_outputs = model(eval_split)
                        eval_loss += criterion(eval_outputs, eval_split)
                        
                    eval_loss = eval_loss / grad_accum_steps
                    losses["eval"] = eval_loss.item() * grad_accum_steps
                    model.train()
                    tqdm.write(f"Step {step}: train loss {losses['train']:.6f}; eval loss {losses['eval']:.6f}")
            else:
                tqdm.write(f"[STEP {step}] train loss {losses['train']:.6f}, avg {running_avg_window} steps: {running_avg_train_loss:.6f} || train time: {train_time:.3f}, dloader time: {dloader_load_time:.3f} || {finish}") 
   
            if step % save_steps == 0:
                save_ckpt(model, optimizer, scheduler, logs, save_dir, step, max_saves)
                
            if step <= warmup_steps or step % save_steps == 0 :
                model.inference('inference/degraded.jpg', f'inference/ckpt_{step}')

            clear_memory()
            pbar_global.update(1)

        clear_memory()
        model.eval()
        model.save(f"{save_dir}/last")

    except KeyboardInterrupt:
        if step > 0:
            print("\nInterrupted by user. Saving checkpoint (disregarding max_saves)...")
            save_ckpt(model, optimizer, scheduler, logs, save_dir, step-1, max_saves)
        sys.exit()
    print("Training finished.")


# from tqdm import tqdm
# import time

# for item in tqdm(range(25), desc='Outer') as outer:
#     if item % 5 == 0:
#         for inner in tqdm(range(5), desc='Inner', leave=False):
#             time.sleep(0.1)
#             outer.update(1)
