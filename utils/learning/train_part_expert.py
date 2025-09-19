import shutil
import numpy as np
import torch
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders, create_combined_data_loader, create_val_args
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, SSIM_L1_Loss, EdgeWeightedSSIMLoss
from utils.model.moe_varnet import SAGVarNet
from utils.data.mask_augment import calculate_mask_aug_schedule_prob

import os
import gc


def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    optimizer.zero_grad()

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, fname, slice, label_anatomy, grappa = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        # Forward pass 
        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)

        # Backward
        loss.backward()

        # Gradient Accumulation
        accumulation_steps = args.accumulation_steps if hasattr(args, 'accumulation_steps') else 2
        if (iter + 1) % accumulation_steps == 0 or (iter + 1) == len_loader:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping for stability
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()

        # Memory cleanup for large models
        if iter % 10 == 0:
            torch.cuda.empty_cache()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    total_loss = total_loss / len_loader

    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    total_slice_count = 0
    prediction_hit = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            mask, kspace, target, _, fnames, slices, label_anatomy, grappa = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            
            # Forward pass through VarNet SAG
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()
            
            # Memory cleanup during validation
            if iter % 5 == 0:
                torch.cuda.empty_cache()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, save_type="regular"):
    """
    Save model checkpoint for SAG VarNet.
    
    Args:
        args: Command line arguments
        exp_dir: Experiment directory
        epoch: Current epoch
        model: SAG VarNet model to save
        optimizer: Optimizer state
        best_val_loss: Best validation loss so far
        is_new_best: Whether this is the best model
        save_type: Type of save ("regular", "interval", "best")
    """
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir': exp_dir,
        'model_type': 'MoEVarNet',
    }
    
    # Always save most recent model
    torch.save(checkpoint, f=exp_dir / 'model.pt')
    
    # Save best model copy
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        print(f"New best model saved with loss: {best_val_loss:.6f}")
    
    # Save interval checkpoints for no-validation mode
    if save_type == "interval":
        interval_file = exp_dir / f'model_epoch_{epoch}.pt'
        torch.save(checkpoint, interval_file)
        print(f"Saved interval checkpoint: {interval_file}")


def train_epoch_no_val(args, epoch, model, data_loader, optimizer, loss_type):
    """Training epoch for no-validation mode with memory optimizations."""
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.

    for iter, data in enumerate(data_loader):
        mask, kspace, target, maximum, fname, slice, label_anatomy, grappa = data
        mask = mask.cuda(non_blocking=True)
        kspace = kspace.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        # Forward pass through SAG VarNet
        output = model(kspace, mask)
        loss = loss_type(output, target, maximum)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()

        # Regular memory cleanup
        if iter % 10 == 0:
            torch.cuda.empty_cache()

        if iter % args.report_interval == 0:
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'GPU Mem = {torch.cuda.memory_allocated()/1024/1024:.0f}MB '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
   
    return total_loss, time.perf_counter() - start_epoch

        
def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    print(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB')

    # Initialize MoE VarNet
    print("Initializing MoE SAGVarNet...")
    model = SAGVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        n_buffer=args.n_buffer,
        use_checkpoint=args.use_checkpoint,
    )
    model.to(device=device)

    # Load Whole Model if resuming
    if hasattr(args, 'resume') and args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    if args.loss_type == 'SSIM_L1':
        loss_type = SSIM_L1_Loss(
            ssim_weight=getattr(args, 'ssim_weights', 0.8), 
            l1_weight=getattr(args, 'l1_weight', 0.2)
        ).to(device=device)
    elif args.loss_type == 'EW_SSIM':
        loss_type = EdgeWeightedSSIMLoss(
            edge_method=getattr(args, 'ew_ssim_edge_method', 'hybrid'),
            edge_sigma=getattr(args, 'ew_ssim_edge_sigma', 0.8),
            edge_threshold=getattr(args, 'ew_ssim_edge_threshold', 0.1),
            use_pred_edges=getattr(args, 'ew_ssim_use_pred_edges', False),
            blend_ratio=getattr(args, 'ew_ssim_blend_ratio', 0.8)
        ).to(device=device)
    else:
        loss_type = SSIMLoss().to(device=device)
    
    # Use smaller learning rate - only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizer will update {len(trainable_params)} parameter groups")

    # Use smaller learning rate
    if args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=args.lr
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else 1e-4
        )
    
    # Learning rate scheduler
    if args.scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=args.scheduler_step if hasattr(args, 'scheduler_step') else 2,
            gamma=args.scheduler_gamma if hasattr(args, 'scheduler_gamma') else 0.3
        )
    elif args.scheduler_type == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.num_epochs,
            eta_min=args.eta_min,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode='min',
            factor=args.scheduler_factor if hasattr(args, 'scheduler_factor') else 0.3,
            patience=args.scheduler_patience if hasattr(args, 'scheduler_patience') else 5,
        )

    best_val_loss = 1.
    start_epoch = 0
    
    if args.no_validation:
        # No-validation mode: use combined train+val data for training
        print("="*60)
        print("NO-VALIDATION MODE: Using train+val datasets for training")
        print("="*60)
        
        train_loader = create_combined_data_loader(
            args.data_path_train, args.data_path_val, args, shuffle=True
        )
        print(f"Combined dataset size: {len(train_loader.dataset)} slices")
        
        # Track training loss as "best" metric
        best_train_loss = float('inf')
        train_loss_log = np.empty((0, 2))
        
        for epoch in range(start_epoch, args.num_epochs):
            print(f'Epoch #{epoch:2d} ............... {args.net_name} (No Validation) ...............')
            
            # Update current epoch for MRAugment scheduling
            args._current_epoch = epoch
            
            # Update mask augmentation scheduling probability
            if hasattr(train_loader.dataset.transform, 'mask_augmentor') and train_loader.dataset.transform.mask_augmentor is not None:
                mask_schedule_prob = calculate_mask_aug_schedule_prob(epoch, args)
                train_loader.dataset.transform.mask_augmentor.update_schedule_prob(mask_schedule_prob)
                print(f'Mask augmentation schedule probability: {mask_schedule_prob:.3f}')
            
            train_loss, train_time = train_epoch_no_val(args, epoch, model, train_loader, optimizer, loss_type)
            
            # Update learning rate scheduler
            scheduler.step()

            # Save training loss log
            train_loss_log = np.append(train_loss_log, np.array([[epoch, train_loss]]), axis=0)
            file_path = os.path.join(args.val_loss_dir, "train_loss_log")
            np.save(file_path, train_loss_log)
            
            train_loss_tensor = torch.tensor(train_loss).cuda(non_blocking=True)
            
            # Track best training loss
            is_new_best = train_loss < best_train_loss
            best_train_loss = min(best_train_loss, train_loss)
            
            # Save model at intervals (every epoch)
            save_interval = epoch >= 0  # Save at all epochs
            if save_interval or is_new_best or epoch == args.num_epochs - 1:
                save_type = "interval" if save_interval else "regular"
                save_model(args, args.exp_dir, epoch, model, optimizer, best_train_loss, is_new_best, save_type)
            
            print(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss_tensor:.4g} '
                f'BestTrainLoss = {best_train_loss:.4g} TrainTime = {train_time:.4f}s '
                f'LR = {optimizer.param_groups[0]["lr"]:.2e}',
            )

            if is_new_best:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord (Training Loss)@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    else:
        # Standard validation mode
        train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True)
        
        # Create validation-specific arguments and data loader
        val_args = create_val_args(args)
        val_loader = create_data_loaders(data_path = args.data_path_val, args = val_args)
        
        val_loss_log = np.empty((0, 2))
        for epoch in range(start_epoch, args.num_epochs):
            print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
            
            # Update current epoch for MRAugment scheduling
            args._current_epoch = epoch
            
            # Update mask augmentation scheduling probability for training
            if hasattr(train_loader.dataset.transform, 'mask_augmentor') and train_loader.dataset.transform.mask_augmentor is not None:
                mask_schedule_prob = calculate_mask_aug_schedule_prob(epoch, args)
                train_loader.dataset.transform.mask_augmentor.update_schedule_prob(mask_schedule_prob)
                print(f'Training mask augmentation schedule probability: {mask_schedule_prob:.3f}')
            
            # Update mask augmentation scheduling probability for validation
            if hasattr(val_loader.dataset.transform, 'mask_augmentor') and val_loader.dataset.transform.mask_augmentor is not None:
                val_mask_schedule_prob = calculate_mask_aug_schedule_prob(epoch, val_args)
                val_loader.dataset.transform.mask_augmentor.update_schedule_prob(val_mask_schedule_prob)
                print(f'Validation mask augmentation schedule probability: {val_mask_schedule_prob:.3f}')
            
            train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)
            val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
            
            # Update learning rate scheduler
            scheduler.step()
            
            val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
            file_path = os.path.join(args.val_loss_dir, "val_loss_log")
            np.save(file_path, val_loss_log)
            print(f"Loss file saved! {file_path}")

            train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
            val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
            num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

            val_loss = val_loss / num_subjects

            is_new_best = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            
            save_interval = epoch >= 0  # Save at all epochs
            if save_interval or is_new_best or epoch == args.num_epochs - 1:
                save_type = "interval" if save_interval else "regular"
                save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best, save_type)

            print(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
                f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s '
                f'LR = {optimizer.param_groups[0]["lr"]:.2e}',
            )

            if is_new_best:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                start = time.perf_counter()
                save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
                print(
                    f'ForwardTime = {time.perf_counter() - start:.4f}s',
                )

    print("Training completed!")
    torch.cuda.empty_cache()