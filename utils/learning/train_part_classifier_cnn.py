import shutil
import numpy as np
import torch
import warnings
import torch.nn as nn
import time
from pathlib import Path
import copy

from collections import defaultdict
from utils.data.load_data import create_data_loaders, create_combined_data_loader, create_val_args
from utils.model.moe_router import anatomyClassifier_CNN
from utils.data.mask_augment import calculate_mask_aug_schedule_prob
from utils import fastmri
from utils.common.utils import center_crop
import os
import gc

def train_epoch(args, epoch, model, data_loader, optimizer, criterion):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    correct = 0
    total = 0

    for iter, data in enumerate(data_loader):
        # Data Loading
        _, kspace, _, _, _, _, anatomy_label, _ = data
        kspace = kspace.cuda(non_blocking=True)
        anatomy_label = anatomy_label.cuda(non_blocking=True)
    
        # zero grad before forward pass
        optimizer.zero_grad()

        # Forward pass - get logits for training
        logits = model(kspace, train_mode=True)
        
        loss = criterion(logits, anatomy_label)
        
        # Calculate accuracy
        _, predicted = torch.max(logits.data, 1)
        total += anatomy_label.size(0)
        correct += (predicted == anatomy_label).sum().item()
        
        # Backward
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        if iter % args.report_interval == 0:
            accuracy = 100 * correct / total
            print(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item():.4g} '
                f'Acc = {accuracy:.2f}% '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            start_iter = time.perf_counter()
    
    total_loss = total_loss / len_loader
    accuracy = 100 * correct / total
    return total_loss, accuracy, time.perf_counter() - start_epoch

def validate(args, model, data_loader, criterion):
    model.eval()
    total_loss = 0.
    correct = 0
    total = 0
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Data Loading
            _, kspace, _, _, _, _, anatomy_label, _ = data
            kspace = kspace.cuda(non_blocking=True)
            anatomy_label = anatomy_label.cuda(non_blocking=True)
    
            # Forward pass - get logits for validation
            logits = model(kspace, train_mode=True)
            
            loss = criterion(logits, anatomy_label)
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += anatomy_label.size(0)
            correct += (predicted == anatomy_label).sum().item()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, time.perf_counter() - start

def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, save_type="regular"):
    checkpoint = {
        'epoch': epoch,
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'exp_dir': exp_dir,
        'model_type': 'anatomyRouter',
    }
    
    # Always save most recent model
    torch.save(checkpoint, f=exp_dir / 'model.pt')
    
    # Save best model copy
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        print(f"New best model saved with loss: {best_val_loss:.6f}")
    
    # Save interval checkpoints
    if save_type == "interval":
        interval_file = exp_dir / f'model_epoch_{epoch}.pt'
        torch.save(checkpoint, interval_file)
        print(f"Saved interval checkpoint: {interval_file}")

def train(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # Initialize anotomyClassifier_CNN
    print("Initializing anotomyClassifier_CNN...")
    model = anatomyClassifier_CNN()
    model.to(device=device)

    # Load Model if resuming
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
    
    # Cross entropy loss for binary classification
    criterion = nn.CrossEntropyLoss().to(device=device)
    
    # Optimizer
    if args.optim_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
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
  
    # Training log
    train_log = []
    val_log = []

    best_val_loss = float('inf')
    best_val_acc = 0.0
    start_epoch = 0

    # Data Loader 
    if args.no_validation: # No Validation
        print("="*60)
        print("NO VALIDATION MODE: Using All Train + Val Dataset")
        print("="*60)
        train_loader = create_combined_data_loader(
            args.data_path_train, args.data_path_val, args, shuffle=True
        )
        print(f"Combined dataset size: {len(train_loader.dataset)} slices")
    else: # with Validation
        print("="*60)
        print("VALIDATION MODE: Using Only Train Val Dataset")
        print("="*60)
        train_loader = create_data_loaders(
            args.data_path_train, args, shuffle=True
        )
        val_args = create_val_args(args)
        val_loader = create_data_loaders(
            args.data_path_val, val_args, shuffle=False
        )
        print(f"Train dataset size: {len(train_loader.dataset)} slices")
        print(f"Validation dataset size: {len(val_loader.dataset)} slices")
        
    # Train Epochs
    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... Anatomy Router Training ...............')
        
        # Training
        train_loss, train_acc, train_time = train_epoch(args, epoch, model, train_loader, optimizer, criterion)
        train_log.append([epoch, train_loss, train_acc])
        
        # Validation
        if not args.no_validation:
            val_loss, val_acc, val_time = validate(args, model, val_loader, criterion)
            val_log.append([epoch, val_loss, val_acc])
            
            # Update learning rate scheduler with validation loss
            scheduler.step()
            
            # Check if this is the best model
            is_new_best = val_loss < best_val_loss
            if is_new_best:
                best_val_loss = val_loss
                best_val_acc = val_acc
            
            # Save model at intervals (every epoch)
            save_interval = epoch >= 0  # Save at all epochs
            save_type = "interval" if save_interval else "regular"
            save_model(args, args.exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, save_type)
            
            print(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
                f'TrainLoss = {train_loss:.4g} TrainAcc = {train_acc:.2f}% '
                f'ValLoss = {val_loss:.4g} ValAcc = {val_acc:.2f}% '
                f'TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s '
                f'LR = {optimizer.param_groups[0]["lr"]:.2e}',
            )
            
            if is_new_best:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        else:
            # No validation mode - use training loss
            scheduler.step()
            
            is_new_best = train_loss < best_val_loss
            if is_new_best:
                best_val_loss = train_loss
                best_val_acc = train_acc
            
            # Save model at intervals (every epoch)
            save_interval = epoch >= 0  # Save at all epochs
            save_type = "interval" if save_interval else "regular"
            save_model(args, args.exp_dir, epoch, model, optimizer, best_val_loss, is_new_best, save_type)
            
            print(
                f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
                f'TrainLoss = {train_loss:.4g} TrainAcc = {train_acc:.2f}% '
                f'TrainTime = {train_time:.4f}s '
                f'LR = {optimizer.param_groups[0]["lr"]:.2e}',
            )
            
            if is_new_best:
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord (Training)@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    
    # Save training logs
    if hasattr(args, 'exp_dir'):
        np.save(args.exp_dir / "train_log.npy", np.array(train_log))
        if val_log:
            np.save(args.exp_dir / "val_log.npy", np.array(val_log))
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    torch.cuda.empty_cache()
