from utils.model.moe_varnet import MoE_SAGVarNet
import os
import torch
import torch.nn as nn

def combine(args):
    """
    Combine anatomyRouter and two experts (brain, knee) into one model
    """
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())
    print(f'GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB')


    print("Initializing MoE_SAGVarNet...")
    
    model = MoE_SAGVarNet(
        brain_num_cascades=args.brain_num_cascades,
        brain_sens_chans=args.brain_sens_chans,
        brain_sens_pools=args.brain_sens_pools,
        brain_chans=args.brain_chans,
        brain_pools=args.brain_pools,
        brain_n_buffer=args.brain_n_buffer,
        brain_use_checkpoint=False,
        knee_num_cascades=args.knee_num_cascades,
        knee_sens_chans=args.knee_sens_chans,
        knee_sens_pools=args.knee_sens_pools,
        knee_chans=args.knee_chans,
        knee_pools=args.knee_pools,
        knee_n_buffer=args.knee_n_buffer,
        knee_use_checkpoint=False
    )
    model.to(device=device)

    # Load CNN
    if hasattr(args, 'classifier_cnn_path') and args.classifier_cnn_path:
        if os.path.exists(args.classifier_cnn_path):
            print(f"Loading Classifier from checkpoint: {args.classifier_cnn_path}")
            checkpoint = torch.load(args.classifier_cnn_path, map_location=device, weights_only=False)
            model.anatomy_router.classifier_cnn.load_state_dict(checkpoint['model'])

    # Load CNN
    if hasattr(args, 'brain_expert_path') and args.brain_expert_path:
        if os.path.exists(args.brain_expert_path):
            print(f"Loading Brain Expert from checkpoint: {args.brain_expert_path}")
            checkpoint = torch.load(args.brain_expert_path, map_location=device, weights_only=False)
            model.brain_expert.load_state_dict(checkpoint['model'])

    # Load CNN
    if hasattr(args, 'knee_expert_path') and args.knee_expert_path:
        if os.path.exists(args.knee_expert_path):
            print(f"Loading Knee Expert from checkpoint: {args.knee_expert_path}")
            checkpoint = torch.load(args.knee_expert_path, map_location=device, weights_only=False)
            model.knee_expert.load_state_dict(checkpoint['model'])

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Save 
    checkpoint = {
        'args': args,
        'model': model.state_dict(),
        'exp_dir': args.exp_dir,
        'model_type': 'MoE_SAGVarNet',
    }
    torch.save(checkpoint, f=args.exp_dir / 'best_model.pt')
    print(f"MoE_SAGVarNet Combined & Saved: {args.exp_dir / 'best_model.pt'}")