import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.moe_varnet import MoE_SAGVarNet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, _, _) in data_loader:
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )

    return reconstructions, None

def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # Initialize MoE_SAGVarNet with parameters from checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    print(f"Loading MoE_SAGVarNet with:")
    print(f"  Brain Sensitivity Channels: {args.brain_sens_chans}")
    print(f"  Brain Sensitivity Pools: {args.brain_sens_pools}")
    print(f"  Brain Channels: {args.brain_chans}")
    print(f"  Brain Pools: {args.brain_pools}")
    print(f"  Brain Cascades: {args.brain_num_cascades}")
    print(f"  Brain Number of Buffers: {args.brain_n_buffer}")
    print(f"  Knee Sensitivity Channels: {args.knee_sens_chans}")
    print(f"  Knee Sensitivity Pools: {args.knee_sens_pools}")
    print(f"  Knee Channels: {args.knee_chans}")
    print(f"  Knee Pools: {args.knee_pools}")
    print(f"  Knee Cascades: {args.knee_num_cascades}")
    print(f"  Knee Number of Buffers: {args.knee_n_buffer}")
    
    model = MoE_SAGVarNet(
        brain_sens_chans=args.brain_sens_chans,
        brain_sens_pools=args.brain_sens_pools,
        brain_num_cascades=args.brain_num_cascades,
        brain_chans=args.brain_chans,
        brain_pools=args.brain_pools,
        brain_n_buffer=args.brain_n_buffer,
        brain_use_checkpoint=False,
        knee_sens_chans=args.knee_sens_chans,
        knee_sens_pools=args.knee_sens_pools,
        knee_num_cascades=args.knee_num_cascades,
        knee_chans=args.knee_chans,
        knee_pools=args.knee_pools,
        knee_n_buffer=args.knee_n_buffer,
        knee_use_checkpoint=False,
    )
    model.to(device=device)
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)