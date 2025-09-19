import numpy as np
import torch

from collections import defaultdict
from utils.common.utils import save_reconstructions
from utils.data.load_data import create_data_loaders
from utils.model.moe_varnet import SAGVarNet

def test(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)

    with torch.no_grad():
        for (mask, kspace, _, _, fnames, slices, anatomy_label, _) in data_loader:
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

    # Initialize SAG VarNet with parameters from checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    
    print(f"Loading SAG VarNet with:")
    print(f"  Sensitivity Channels: {args.sens_chans}")
    print(f"  Sensitivity Pools: {args.sens_pools}")
    print(f"  Channels: {args.chans}")
    print(f"  Pools: {args.pools}")
    print(f"  Cascades: {args.num_cascades}")
    print(f"  Number of Buffers: {args.n_buffer}")
    
    model = SAGVarNet(
        num_cascades=args.num_cascades,
        sens_chans=args.sens_chans,
        sens_pools=args.sens_pools,
        chans=args.chans,
        pools=args.pools,
        n_buffer=args.n_buffer,
        use_checkpoint=False,
    )
    model.to(device=device)
    
    print(f"Loading checkpoint from epoch {checkpoint['epoch']} with best val loss: {checkpoint['best_val_loss']}")
    model.load_state_dict(checkpoint['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    reconstructions, inputs = test(args, model, forward_loader)
    save_reconstructions(reconstructions, args.forward_dir, inputs=inputs)