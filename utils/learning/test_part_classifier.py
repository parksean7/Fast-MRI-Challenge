import torch
from utils.data.load_data import create_data_loaders
from utils.model.moe_router import anatomyClassifier_CNN, anatomyClassifier_Intensity, anatomyClassifier_Shape 
from utils import fastmri
from utils.common.utils import center_crop

def test(args, model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            # Data Loading
            _, kspace, _, _, fnames, slices, anatomy_label, _ = data
            kspace = kspace.cuda(non_blocking=True)
            anatomy_label = anatomy_label.cuda(non_blocking=True)
            true_label = anatomy_label[0].item()
            
            # Inference
            if (args.classifier == 'intensity'):  
                predicted, intensity = model(kspace, debug_mode=True)
                if predicted == true_label:
                    correct += 1
                    # print(f"  [Correct] {fnames[0]} [{slices[0]}] -> {intensity}")
                else:
                    # Print incorrect predictions with probability info
                    print(f"  [Wrong] {fnames[0]} [{slices[0]}] -> {intensity}")
            else:
                predicted = model(kspace)
                if predicted == true_label:
                    correct += 1
                else:
                    # Print incorrect predictions with probability info
                    print(f"  [Wrong] {fnames[0]} [{slices[0]}]")
            total += 1
    
    accuracy = 100 * correct / total
    print(f"Accuracy = [{correct} / {total}] = {accuracy:.2f}%")

def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    print(f"Loading anatomyClassifier ...")

    if (args.classifier == 'cnn'):  
        print("  Version: anatomyClassifier_CNN")
        model = anatomyClassifier_CNN()
        model.to(device=device)
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        print(f"  Loading checkpoint from epoch {checkpoint['epoch']}")
        model.load_state_dict(checkpoint['model'])

    elif (args.classifier == 'shape'):  
        print("  Version: anatomyClassifier_Shape")
        model = anatomyClassifier_Shape()
        model.to(device=device)
    elif (args.classifier == 'intensity'):  
        print("  Version: anatomyClassifier_Intensity")
        model = anatomyClassifier_Intensity()
        model.to(device=device)
    else:
        raise TypeError(f"anatomyClassifier Type Not Specified")
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    test(args, model, forward_loader)