import torch
from utils.data.load_data import create_data_loaders
from utils.model.moe_router import anatomyRouter

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

            # Inference
            predicted = model(kspace)
            
            if predicted == anatomy_label[0].item():
                correct += 1
            else:
                print(f"Wrong! {fnames[0]} [{slices[0]}]")

            total += 1
    
    accuracy = 100 * correct / total
    print(f"Accuracy = [{correct} / {total}] = {accuracy:.2f}%")

def forward(args):

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())

    # Initialize anatomyClassifier with parameters from checkpoint
    checkpoint_cnn = torch.load(args.model_path_cnn, map_location='cpu', weights_only=False)
    
    model = anatomyRouter()
    model.to(device=device)
    
    print(f"Loading checkpoint from {args.model_path_cnn}")

    model.classifier_cnn.load_state_dict(checkpoint_cnn['model'])
    
    forward_loader = create_data_loaders(data_path = args.data_path, args = args, isforward = True)
    test(args, model, forward_loader)