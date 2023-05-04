import torch
import torch.nn as nn

from dataset import Cost2100DataLoader
from models import TransformCoding
from utils import init_device, logger, Tester
from utils.parser import args


def main():
    logger.info('=> PyTorch Version: {}'.format(torch.__version__))
    # Environment initialization
    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)

    # Define model
    model = TransformCoding(args.cr, None, args.bit_depth)
    model.to(device)
    logger.info(
        f'Transform coding with CR=1/{args.cr}, bit depth: {args.bit_depth}, coefficients used: {model.coeffs_used}'
    )

    # Create the data loader
    train_loader, val_loader, test_loader = Cost2100DataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=pin_memory,
        scenario=args.scenario)()

    # Define loss function
    criterion = nn.MSELoss().to(device)

    # Inference mode only
    Tester(model, device, criterion)(test_loader)


if __name__ == "__main__":
    main()
