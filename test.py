import argparse
import torch
import logging
import dataset.util.data_loader as MyDataLoader
from dataset.dataset_config import dataset_classes
from models.my_model import creat_model, evaluate_model
import os
from torch.utils.tensorboard import SummaryWriter

def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('HARMODEL_TEST')
    logger.handlers = []  # Clear existing handlers
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to root logger

    log_file_path = os.path.join(output_dir, 'test_log.log')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='Testing HAR Models',
        prog="HAR_Models_Test"
    )

    parser.add_argument("--dataset", type=str, default="WISDM", help="Dataset to use for testing")
    parser.add_argument("--model_name", type=str, default="AdaMaskNet", help="Name of the model to test")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for testing")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_dir/checkpoint_epoch_200.pth", help="Path to the checkpoint to load for testing")

    parser.add_argument("--output_dir", default='test_output', help='Directory to save test results')

    return parser.parse_args()

def load_checkpoint(checkpoint_path, model):
    if os.path.isfile(checkpoint_path):
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully.")
    else:
        raise FileNotFoundError(f"No checkpoint found at '{checkpoint_path}'.")

def main(args):
    logger = setup_logging(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')
    writer = SummaryWriter(log_dir=args.output_dir)

    # Load test dataset
    _, test_loader, input_shape = MyDataLoader.load_dataset(args.dataset, args.batch_size)
    if test_loader is None:
        logger.error("Test loader is None. Please check the dataset preparation.")
        return

    # Create model
    model = creat_model(
        dataset=args.dataset,
        num_classes=dataset_classes[args.dataset],
        model_name=args.model_name,
        input_shape=input_shape
    )
    if model is None:
        logger.error("Failed to create the model. Please check the model configuration.")
        return
    else:
        logger.info(f"Model created: {args.model_name}")

    model = model.to(device)

    # Load model weights
    load_checkpoint(args.checkpoint_path, model)

    # Evaluate model
    evaluate_model(
        dataset=args.dataset,
        test_loader=test_loader,
        model_name=args.model_name,
        model=model,
        device=device,
        writer=writer
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)
