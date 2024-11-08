import argparse
import torch
import logging
import dataset.util.data_loader as MyDataLoader
from dataset.dataset_config import dataset_classes
from models.my_model import creat_model, train_model, evaluate_model
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
import getpass

def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print(f"Loading checkpoint '{path}'")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded. Resuming from epoch {epoch}.")
        return epoch
    else:
        print(f"No checkpoint found at '{path}'. Starting from scratch.")
        return 0

def create_log_dir(checkpoint='checkpoint', log_path='/data/LOG/train_log'):
    base_dir = os.path.join(log_path, getpass.getuser())
    exp_name = os.path.basename(os.path.abspath('.'))
    log_dir = os.path.join(base_dir, exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(log_dir)

    if not os.path.exists(checkpoint):
        cmd = "ln -s {} {}".format(log_dir, checkpoint)
        os.system(cmd)
        print(log_dir, "exists")


def setup_logging(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    logger = logging.getLogger('HARMODEL')
    logger.setLevel(logging.DEBUG)  # 设置最低级别的日志信息


    log_file_path = os.path.join(output_dir, 'training_log.log')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # 确保控制台处理器的日志级别设置为DEBUG

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def parse_args():
    parser = argparse.ArgumentParser(
        description='A HAR Models',
        prog="HAR_Models_Basic_Model"
    )

    parser.add_argument("--dataset", type=str, default="WISDM")
    parser.add_argument("--model_name", type=str, default="AdaMaskNet")

    # set Trainning parameters
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="learning rate")

    parser.add_argument("--resume", type=str, default=False, help="path to the checkpoint to resume training")
    parser.add_argument("--checkpoint_path", type=str, default="", help="path to the checkpoint to resume training")

    exp_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
    create_log_dir(checkpoint='checkpoint', log_path='/home')
    work_dir = 'checkpoint/p_{}'.format(exp_time)
    parser.add_argument("--output_dir", default=work_dir, help='path where to save, empty for no saving')

    return parser.parse_args()


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)

    # Remove unexpected keys
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
        model.load_state_dict(state_dict, strict=False)
    else:
        raise KeyError("model_state_dict not found in checkpoint")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError:
            print("Optimizer state_dict mismatch, skipping optimizer state loading.")

    start_epoch = checkpoint.get('epoch', 0)
    return start_epoch


def main(argUniMiB_SHARs):
    logger = setup_logging(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Using device: {device}')

    # read_dataset
    train_loader, test_loader, input_shape = MyDataLoader.load_dataset(args.dataset, args.batch_size)
    if train_loader is not None and test_loader is not None:
        model = creat_model(dataset = args.dataset,
                            num_classes= dataset_classes[args.dataset],
                            model_name=args.model_name,
                            input_shape=input_shape)
        model = model.to(device)

        if model is None:
            logger.error("Failed to create the model. Please check the model configuration.")
            return
        else:
            logger.info(model)


        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        start_epoch = 0
        if args.resume:
            print("Resuming from checkpoint...")
            if os.path.isfile(args.checkpoint_path):
                # start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)
                start_epoch = load_checkpoint(args.checkpoint_path, model, optimizer)
            else:
                logger.error(f"No checkpoint found at '{args.checkpoint_path}'. Starting from scratch.")

        writer = SummaryWriter(log_dir=args.output_dir)

        train_model(args.dataset,
                    train_loader,
                    test_loader,
                    args.model_name,
                    model,
                    criterion,
                    optimizer,
                    args.epochs,
                    learning_rate = args.learning_rate,
                    output_dir = args.output_dir,
                    device = device,
                    start_epoch = start_epoch,
                    writer = writer)

        evaluate_model(args.dataset,
                       test_loader,
                       args.model_name, model, device, writer, args.epochs)




if __name__ == '__main__':
    args = parse_args()
    main(args)
