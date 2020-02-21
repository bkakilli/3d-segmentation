import argparse

import numpy as np

from scripts.generate_results import get_segmentation_metrics
from models.model_sseg import HGCN
from utils import misc, data_loader

import torch
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    parser.add_argument('--train', action='store_true', help='Trains the model if provided')
    parser.add_argument('--test', action='store_true', help='Evaluates the model if provided')
    parser.add_argument('--dataset', type=str, default='s3dis', choices=['s3dis'], help='Experiment dataset')
    parser.add_argument('--prefix', type=str, default='', help='Path prefix')
    parser.add_argument('--logdir', type=str, default='log', help='Name of the experiment')
    parser.add_argument('--model_path', type=str, help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=2, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of episode to train')
    parser.add_argument('--use_adam', action='store_true', help='Uses Adam optimizer if provided')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay rate')
    parser.add_argument('--decay_step', type=float, default=50, help='Learning rate decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, help='Embedding dimensions')
    parser.add_argument('--k', type=int, default=20, help='K of K-Neareset Neighbors')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--print_summary', type=bool,  default=True, help='Whether to print epoch summary')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA id. -1 for CPU')
    parser.add_argument('--no_augmentation', action='store_true', help='Disables training augmentation if provided')

    return parser.parse_args()


def main():
    # Temporary fix for:
    # RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
    torch.backends.cudnn.enabled = False

    args = get_arguments()

    # Seed RNG
    misc.seed(args.seed)

    model = HGCN(args)
    train_loader, valid_loader, test_loader = data_loader.get_loaders(args)

    if args.train:
        train(model, train_loader, valid_loader, args)

    if args.test:
        test(model, test_loader, args)


def run_one_epoch(model, tqdm_iterator, mode, loss_fcn=None, get_locals=False, optimizer=None, loss_update_interval=1000):
    """Definition of one epoch procedure.
    """
    if mode == "train":
        assert optimizer is not None and loss_fcn is not None
        model.train()
    else:
        model.eval()
        # Following is a hack for CUDA OF OF MEMORY issue when backward is not called.
        # Reverse is done at the end of the function
        param_grads = []
        for param in model.parameters():
            param_grads += [param.requires_grad]
            param.requires_grad = False

    summary = {"losses": [], "logits": [], "labels": []}

    device = next(model.parameters()).device

    for i, (X_cpu, y_cpu) in enumerate(tqdm_iterator):
        X, y = X_cpu.to(device), y_cpu.to(device)

        if mode == "train":
            optimizer.zero_grad()

        logits = model(X)
        if loss_fcn is not None:
            loss = loss_fcn(logits.view(-1, logits.shape[-1]), y.view(-1))
            summary["losses"] += [loss.item()]

        if mode == "train":
            # Apply back-prop
            loss.backward()
            optimizer.step()

            # Display
            if loss_fcn is not None and i%loss_update_interval == 0:
                tqdm_iterator.set_description("Loss: %.3f" % (np.mean(summary["losses"])))

        if get_locals:
            summary["logits"] += [logits.cpu().detach().numpy()]
            summary["labels"] += [y_cpu.numpy()]

    # Following is the reverse of the hack defined above
    if mode != "train":
        for param, value in zip(model.parameters(), param_grads):
            param.requires_grad = value

    if get_locals:
        summary["logits"] = np.concatenate(summary["logits"], axis=0)
        summary["labels"] = np.concatenate(summary["labels"], axis=0)

    return summary

def test(model, test_loader, args):

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and loss
    model = model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()

    # Get current state
    state = torch.load(args.model_path)
    model.load_state_dict(state["model_state_dict"])
    print("Loaded pre-trained model from %s"%args.model_path)

    def test_one_epoch():
        iterations = tqdm(test_loader, ncols=100, unit='batch', desc="Testing")
        ep_sum = run_one_epoch(model, iterations, "test", ce_loss, get_locals=True, loss_update_interval=-1)

        preds = ep_sum["logits"].argmax(axis=-1)
        metrics = get_segmentation_metrics(ep_sum["labels"], preds)

        summary = {"Loss/test": str(np.mean(ep_sum["losses"]))}
        summary["Overall Accuracy"] = str(metrics["overall_accuracy"])
        summary["Mean Class Accuracy"] = str(metrics["mean_class_accuracy"])
        # summary["IoU per Class"] = metrics["iou_per_class"]
        # summary["Average IoU"] = metrics["iou_average"]
        return summary

    summary = test_one_epoch()
    print("Testing summary:\n%s" % (misc.json.dumps(summary, indent=2)))


def train(model, train_loader, valid_loader, args):
    """Trainer function for PointNet
    """

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and loss
    model = model.to(device)
    ce_loss = torch.nn.CrossEntropyLoss()

    # Set optimizer (default SGD with momentum)
    if args.use_adam:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True)

    # Get current state
    state = misc.persistence(args.logdir, args.model_path, module_name=model.__class__.__module__, main_file=__file__)
    init_epoch = state["epoch"]

    if state["model_state_dict"]:
        print("Loading pre-trained model from %s"%args.model_path)
        model.load_state_dict(state["model_state_dict"])

    if state["optimizer_state_dict"]:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  milestones=np.arange(
                                                      args.decay_step,
                                                      args.epochs,
                                                      args.decay_step).tolist(),
                                                  gamma=args.lr_decay,
                                                  last_epoch=init_epoch-1)


    def train_one_epoch():
        iterations = tqdm(train_loader, ncols=100, unit='batch', leave=False)
        ep_sum = run_one_epoch(model, iterations, "train", ce_loss, optimizer=optimizer, loss_update_interval=10)

        summary = {"Loss/train": np.mean(ep_sum["losses"])}
        return summary

    def eval_one_epoch():
        iterations = tqdm(valid_loader, ncols=100, unit='batch', leave=False, desc="Validation")
        ep_sum = run_one_epoch(model, iterations, "test", ce_loss, get_locals=True, loss_update_interval=-1)

        preds = ep_sum["logits"].argmax(axis=-1)
        metrics = get_segmentation_metrics(ep_sum["labels"], preds)

        summary = {"Loss/validation": np.mean(ep_sum["losses"])}
        summary["Overall Accuracy"] = metrics["overall_accuracy"]
        summary["Mean Class Accuracy"] = metrics["mean_class_accuracy"]
        # summary["IoU per Class"] = metrics["iou_per_class"]
        # summary["Average IoU"] = metrics["iou_average"]
        return summary

    # Train for multiple epochs
    tensorboard = SummaryWriter(log_dir=misc.join_path(args.logdir, "logs"))
    tqdm_epochs = tqdm(range(init_epoch, args.epochs), total=args.epochs, initial=init_epoch, unit='epoch', ncols=100, desc="Progress")
    for e in tqdm_epochs:
        train_summary = train_one_epoch()
        valid_summary = eval_one_epoch()
        # valid_summary={"Loss/validation":0}
        summary = {**train_summary, **valid_summary}
        summary["LearningRate"] = lr_scheduler.get_lr()[-1]

        # Write summary
        for name, val in summary.items():
            tensorboard.add_scalar(name, val, global_step=e+1)

        # Update learning rate and save checkpoint
        lr_scheduler.step()
        misc.save_checkpoint(args.logdir, {
            "epoch": e+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": summary["Loss/validation"]
        })

        if args.print_summary:
            tqdm_epochs.clear()
            train_loss, eval_loss = train_summary["Loss/train"], valid_summary["Loss/validation"]
            print("Epoch %d: Loss(T): %.4f, Loss(V): %.4f" % (e+1, train_loss, eval_loss))


if __name__ == "__main__":
    main()
