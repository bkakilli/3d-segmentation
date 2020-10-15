import argparse

import numpy as np

from scripts.generate_results import get_segmentation_metrics
from models.model_sseg_rev import HGCN
from utils import misc, data_loader

import torch
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from svstools.misc import slack_message


def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    parser.add_argument('--train', action='store_true', help='Trains the model if provided')
    parser.add_argument('--test', action='store_true', help='Evaluates the model if provided')
    parser.add_argument('--dataset', type=str, default='s3dis', help='Experiment dataset')
    parser.add_argument('--root', type=str, default=None, help='Path to data')
    parser.add_argument('--crossval_id', type=int, default=1, help='Split ID to train')
    parser.add_argument('--logdir', type=str, default='log', help='Name of the experiment')
    parser.add_argument('--model_path', type=str, help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of episode to train')
    parser.add_argument('--use_adam', action='store_true', help='Uses Adam optimizer if provided')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Learning rate decay rate')
    parser.add_argument('--decay_step', type=float, default=20, help='Learning rate decay step')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay rate (L2 regularization)')
    parser.add_argument('--workers', type=int, default=0, help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--print_summary', type=bool,  default=True, help='Whether to print epoch summary')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA id. -1 for CPU')
    parser.add_argument('--no_augmentation', action='store_true', help='Disables training augmentation if provided')
    parser.add_argument('--no_parallel', action='store_true', help='Forces to use single GPU if provided')
    parser.add_argument('--webhook', type=str, default='', help='Slack Webhook for notifications')

    return parser.parse_args()


def main():
    # Temporary fix for:
    # RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.
    # torch.backends.cudnn.enabled = False

    args = get_arguments()

    # Seed RNG
    misc.seed(args.seed)

    train_loader, valid_loader, test_loader = data_loader.get_loaders(args)

    # dimensions = [input_dim, local_feat_dim, graph_feat_dim]
    # k = [k_local, k_graph]
    config = {
        "hierarchy_config": [
            {"h_level": 5, "dimensions": [6, 32, 64], "k": [16, 8]},
            {"h_level": 3, "dimensions": [64, 128, 128], "k": [8, 4]},
            {"h_level": 1, "dimensions": [128, 256, 256], "k": [16, 8]},
        ],
        "input_dim": 6,
        "classifier_dimensions": [512, train_loader.dataset.num_labels],
    }
    model = HGCN(**config)
    if torch.cuda.device_count() > 1 and not args.no_parallel:
        model = torch.nn.DataParallel(model)

    if args.train:
        train(model, train_loader, valid_loader, args)

    if args.test:
        test(model, test_loader, args)


def run_one_epoch(model, tqdm_iterator, mode, get_locals=False, optimizer=None, loss_update_interval=1000):
    """Definition of one epoch procedure.
    """
    if mode == "train":
        assert optimizer is not None
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

    for i, batch_cpu in enumerate(tqdm_iterator):
        # X, groups, y = X_cpu.to(device), G_cpu.to(device), y_cpu.to(device)
        batch = misc.move_to(batch_cpu, device)
        X, y, meta = batch
        X_cpu, y_cpu, meta_cpu = batch_cpu

        if optimizer:
            optimizer.zero_grad()

        logits = model(X, meta)
        loss_fcn = model.module.get_loss if isinstance(model, torch.nn.DataParallel) else model.get_loss
        loss = loss_fcn(logits, y)
        summary["losses"] += [loss.item()]

        if mode == "train":

            # Apply back-prop
            loss.backward()
            optimizer.step()

            # Display
            if loss_update_interval > 0 and i%loss_update_interval == 0:
                tqdm_iterator.set_description("Loss: %.3f" % (np.mean(summary["losses"])))

        if get_locals:
            summary["logits"] += [logits.cpu().detach().numpy()]
            summary["labels"] += [y_cpu.numpy()]

        # torch.cuda.empty_cache()

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

    # Get current state
    if args.model_path is not None:
        state = torch.load(args.model_path)
        model.load_state_dict(state["model_state_dict"])
    print("Loaded pre-trained model from %s"%args.model_path)

    def test_one_epoch():
        iterations = tqdm(test_loader, ncols=100, unit='batch', desc="Testing")
        ep_sum = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        preds = ep_sum["logits"].argmax(axis=-1)
        metrics = get_segmentation_metrics(ep_sum["labels"], preds)

        summary["Loss/test"] = np.mean(ep_sum["losses"])
        summary["Overall Accuracy"] = metrics["overall_accuracy"]
        summary["Mean Class Accuracy"] = metrics["mean_class_accuracy"]
        summary["IoU per Class"] = np.array2string(metrics["iou_per_class"], 1000, 3, False)
        summary["Average IoU"] = metrics["iou_average"]
        return summary

    summary = test_one_epoch()
    summary_string = misc.json.dumps(summary, indent=2)
    print("Testing summary:\n%s" % (summary_string))

    if args.webhook != '':
        slack_message("Test result in %s:\n%s" % (args.logdir, summary_string), url=args.webhook)


def train(model, train_loader, valid_loader, args):
    """Trainer function for PointNet
    """
    if args.webhook != '':
        slack_message("Training started in %s (%s)" % (args.logdir, misc.gethostname()), url=args.webhook)
        

    # Set device
    assert args.cuda < 0 or torch.cuda.is_available()
    device_tag = "cpu" if args.cuda == -1 else "cuda:%d"%args.cuda
    device = torch.device(device_tag)

    # Set model and label weights
    model = model.to(device)
    model.labelweights = torch.tensor(train_loader.dataset.labelweights, device=device, requires_grad=False)

    # Set optimizer (default SGD with momentum)
    if args.use_adam:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)

    # Get current state
    state = misc.persistence(args, module_name=model.__class__.__module__, main_file=__file__)
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
        ep_sum = run_one_epoch(model, iterations, "train", optimizer=optimizer, loss_update_interval=1)

        summary = {"Loss/train": np.mean(ep_sum["losses"])}
        return summary

    def eval_one_epoch():
        iterations = tqdm(valid_loader, ncols=100, unit='batch', leave=False, desc="Validation")
        ep_sum = run_one_epoch(model, iterations, "test", get_locals=True, loss_update_interval=-1)

        preds = ep_sum["logits"].argmax(axis=-2)
        metrics = get_segmentation_metrics(ep_sum["labels"], preds)

        summary = {}
        summary["Loss/validation"] = float(np.mean(ep_sum["losses"]))
        summary["Overall Accuracy"] = float(metrics["overall_accuracy"])
        summary["Mean Class Accuracy"] = float(metrics["mean_class_accuracy"])
        summary["IoU per Class"] = metrics["iou_per_class"].reshape(-1).tolist()
        summary["Average IoU"] = float(metrics["iou_average"])
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

        if args.print_summary:
            tqdm_epochs.clear()
            print("Epoch %d summary:\n%s\n" % (e+1, misc.json.dumps((summary), indent=2)))

        # Update learning rate and save checkpoint
        lr_scheduler.step()
        misc.save_checkpoint(args.logdir, {
            "epoch": e+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": summary["Loss/validation"],
            "summary": summary
        })

        # Write summary
        for name, val in summary.items():
            if name == "IoU per Class": continue
            tensorboard.add_scalar(name, val, global_step=e+1)

    if args.webhook != '':
        slack_message("Training finished in %s (%s)" % (args.logdir, misc.gethostname()), url=args.webhook)

if __name__ == "__main__":
    main()
