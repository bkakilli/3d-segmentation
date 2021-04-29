import importlib

from torch.utils.data import DataLoader

def get_loaders(args):

    print("Loading dataset.")
    dataset = importlib.import_module(".dataset", package="datasets."+args.dataset)

    dataloaders = []
    for split in ["train", "val", "test"]:
        d = dataset.Dataset(split=split, **vars(args))
        dl = DataLoader(d, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=dataset.custom_collate_fn)
        dataloaders.append(dl)

    return dataloaders