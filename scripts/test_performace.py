import numpy as np
import torch
import os
from models.model_sseg import HGCN
import argparse
from utils import data_loader

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
    args=get_arguments()
    checkpoint_path='./seg/log/checkpoints'
    load_check='ckpt_20200128_051127.pt'
    load_check_path=os.path.join(checkpoint_path,load_check)

    checkpoint=torch.load(load_check_path)
    # print(checkpoint.keys())
    network=HGCN(args)
    network.load_state_dict(checkpoint['model_state_dict'])  
    network.eval()
    
    
    acc_li=[]
    confusion_matrix=np.zeros((network.num_classes,network.num_classes))
    #confusion_matrix[i,j] means truth is i,prediction is j.
    
    
    train_loader, valid_loader, test_loader = data_loader.get_loaders(args)
    for step,(inpt,label) in enumerate(test_loader):
        if inpt.shape[0]==1:
            continue
        label=label.view(-1).numpy()
        out=network(inpt)
        out=out.cpu().detach().numpy()
        out=np.argmax(out,axis=2).reshape(-1)
        accuracy=(np.sum((label==out).astype(np.int)))/len(out)
        for i in range(out.shape[0]):
            truth=label[i]
            prediction=out[i]
            confusion_matrix[truth,prediction]+=1
        print('step: ',step,'acc:',accuracy)
        acc_li.append(accuracy)
        # np.save('acc_li',acc_ls)
    acc_ls=np.array(acc_li)
    print('thie mean is',np.mean(acc_ls))
    np.save('acc_li',acc_ls)
    np.save('confusion_ma',confusion_matrix)

if __name__=='__main__':
    main()

