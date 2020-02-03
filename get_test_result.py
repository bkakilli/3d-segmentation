from scripts.test_performace import get_matrix
from scripts.get_accuracy import accuracy_calculation
import argparse
from models.model_sseg import HGCN


def get_arguments():

    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')

    parser.add_argument('--train', action='store_true', help='Trains the model if provided')
    parser.add_argument('--test', action='store_true', help='Evaluates the model if provided')
    parser.add_argument('--dataset', type=str, default='s3dis', choices=['shapenetparts'], help='Experiment dataset')
    parser.add_argument('--prefix', type=str, default='', help='Path prefix')
    parser.add_argument('--logdir', type=str, default='log', help='Name of the experiment')
    parser.add_argument('--model_path', type=str, help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=2, help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, help='Number of episode to train')
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

if __name__=='__main__':
    arg=get_arguments()
    arg.test=True
    model=HGCN(arg)
    confusion_matrix=get_matrix(arg,model)
    cal_class=accuracy_calculation(confusion_matrix)
    print(calcula_class.get_over_all_accuracy())
    print(calcula_class.get_intersection_union_per_class())
    print(calcula_class.get_average_intersection_union())

