
## Installation (for python3)
First install pytorch as detailed here: https://pytorch.org/get-started/locally/

In your virtual environment, run command:
```bash
# To install dependencies
pip install -r requirements.txt
```


### Download
Go to Google drive folder: https://drive.google.com/drive/folders/1AouXCya-Nlb5gWyiy2Ke7d5t6gwS9nRf?usp=sharing

Download test_results.npz and checkpoint file (.pt extension) into main folder.

If you will run test and train, download ShapeNetParts dataset and put into `data/shapenetcore_partanno_segmentation_benchmark_v0_normal` folder. Download link: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip

## Displaying the saved results
```bash
python generate_results.py test_results.npz
```

## Testing the model

```bash
# Run pretrained model
python main_partseg.py --test --model_path=ckpt_20191203_202957.pt
```

## Training a new model
```bash
python main_partseg.py --train --logdir=logs/my_training

# To see what hyper-parameters you can change, run:
python main_partseg.py --help
```

The trained model will be saved in `logs/my_training/checkpoints` folder after every epoch.

### Tensorboard
Training generates tensorboard logs. Simply run;
```bash
tensorboard --logdir=logs --port=6006
```
in a seperate terminal. Then in your browser; http://localhost:6006