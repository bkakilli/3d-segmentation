
### Download
Go to Google drive folder: https://drive.google.com/drive/folders/1AouXCya-Nlb5gWyiy2Ke7d5t6gwS9nRf?usp=sharing

Download results.zip, data.zip, and logs.zip. Extract the contents in the main folder.
```bash
unzip data.zip logs.zip results.zip
```

## Installation
### (Option 1): docker way

```bash
# Build the docker image
docker build -t ann/burak .
```
### (Option 2): classic (pip) way with python3
In your virtual environment, run command:
```bash
# To install dependencies
pip install -r requirements.txt
```

## Displaying the saved results

*(Display the part segmentations is only available with the classic way)*

Docker way:
```bash
docker run --rm -it -v $PWD:/seg ann/burak \
    python generate_results.py --no_visualization test_results.npz
```

Classic way **(With visualizations)**
```bash
python generate_results.py test_results.npz
```

## Testing the model

Docker way:
```bash
MODEL_PATH=logs/first_partseg/checkpoints/ckpt_20191203_202957.pt

# Run pretrained model
docker run --rm -it -v $PWD:/seg ann/burak \
    python main_partseg.py --test --model_path=$MODEL_PATH
```

Classic way:
```bash
MODEL_PATH=logs/first_partseg/checkpoints/ckpt_20191203_202957.pt

# Run pretrained model
python main_partseg.py --test --model_path=$MODEL_PATH
```

## Training a new model
Docker way:
```bash
docker run --rm -it -v $PWD:/seg ann/burak \
    python main_partseg.py --train --logdir=logs/my_training
```

Classic way:
```bash
python main_partseg.py --train --logdir=logs/my_training
```