# OpenVino Post-training Optimization Demo


This is a demo code that shows how to use OpenVino Post-training Quantization (POT) framework to optimize Deep Neural Networks (DNNs) by using the quantization technique. Swin transformer and MobilenetV2 models are considered in this demo. 

## Installation
The code uses the openvino 2022.3.0 version and may not work with the newest versions. Please use the requirements.txt to install the dependencies

```
pip install -r requirements.txt
```


## Requirements
In order the run the experiments using the models pretrained on ImageNet, validation data need to be downloaded and placed in ```data/val```. Note that the data folder is created when you run the main script for the first time.

The code expects that the dataset is organized in a particular hierarchy. Please refer to this repo  https://gist.github.com/antoinebrl/7d00d5cb6c95ef194c737392ef7e476a. 

## Usage 
To run the experiments, you need to run the main.py scripts. 
The default configuration will run the quantization on Swin transformer model pretrained on ImageNet, using the default POT configuration. 

```
python main.py
```

To run the experiment by exclusing LayerNorm layer from quantization, please run: 

```
python main.py --exclude_MVN 
```

You can use the arguments ```--model_name``` and ```--dataset_name``` to specify the model name and the dataset used for evaluation and calibration. For example: 

```
python main.py --model_name mobilenet_v2 --dataset_name cifar10
```
