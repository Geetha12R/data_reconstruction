<h1 align="center"> Reconstructing Training Data <br> from Trained Convolutional Neural Networks </h1>

<h3 align="center"> 
<a href="https://github.com/Geetha12R/" target="_blank">Geetha Ranganathan</a>*, 
</h3>

#### 

## Setup

Create a copy of ```setting.default.py``` with the name ```setting.py```, and make sure to change the paths inside to match your system. 

Create and initialize new conda environment using the supplied ```environment.yml``` file (using python 3.8 and pytorch 11.1.0 and CUDA 11.3) :
```
conda env create -f environment.yaml
conda activate rec
```


## Running the Code

### Notebooks
For quick access, start by running the provided notebook for analysing the (already provided) 
reconstructions for multi-class CIFAR10 (vehicles/animals) using a trained Convolutional Neural Network:

- ```reconstruction_cifar10_cnn.ipynb```
<!-- - ```reconstruction_mnist.ipynb``` -->


### Reproducing the provided trained models and their reconstructions

All training/reconstructions are done by running ```Main.py``` with the right parameters.  
Inside ```command_line_args``` directory we provide command-lines with necessary arguments 
for reproducing the training of the provided models and their provided reconstructions
(those that are analyzed in the notebooks)  


#### Training
For reproducing the training of the provided a trained CNN models (with 2 Conv-layers and 3 fully connected layers):

 - CIFAR10 model (for reproduction run ```command_line_args/train_cifar10_vehicles_animals_multi.txt```)

#### Hyperparameter Search

To find the right hyperparameters for reconstructing samples from the above models 
(or any other models in our paper) we used Weights & Biases sweeps.
In general, it is still an open question how to find the right hyperparameters 
for our losses without trial and error.

Try out different hyperparameter ranges by a ```sweep.yaml``` and run it with:
```wandb sweep <path>/sweep.yaml```

#### Reconstructions

In ```reconstructions``` directory we provide reconstruction (results of a single run) of the models above.


These reconstructions can be reproduced by running the following commandlines (the right hyperparameter can be found there):

- CIFAR10: ```command_line_args/reconstruct_cifar10_multi_args.txt``` 


### Training/Reconstructing New Learning Problems

One should be able to train/reconstruct models for new problems by adding a 
new python file under ```problems``` directory.

Each problem file should contain the logic of how to load the data and 
the parameters necessary to build the model. 


#### Note: This work is an extension of https://arxiv.org/pdf/2206.07758.pdf
