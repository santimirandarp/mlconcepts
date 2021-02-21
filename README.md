# Newbie ML
This repository holds attempts to understand how neural networks work from a begginers point of view. 

## Contribute
* Clone `git clone https://www.github.com:santimirandarp/mlconcepts.git` 
* test the python scripts (**1**, **2**, **3**,..). 
* Push changes to repo.

Feel free to write new neural networks and place them in a numbered python file as the other networks.

## General Description
There are a few basic steps we usually have to go through:

1. Find the data
To practice [ready-to-use datasets](https://machinelearningmastery.com/results-for-standard-classification-and-regression-machine-learning-datasets/) are used, otherwise we need to create it. See the process of [creating datasets here](https://www.github.com:santimirandarp/mlimg.git). 

2. Treat/Prepare the data
Split into training and test. Datasets are cleaned up. Utility functions are defined to normalize and center data.

A few functions for this matter are under `csvfn.py`.

3. Define a Neural Network 
Depending on the problem we define a network type: _architecture_ and _size_ of the network (width and depth). 

We start off by building single or multivariate models (regression or classification). Next we look for more performance and flexibility using multiclassification, shallow networks and deep networks. All are fully connected NNs.

The last step is to create fully connected deep networks from scratch.

4. Calculate predictions and cost
This the forward propagation step. 

5. Update parameters
The model improves itself (learns). All models are deduced using gradient descent to drive the learning process.

## Some findings

* Non-normalized data will usually blow up calculations, as we find out testing (both in regression and classfication, as backpropagation is the same). The reason is gradients depend on features' magnitude.
* Standard Regression and Classification have the same backpropagation algorithm.

## General script
After writing many concrete algorithms we start to feel like creating more abstract functions for each task (they repeat again and again).

`functions.py` does exactly that for simple networks. It uses the fact that backpropagation is the same for regression and classification. 
