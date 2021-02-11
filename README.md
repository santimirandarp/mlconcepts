# Newbie ML
This repository holds attempts to understand how neural networks work from a begginers point of view. 

There are a few basic steps we always have to go through:

1. Find the data
This step can take us a while. To practice there are ready-to-use datasets, but otherwise we need to create it. I've gone through this process here. 

On this repo, standard datasets are used because the main goal is understanding the neural network: performance, accuracy, basic steps and differences, variation with iteration's cycles, and so on.

2. Treat the data
Datasets will probably need to be cleaned up, normalized and split into training, validation and test. We use just one split into training and test, mostly because datasets are small.

3. Define a neural network 
Depending on the problem we define an architecture and size of the network (width and depth). 

On this set of problems, we start off by using single or multivariate models (regression or classification). Nice enough, the backpropagation is almost the same, as indicated on wikipedia (and it's deduced from scratch on the docs).

For multiclassification, a model is built up from scratch; and same thing for the first deep network (a shallow network).

The last step is to create fully connected deep networks from scratch.

4. Calculate predictions and cost
This the forward propagation step. In a few examples we try different functions (for deep NNs).

5. Update parameters
This is the essential step, where the model improves itself (learns). All models are deduced using gradient descent to drive the learning process.
