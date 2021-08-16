# Non-Euclidean Neural Networks
Current project on understanding and developing novel neural network models in a hyperbolic setting. Here we are using the PyTorch framework and the geoopt package (https://github.com/geoopt/geoopt). 

Please ensure that the PyTorch version installed is >= 1.9.0. The geeopt package can be installed using pip:
```
pip install git+https://github.com/geoopt/geoopt.git
```

## Current status of project:
- Successfully constructed a feed-forward hyperbolic neural network model. We experimented on the performance of the model by running classification tasks on classic data sets such as MNIST, MNIST-Kuzushiji and CIFAR-10.  
- We found that the model's capability is sufficient for simpler data sets such as MNIST digits, where we obtained an average class prediction accuracy of ~98%
- The performance of the model was slightly lower for the MNIST-Kuzushiji data, with an average class prediction accuracy of ~90%. We suspect that the model could not generalize to learn the relevant features of the data set, as there are more intricate features (eg. more complicated strokes) for the japanese handwritten words.
- The model performed even worse for the CIFAR-10 data set, where we obtained an average class prediction accuracy of ~50%, even for a binary classification task. We experimented with different combinations of loss functions, optimizers and hyperparameters, but to no avail.
- We decided to move on to a different type of architecture, namely the Variational Auto-Encoders (VAEs). Currently, we are working on building a Euclidean VAE with the help of pytorch-lighting. The use of pytorch-lighting was used to help us streamline the model's pipeline
