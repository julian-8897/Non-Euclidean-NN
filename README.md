# Non-Euclidean Neural Networks
Current project on understanding and developing novel neural network models in a hyperbolic setting. Here we are using the PyTorch framework and the geoopt package (https://github.com/geoopt/geoopt). 

Please ensure that the PyTorch version installed is >= 1.9.0. The geeopt package can be installed using pip:
```
pip install git+https://github.com/geoopt/geoopt.git
```

Models Implemented:
1. Euclidean & Hyperbolic Feed-forward networks, evaluated on MNIST, K-MNIST, CIFAR-10 data sets.
2. Euclidean & Hyperbolic VAE, evaluated on MNIST, and CelebA data sets.

Notes:
1. Model architectures can be found under the 'models' package
2. Additional supplementary modules can be found under the 'hypmath' package
