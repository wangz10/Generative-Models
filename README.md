# Lecture on Generative Models

## Getting started

1. Run the Jupyter Notebook Server 

The Jupyter notebook server can be started through the Docker image [jupyter/tensorflow-notebook](https://hub.docker.com/r/jupyter/tensorflow-notebook/) by running the shell script:

```
$ ./run_docker.sh
```

2. Train the Deep Generative Models 

GPU is not required to train these model because of the small dataset size and relatively small neural networks. Training of the models can be done by running the [Train_VAE_on_MNIST.ipynb](https://github.com/wangz10/Generative-Models/blob/master/Train_VAE_on_MNIST.ipynb) and [Train_BiGAN_on_MNIST.ipynb](https://github.com/wangz10/Generative-Models/blob/master/Train_BiGAN_on_MNIST.ipynb) notebooks.

3. Run the [Main Tutorial](https://github.com/wangz10/Generative-Models/blob/master/Main.ipynb)

## References

- [Ng AY & Jordan MI: On Discriminative vs. Generative classifiers](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
- [Kingma & Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Goodfellow IJ et al: Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Donahue et al: Adversarial Feature Learning](https://arxiv.org/abs/1605.09782)
- [Flow-based Deep Generative Models](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
- [Glow: Better Reversible Generative Models](https://blog.openai.com/glow/)
