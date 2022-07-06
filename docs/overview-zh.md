# Overview

## 背景: JAX

[JAX](https://github.com/google/jax) 等价于 NumPy + autodiff + GPU/TPU。

JAX有一套类似NumPy的API来支持快速科学计算和机器学习。

JAX包含了很多功能强大的转换，可以任意组合:

* 自动微分（Autodiff） (`jax.grad`): 高阶导数
* JIT编译 (`jax.jit`): tracePython函数，优化提速
* 向量化 (`jax.vmap`): 自动批处理
* 并行化 (`jax.pmap`): 跨加速卡（包括TPU pods）并行代码

如果你此前并没有用过JAX，可以先查看 [JAX for the impatient](notebooks/jax_for_the_impatient) 。

## Flax

[Flax](https://github.com/google/flax) 是一个高性能的神经网络库，底层由JAX支撑。

Flax团伙和JAX团队之间联系紧密，为训练神经网络提供：

* **神经网络 API** (`flax.linen`): Dense, Conv, {Batch|Layer|Group} Norm, Attention, Pooling, {LSTM|GRU} Cell, Dropout

* **Utilities and patterns**: replicated training, serialization and checkpointing, metrics, prefetching on device

* **Educational examples** that work out of the box: MNIST, LSTM seq2seq, Graph Neural Networks, Sequence Tagging

* **Fast, tuned large-scale end-to-end examples**: CIFAR10, ResNet on ImageNet, Transformer LM1b

## Code Examples

See the [What does Flax look like](https://github.com/google/flax#what-does-flax-look-like) section of our README.


## TPU support

所有的例子都在TPU上运行:

* [Launching jobs on Google Cloud](https://github.com/google/flax/tree/main/examples/cloud): provides a simple script that can be used to create a new VM on Google Cloud, train an example on that VM and then shutting it down.
* [Flax Examples](https://github.com/google/flax/tree/main/examples): Some of our examples requiring GPU/TPU support have instructions on how to run them on these devices (see `imagenet` and `wmt`).
* [Cloud TPU VM Quickstart](https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm): A brief introduction to working with JAX and Cloud TPU.
