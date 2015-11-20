# Evaluation of Deep Learning Toolkits 
In this study, I evaluate some popular deep learning toolkits. The candidates are listed in alphabetical order: [TensorFlow](https://github.com/tensorflow/tensorflow), [Theano](https://github.com/Theano/Theano), and [Torch](https://github.com/torch/torch7) [0]. This is a dynamic document and the evaluation, to the best of my knowledge, is based on the current state of their code.

I also provide ratings in each area because for a lot of people, ratings are useful. However, keep in mind that ratings are inherently subjective [1].

If you find something wrong or incomplete, please help improve by creating an issue.

## Modeling Capability
In this section, we evaluate each toolkit's ability to train common and state-of-the-art networks <u>without writing too much code</u>. Some of these networks are:
- ConvNets: AlexNet, OxfordNet, GoogleNet
- RecurrentNets: plain RNN, LSTM/GRU, bidirectional RNN
- Sequential modeling with attention.

In addition, we also evaluate the flexibility to create a new type of model.

#### TensorFlow: 4
**For state-of-the-art models**
- RNN API and implementation are suboptimal. The team also commented about it [here](https://github.com/tensorflow/tensorflow/issues/7) and [here](https://groups.google.com/a/tensorflow.org/forum/?utm_medium=email&utm_source=footer#!msg/discuss/B8HyI0tVtPY/aR43OIuUAwAJ).
- Bidirectional RNN [not available yet](https://groups.google.com/a/tensorflow.org/forum/?utm_medium=email&utm_source=footer#!msg/discuss/lwgaL7WEuW4/UXaL4bYkAgAJ)
- No 3D convolution, which is useful for video recognition

**For new models**
In TF as in Theano, a network is specified as a symbolic graph of vector operations, such as matrix add/multiply or convolution. A layer is just a composition of those operations. The fine granularity of the building blocks (operations) allows users to invent new complex networks without worrying about backpropagation.

The public release of TF doesn’t yet support loop and condition controls in the graph definition. This makes RNN implementations less ideal because they have to use Python loops and no graph compiler optimization can be made.

Google claimed to have this in their [white paper](http://download.tensorflow.org/paper/whitepaper2015.pdf) and [details are still being worked out](https://github.com/tensorflow/tensorflow/issues/208).


#### Theano: 5
**State-of-the-art models**
Theano has implementation for almost all of state-of-the-art networks, either in the form of a higher-level framework (e.g. [Blocks](https://github.com/mila-udem/blocks), [Keras](https://github.com/fchollet/keras), etc.) or in pure Theano. In fact, many recent research ideas (e.g. attentional model) started here.

**New models**
Theano pioneered the trend of using symbolic graph for programming a network. Theano's symbolic API supports looping control, so-called [scan], which makes implementing RNNs easy and efficient. Users don't always have to define a new model at the tensor operations level. There are a few higher-level frameworks, mentioned above, which make model definition and training simpler.

#### Torch: 4.5
**State-of-the-art models**
- Excellent for conv nets. It's worth noting that temporal convolution can be done in TensorFlow/Theano via `conv2d` but that's a trick. The native interface for temporal convolution  in Torch makes it slightly more intuitive to use. 
- Rich set of RNNs available through a [non-official extension](https://github.com/Element-Research/rnn) [2]

**New models**
In Torch, there are multiple ways (stack of layers or graph of layers) to define a network but essentially, a network is defined as a graph of layers. Because of this coarser granularity, Torch is considered less flexible because for new layer types. For new layer types, users have to implement the full forward, backward, and gradient input update.

For those familiar with Caffe, this layerwise design is similar to Caffe. However, defining a new layer in Torch is much easier because you don't have to program in C++. Plus, in Torch, the difference between new layer definition and network definition is minimal. In Caffe, layers are defined in C++ while networks are defined via `Protobuf`.

<center>
<img src="http://i.snag.gy/0loNv.jpg" height="450">  <img src="https://camo.githubusercontent.com/49ac7d0f42e99d979c80a10d0ffd125f4b3df0ea/68747470733a2f2f7261772e6769746875622e636f6d2f6b6f7261796b762f746f7263682d6e6e67726170682f6d61737465722f646f632f6d6c70335f666f72776172642e706e67" height="450"><br>
<i>Left: graph model of CNTK/Theano/TensorFlow; Right: graph model of Caffe/Torch</i>
</center>


## Interfaces

#### TensorFlow: 4.9
TF supports two interfaces: Python and C++. This means that you can do experiments in a rich, high-level environment and deploy your model in an environment that requires native code or low latency.  

It would be perfect if TF supports `F#` or `TypeScript`. The lack of static type in Python is just ... painful :).

#### Theano: 4.5
Python

#### Torch: 4
Torch runs on LuaJIT, which is amazingly fast (comparable with industrial languages such as C++/C#/Java). Hence developers don't have to think about symbolic programming, which can be limited, when using Torch. They can just write all kinds of computations without worrying about performance penalty.

However, let's face it, Lua is not yet a mainstream language.

## Model Deployment
How easy to deploy a new model?

#### TensorFLow: 4.5
TF supports C++ interface and the library can be compiled/optimized on ARM architectures because it uses [Eigen](eigen.tuxfamily.org) (instead of a BLAS library). This means that you can deploy your trained models on a variety of devices (servers or mobile devices) without having to implement a separate model decoder or load Python/LuaJIT interpreter [3].

TF doesn't work on Windows yet so TF models can't be deployed on Windows devices though.

#### Theano: 3
The lack of low-level interface and the inefficiency of Python interpreter makes Theano less attractive for industrial users. For a large model, the overhead of Python isn’t too bad but the dogma is still there.

The cross-platform nature (mentioned below) enables a Theano model to be deployed in a Windows environment. Which helps it gain some points.

#### Torch: 3
Torch require LuaJIT to run models. This makes it less attractive than bare bone C++ support of TF. It’s not just the performance overhead, which is minimal. The bigger problem is integration, at API level, with a bigger production  pipeline.


## Performance
### Single-GPU
All of these toolkits call cuDNN so as long as there’s no major computations or memory allocations at the outer level, they should perform similarly.

Soumith@FB has done some [benchmarking for ConvNets](https://github.com/soumith/convnet-benchmarks). Deep Learning is not just about feedforward convnets, not just about ImageNet, and certainly not just about a few passes over the network. However, Soumith’s benchmark is the only notable one as of today. So we will base the Single-GPU performance rating based on his benchmark.

#### TensorFLow: 3
TF only uses cuDNN v2 and even so, its performance is ~1.5x slower than Torch with cuDNN v2. It also runs out of memory when training GoogleNet with batch size 128. More details [here](https://github.com/soumith/convnet-benchmarks/issues/66).

A few issues have been identified in that thread: excessive memory allocation, different tensor layout from cuDNN’s, no in-place op, etc.

#### Theano: 3
On big networks, Theano’s performance is on par with Torch7, according to [this benchmark](http://arxiv.org/pdf/1211.5590v1.pdf). The main issue of Theano is startup time, which is terrible, because Theano has to compile C/CUDA code to binary. We don’t always train big models. In fact, DL researchers often spend more time debugging than training big models. TensorFlow doesn’t have this problem. It simply maps the symbolic tensor operations to the already-compiled corresponding function calls.

Even `import theano` takes time because this `import` apparently does a lot of stuffs. Also, after `import Theano`, you are stuck with a pre-configured device (e.g. `GPU0`).

#### Torch: 5
Simply awesome without the \*bugs\* that TensorFlow and Theano have.

### Multi-GPU
I haven’t yet tried training these toolkits on multiple GPUs so this evaluation is currently about the ease for multi-GPU and/or distributed training.

#### TensorFlow: 4
The programming model for using multiple GPUs in a single box is fairly straight-forward. The memory transfer from GPU to CPU and aggregating results from multiple GPUs is fairly seamless. TF provides an [example here](https://github.com/tensorflow/tensorflow/blob/1d76583411038767f673a0c96174c80eaf9ff42f/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py).

Distributed training isn't publicly available yet but [there's a plan](https://github.com/tensorflow/tensorflow/issues/23) and it's high-priority.

#### Theano: 2
Theano doesn’t support multi-GPU natively. There’s a lot of low-level programming that programmers need to do, e.g. spawning processes and aggregating results using the multiprocessing library. A tutorial is provided [here](https://github.com/Theano/Theano/wiki/Using-Multiple-GPUs).

#### Torch: 3.
Torch’s multi-GPU training (available by using [fbcunn](https://github.com/facebook/fbcunn) package) looks less seamless than TensorFlow but more so than Theano. [Here’s an illustrating example](https://github.com/soumith/imagenet-multiGPU.torch/blob/master/train.lua).

There's no known plan for distributed training. For most organizations, distributed training is a hype though. Even if a company or a team can afford a cluster of GPUs, running jobs in parallel (due to hyper-parameter sweep and/or multiple users) often provides better utilization of the cluster.

## Model Debugging
TF has a visualization companion called TensorBoard. Using TensorBoard, you can track any variable (weights change, accuracy, etc.) over time. This is useful for debugging and analyzing the learning curves of models. The [logging mechanism](http://tensorflow.org/how_tos/summaries_and_tensorboard/index.html#serializing-the-data) is fairly simple.


## Architecture
Developer Zone

#### TensorFlow: 5
TF has a clean, modular architecture with multiple frontends and execution platforms. Details are in the [white paper](http://download.tensorflow.org/paper/whitepaper2015.pdf).

<img src="http://i.snag.gy/sJlZe.jpg" width="500">

#### Theano: 3
The architecture is fairly hacky: the whole code base is Python where C/CUDA code is packaged as Python string. This makes it hard to navigate, debug, refactor, and hence contribute as developers.

#### Torch: 5
Torch7 and nn libraries are also well-designed with clean, modular interfaces.

## Ecosystem
#### TensorFlow: 5
Python and C++

#### Theano: 4
Python

#### Torch: 3
Lua is not a mainstream language and hence libraries built for it are not as rich as ones built for Python.

Plus, there’s an increasingly-popular JIT project for Python, called [PyPy](http://pypy.org/). The advantage of LuaJIT would become minimal if/when PyPy becomes as fast as LuaJIT and fully-compatible with CPython.


## Cross-platform
While Theano works on all OSes, TF and Torch do not work on Windows and there's no known plan to port from either camp.

<br>
___ 

**End Notes**

[0] There are other popular toolkits that I haven’t included in my review yet for a wide variety of reasons: [Caffe](https://github.com/BVLC/caffe), [CNTK](http://cntk.codeplex.com), [MXNet](https://github.com/dmlc/mxnet).

[1] Note that I don’t aggregate ratings because different users/developers have different priorities.

[2] Disclaimer: I haven’t analyzed this extension carefully.

[3] See my [blog post](http://www.kentran.net/2014/12/challenges-in-machine-learning-practice.html) for why this is desirable.