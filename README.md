# Evaluation of Deep Learning Toolkits

**Warning**: this research was done in late 2015 with slight modifications in early 2016. Many toolkits have improved significantly since then.

**Abstract.** In this study, I evaluate some popular deep learning toolkits. The candidates are listed in alphabetical order: [Caffe](https://github.com/BVLC/caffe), [CNTK](https://cntk.codeplex.com/), [TensorFlow](https://github.com/tensorflow/tensorflow), [Theano](https://github.com/Theano/Theano), and [Torch](https://github.com/torch/torch7).

I also provide ratings in some areas because for a lot of people, ratings are useful. However, keep in mind that ratings are inherently subjective [1].

If you find something wrong or inadequate, please help improve by filing an issue.

**Table of contents**

1. [Modeling Capability](#modeling-capability)
- [Interfaces](#interfaces)
- [Model Deployment](#model-deployment)
- [Performance](#performance)
- [Architecture](#architecture)
- [Ecosystem](#ecosystem)
- [Cross-platform](#cross-platform) 

___

## Modeling Capability
In this section, we evaluate each toolkit's ability to train common and state-of-the-art networks <u>without writing too much code</u>. Some of these networks are:

- ConvNets: AlexNet, OxfordNet, GoogleNet
- RecurrentNets: plain RNN, LSTM/GRU, bidirectional RNN
- Sequential modeling with attention.

In addition, we also evaluate the flexibility to create a new type of model.

#### Caffe <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
Caffe is perhaps the first mainstream industry-grade deep learning toolkit, started in late 2013, due to its excellent convnet implementation (at the time). It is still the most popular toolkit within the computer vision community, with many extensions being actively added. 

However, its support for recurrent networks and language modeling in general is poor, due to its legacy architecture, which's limitations are detailed in the [architecture section](#architecture).

#### CNTK <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_2_stars.png">
CNTK is a deep learning system started by the speech people who [started the deep learning craze](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.185.1908&rep=rep1&type=pdf) and grown into a more general platform-independent deep learning system. It is better known in the speech community than in the general deep learning community.

In CNTK (as in TensorFlow and Theano), a network is specified as a symbolic graph of vector operations, such as matrix add/multiply or convolution. A layer is just a composition of those operations. The fine granularity of the building blocks (operations) allows users to invent new complex layer types without implementing them in a low-level language (as in Caffe).

As of today, CNTK is not usable for a variety of tasks such as sequence-2-sequence.


#### TensorFlow <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_and_a_half_stars.png">
**State-of-the-art models**

- RNN API and implementation are suboptimal. The team also commented about it [here](https://github.com/tensorflow/tensorflow/issues/7) and [here](https://groups.google.com/a/tensorflow.org/forum/?utm_medium=email&utm_source=footer#!msg/discuss/B8HyI0tVtPY/aR43OIuUAwAJ).
- Bidirectional RNN [not available yet](https://groups.google.com/a/tensorflow.org/forum/?utm_medium=email&utm_source=footer#!msg/discuss/lwgaL7WEuW4/UXaL4bYkAgAJ)
- No 3D convolution, which is useful for video recognition

**New models**
Since TF uses symbolic graph of vector operations approach, specifying a new network is fairly easy. Although it doesn't support symbolic loop yet (at least not well tested/documented, as of 05/2016), RNNs can be made easy and efficient using the [bucketing trick](https://www.tensorflow.org/versions/r0.8/tutorials/seq2seq/index.html#bucketing-and-padding).

However, TF has a major weakness in terms of modeling flexibility. Every computational flow has be constructed as a static graph. That makes some computations difficult, such as [beam search](https://github.com/tensorflow/tensorflow/issues/654) (which is used frequently in sequence prediction tasks). 


#### Theano <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_and_a_half_stars.png">
**State-of-the-art models.** Theano has implementation for most state-of-the-art networks, either in the form of a higher-level framework (e.g. [Blocks](https://github.com/mila-udem/blocks), [Keras](https://github.com/fchollet/keras), etc.) or in pure Theano.

**New models.** Theano pioneered the trend of using symbolic graph for programming a network. Theano's symbolic API supports looping control, so-called [scan](http://deeplearning.net/software/theano/tutorial/loop.html), which makes implementing RNNs easy and efficient. Users don't always have to define a new model at the tensor operations level. There are a few higher-level frameworks, mentioned above, which make model definition and training simpler.

#### Torch <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_5_stars.png">
**State-of-the-art models**

- Excellent for conv nets. It's worth noting that temporal convolution can be done in TensorFlow/Theano via `conv2d` but that's a trick. The native interface for temporal convolution  in Torch makes it slightly more intuitive to use. 
- Rich set of RNNs available through a [non-official extension](https://github.com/Element-Research/rnn) [2]

**New models.** In Torch, there are multiple ways (stack of layers or graph of layers) to define a network but essentially, a network is defined as a graph of layers. Because of this coarser granularity, Torch is sometimes considered less flexible because for new layer types, users have to implement the full forward, backward, and gradient input update.

However, unlike Caffe, defining a new layer in Torch is much easier because you don't have to program in C++. Plus, in Torch, the difference between new layer definition and network definition is minimal. In Caffe, layers are defined in C++ while networks are defined via `Protobuf`.

Torch is more flexible than TensorFlow and Theano in that it is imperative while TF/Theano are declarative (i.e. one has to declare a computational graph). That makes some operations, e.g. beam search, much easier to do in Torch.

---
<center>
<img src="http://i.snag.gy/0loNv.jpg" height="450">  <img src="https://camo.githubusercontent.com/49ac7d0f42e99d979c80a10d0ffd125f4b3df0ea/68747470733a2f2f7261772e6769746875622e636f6d2f6b6f7261796b762f746f7263682d6e6e67726170682f6d61737465722f646f632f6d6c70335f666f72776172642e706e67" height="450"><br>
<i>Left: graph model of CNTK/Theano/TensorFlow; Right: graph model of Caffe/Torch</i>
</center>


## Interfaces

#### Caffe <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
Caffe has `pycaffe` interface but that's a mere secondary alternative to the command line interface. The model has to be defined in protobuf (usually with a plain text editor), even if you use `pycaffe`.

#### CNTK <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_2_and_a_half_stars.png">
The way to use CNTK, similar to Caffe, is to specify a config file and run command line. CNTK is slightly worse than Caffe because there's no Python or any other high-level language interface.


#### TensorFlow <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_and_a_half_stars.png">
TF supports two interfaces: Python and C++. This means that you can do experiments in a rich, high-level environment and deploy your model in an environment that requires native code or low latency.  

It would be perfect if TF supports `F#` or `TypeScript`. The lack of static type in Python is just ... painful :).

#### Theano <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_stars.png">
Python

#### Torch <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_stars.png">
Torch runs on LuaJIT, which is amazingly fast (comparable with industrial languages such as C++/C#/Java). Hence developers don't have to think about symbolic programming, which can be limited. They can just write all kinds of computations without worrying about performance penalty.

However, let's face it, Lua is not yet a mainstream language.

## Model Deployment
How easy to deploy a new model?

#### Caffe <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_5_stars.png">
Caffe is C++ based, which can be compiled on a variety of devices. It is cross-platform (windows port is available and maintained [here](https://github.com/MSRDL/caffe)). Which makes Caffe the best choice with respect deployment.

#### CNTK <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_and_a_half_stars.png">
Like Caffe, CNTK is also C++ based and is cross-platform. Hence, deployment should be easy in most cases. However, to my understanding, it doesn't work on ARM architecture, which limits its its capability on mobile devices. 

#### TensorFlow <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_4_and_a_half_stars.png">
TF supports C++ interface and the library can be compiled/optimized on ARM architectures because it uses [Eigen](http://eigen.tuxfamily.org) (instead of a BLAS library). This means that you can deploy your trained models on a variety of devices (servers or mobile devices) without having to implement a separate model decoder or load Python/LuaJIT interpreter [3].

TF doesn't work on Windows yet so TF models can't be deployed on Windows devices though.

#### Theano <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
The lack of low-level interface and the inefficiency of Python interpreter makes Theano less attractive for industrial users. For a large model, the overhead of Python isn’t too bad but the dogma is still there.

The cross-platform nature (mentioned below) enables a Theano model to be deployed in a Windows environment. Which helps it gain some points.

#### Torch <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
Torch require LuaJIT to run models. This makes it less attractive than bare bone C++ support of Caffe/CNTK/TF. It’s not just the performance overhead, which is minimal. The bigger problem is integration, at API level, with a larger production pipeline.


## Performance
### Single-GPU
All of these toolkits call cuDNN so as long as there’s no major computations or memory allocations at the outer level, they should perform similarly.

Soumith@FB has done some [benchmarking for ConvNets](https://github.com/soumith/convnet-benchmarks). Deep Learning is not just about feedforward convnets, not just about ImageNet, and certainly not just about a few passes over the network. However, Soumith’s benchmark is the only notable one as of today. So we will base the Single-GPU performance rating based on his benchmark.

#### TensorFlow and Torch <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_5_stars.png">

TensorFlow used to be slow when it first came out but as of 05/2016, it has reached the ballpark of other frameworks in terms of ConvNet speed. This is not surprising because every framework nowadays calls CuDNN for the actual computations.

Here's my latest micro benchmark of TensorFlow 0.8 vs before. The measurement is latency, in milliseconds, for one full minibatch forward-backward pass on a single Titan X GPU. 

| Network | TF 0.6 [[ref](https://github.com/soumith/convnet-benchmarks/blob/efb3d9321d14856f49951980dbea2f554190161a/README.md)]                                                                     | TF 0.8 [my run] | Torch FP32 [my run] |
|:------------------------:|:-----------------------------------------------------------------------------------------------------------:| ----------:| ------------:|
| AlexNet      | 292  | 97  |  81  |
| Inception v1 | 1237 | 518 |  470 |


#### Theano <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
On big networks, Theano’s performance is on par with Torch7, according to [this benchmark](http://arxiv.org/pdf/1211.5590v1.pdf). The main issue of Theano is startup time, which is terrible, because Theano has to compile C/CUDA code to binary. We don’t always train big models. In fact, DL researchers often spend more time debugging than training big models. TensorFlow doesn’t have this problem. It simply maps the symbolic tensor operations to the already-compiled corresponding function calls.

Even `import theano` takes time because this `import` apparently does a lot of stuffs. Also, after `import Theano`, you are stuck with a pre-configured device (e.g. `GPU0`).

### Multi-GPU
TBD 

## Architecture
Developer Zone

#### Caffe <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
Caffe's architecture was considered excellent when it was born but in the modern standard, it is considered average. The main pain points of Caffe are its layer-wise design in C++ and the protobuf interface for model definition.

**Layer-wise design.** The building block of a network in Caffe is layer. 
- For new layer types, you have to define the full forward, backward, and gradient update. You can see an  already [long-list of layers implemented in (official) caffe](https://github.com/BVLC/caffe/tree/master/src/caffe/layers).
- What's worse is that if you want to support both CPU and GPU, you need to implement extra functions, e.g. [`Forward_gpu` and `Backward_gpu`](https://github.com/BVLC/caffe/blob/master/src/caffe/layers/cudnn_conv_layer.cu).
- Worse, you need to assign an int id to your layer type and add that to the [proto file](https://github.com/BVLC/caffe/blob/master/src/caffe/proto/caffe.proto#L1046). If your pull request is not merged early, you may need to change the id because someone else already claims that.

**Protobuf.** Caffe has `pycaffe` interface but that's a mere replacement of the command line interface. The model has to be defined in protobuf (usually with a plain text editor), even if you use `pycaffe`.

[*Copied from [my own answer on Quora](https://www.quora.com/How-is-TensorFlow-architected-differently-from-Caffe)*]
 
### CNTK
To be updated ...

#### TensorFlow <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_5_stars.png">
TF has a clean, modular architecture with multiple frontends and execution platforms. Details are in the [white paper](http://download.tensorflow.org/paper/whitepaper2015.pdf).

<img src="http://i.snag.gy/sJlZe.jpg" width="500">

#### Theano <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_3_stars.png">
The architecture is fairly hacky: the whole code base is Python where C/CUDA code is packaged as Python string. This makes it hard to navigate, debug, refactor, and hence contribute as developers.

#### Torch <img src="http://www.wpclipart.com/signs_symbol/stars/5_star_rating_system/.cache/5_Star_Rating_System_5_stars.png">
Torch7 and nn libraries are also well-designed with clean, modular interfaces.

## Ecosystem
- Caffe and CNTK: C++
- TensorFlow: Python and C++
- Theano: Python
- Torch: Lua is not a mainstream language and hence libraries built for it are not as rich as ones built for Python.


## Cross-platform
Caffe, CNTK, and Theano work on all OSes. TensorFlow and Torch do not work on Windows and there's no known plan to port from either camp.

<br>
___ 

**Footnotes**

[1] Note that I don’t aggregate ratings because different users/developers have different priorities.

[2] Disclaimer: I haven’t analyzed this extension carefully.

[3] See my [blog post](http://www.kentran.net/2014/12/challenges-in-machine-learning-practice.html) for why this is desirable.
