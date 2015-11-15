#TensorFlow vs. Theano vs. Torch
In this study, I evaluate some popular deep learning frameworks. The candidates are listed in alphabet order: [TensorFlow](https://github.com/tensorflow/tensorflow), [Theano](https://github.com/Theano/Theano), and [Torch](https://github.com/torch/torch7). This is a dynamic document and the evaluation is based the current state of their code, not what the authors claim in white papers. 

This evaluation is mostly technical, to the best of my knowledge, and doesn't take into account the community size or who use what. If you find something wrong or incomplete, please help improve by creating an issue.

##TensorFlow
<img src="http://www.androidcentral.com/sites/androidcentral.com/files/styles/large/public/article_images/2015/11/tensorflow.png" width="200">

###Strengths
* Similar to Theano, TF allows specifying a symbolic graph of the network architecture via the Python interface. Since Theano pioneered the movement of symbolic graph and auto-gradient, I'll explain why this is great in Theano's section.
* Quick deployment of model: trained models can be deployed easily on a variety of devices (servers or mobile devices) without having to implement a separate model decoder or load Python/LuaJIT interpreter [1]. 
* TensorBoard for visualization of the network architectures and the learning curves of trained models

###Weaknesses
* TF performs much worse than its competitors, in both speed and memory usage, according to a [benchmark study by Soumith](https://github.com/soumith/convnet-benchmarks/issues/66). Note: Google is fixing these performance bugs. I'll update this remark accordingly when the issue is resolved.
* Symbolic loops (like `scan` in Theano) isn't ready (see [discussion here](https://github.com/tensorflow/tensorflow/issues/208)). Symbolic loops allow loops in a network (such as RNN) to be compiled, for better efficiency, rather than interpreted by Python (more details [here](http://deeplearning.net/software/theano/tutorial/loop.html)).

##Theano
<img src="http://deeplearning.net/software/theano/_static/theano_logo_allblue_200x46.png" width="200">
###Strengths
Theano is the first framework that uses the symbolic tensor graph [2] to specify a model. Any network can be represented as a graph of tensor flow. Hence, the tensor flow graph provides more flexibility than the layerwise approach used by Torch or Caffe. In the Torch/Caffe approach, you would need to define a new layer (with forward, backward, and gradient update functions) if what you want isn't already in the existing repository of layers.

Why symbolic? The reason is efficiency. For example, you can specify a layer as `y = ReLU(W * x + b)` using Python syntax and only when you hit `run`, the graph processor will compile the graph into high-performance C++/CUDA code and carry out the computations.

**Other strengths**

* There are several higher-level APIs that are built on top of Theano such as [Blocks](https://github.com/mila-udem/blocks), [Keras](https://github.com/fchollet/keras), etc. to make Theano easier to use for certain class of users (e.g. ones who are more familiar with the layerwise design of Caffe and Torch).
* Cross-platform: it works on Windows while TensorFlow and Torch do not (or very hacky to install)

###Weaknesses
* The compilation process (from generated C++ code to binary) is slow. If you train a big net for several days, this overhead is nothing. However, we don't always train big models and this overhead becomes annoying. 
TensorFlow doesn't have this problem. It simply maps the symbolic tensor operations to the corresponding function calls that are already compiled with the library.

* Even `import theano` is also slow because `import` apparently does a lot of stuffs. Also, after `import theano`, you are stuck with a pre-configured device (e.g. GPU0).
* Hard to improve and contribute as developers. The programming model is hacky: the whole code base is Python where C/CUDA code is packaged as Python string ([example here](https://github.com/Theano/Theano/blob/master/theano/tensor/nnet/conv.py#L1615)).

##Torch
<img src="http://blog.johnassael.com/wp-content/uploads/2015/02/Screen-Shot-2015-02-23-at-05.59.09.png" width="200">
###Strengths
* [Fastest among the mix](https://github.com/soumith/convnet-benchmarks) on convolutions
* *#minor* <br/> Have the [most comprehensive set of convolutions](https://github.com/torch/nn/blob/master/doc/convolution.md). It's worth to note that temporal convolution can be done in TensorFlow/Theano via `conv2d` but that's a trick. The native interface for temporal convolution  in Torch makes it slightly more intuitive to use. 

###Weaknesses
* No Python interface!
* *#minor #highly-subjective* <br/> Like Caffe, Torch uses the layerwise approach instead of the mathematical graph approach of Theano/TensorFlow. This means that a network in Torch is a graph of layers while a network in Theano is a graph of mathematical functions (e.g. matrix add/multiply/etc.). Since a layer is just a function composition, Theano gives more flexibility. [3]

___

###End Notes###
[1] See my [blog post](http://www.kentran.net/2014/12/challenges-in-machine-learning-practice.html) for why this is desirable.

[2] Any framework that uses the symbolic tensor flow graph model must have automatic differentiation. That's why I don't emphasize the auto-diff part.

[3] To be fair, the comparison between layerwise design and tensor graph design is similar to the comparison between C# and C++. Both have their merits. However, for Theano and TensorFlow, it's fairly easy to add popular layer types as simple Python functions. Then, you would have both the flexibility of the native Theano and the convenience of Torch for those layers.