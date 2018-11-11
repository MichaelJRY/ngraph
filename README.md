# nGraph Compiler Stack (1.0 Beta) [![Build Status][build-status-badge]][build-status] 


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NervanaSystems/ngraph/blob/master/LICENSE)


<div align="center">
  <h5>
    <a href="https://ngraph.nervanasys.com/docs/latest/project/about.html">
      nGraph architecture and features</a><span> | </span>
    <a href="https://ngraph.nervanasys.com/docs/latest/project/release-notes.html">
      Release notes</a><span> | </span>
    <a href="https://ngraph.nervanasys.com/docs/latest">Documentation</a><span> | </span>
    <a href="#Ecosystem" >nGraph ecosystem</a><span> | </span>
    <a href="#Getting-started-guides" >Getting started guides</a><span> | </span>
    <a href="#How-to-contribute" >How to contribute</a>
 </h5>
</div>


## Getting started guides


|  Framework                 | Installation guide              | Beta notes  
|----------------------------|---------------------------------|-----------------------------------
| TensorFlow*                | [nGraph TensorFlow bridge]      | see [workloads and criterion]
| MXNet*                     | [nGraph-MXNet]                  | see [workloads and criterion] 
| PaddlePaddle*              |        wip                      |   tbd
| [early supporter] of ONNX  | [ngraph-onnx]                   | [Functional] with DenseNet-121, Inception-v1, ResNet-50, Inception-v2, ShuffleNet, SqueezeNet, VGG-19, and 7 more   
| PyTorch*                   | [nGraph for PyTorch developers] |    

Additional work is also being done via [PlaidML].


# Introduction 

C++ API for framework developers and a Python API which can run inference 
on models imported from ONNX have been tested and work with a number of different 
workloads for TensorFlow, MXNet, and . 

See the [Release Notes] for recent changes.


| Backend                                       | current support   | future support |
|-----------------------------------------------|-------------------|----------------|
| Intel® Architecture CPU                       | yes               | yes            |
| Intel® Nervana™ Neural Network Processor (NNP)| yes               | yes            |
| Intel [Movidius™ Myriad™ 2] VPUs              | coming soon       | yes            |
| Intel® Architecture GPUs                      | via PlaidML       | yes            |
| AMD* GPUs                                     | via PlaidML       | yes            |
| NVIDIA* GPUs                                  | via PlaidML       | some           | 
| Field Programmable Gate Arrays (FPGA)         | no                | yes            |


## Ecosystem

![nGraph ecosystem][ngraph-ecosystem]

The **nGraph Compiler** is Intel's graph compiler for Artificial Neural Networks. 
Documentation in this repo describes how you can program any framework 
to run training and inference computations on a variety of Backends including 
Intel® Architecture Processors (CPUs), Intel® Nervana™ Neural Network Processors 
(NNPs), cuDNN-compatible graphics cards (GPUs), custom VPUs like Movidius, and
many others. The default CPU Backend also provides an interactive *Interpreter* 
mode that can be used to zero in on a DL model and create custom nGraph 
optimizations that can be used to further accelerate training or inference, in 
whatever scenario you need. nGraph provides both  a C++ API for framework 
developers and a Python API which can run inference on models imported from 
ONNX. 


## Documentation

See our [build the Library] docs for how to get started.

For this early release, we provide [framework integration guides] to
compile MXNet and TensorFlow-based projects. If you already have a
trained model, we've put together a getting started guide for
[how to import] a deep learning model and start working with the nGraph
APIs.

## Support

Please submit your questions, feature requests and bug reports via
[GitHub issues].

## How to contribute

We welcome community contributions to nGraph. If you have an idea how
to improve the Library:

* See the [contrib guide] for code formatting and style guidelines.
* Share your proposal via [GitHub issues].
* Ensure you can build the product and run all the examples with your patch.
* In the case of a larger feature, create a test.
* Submit a [pull request].
* Make sure your PR passes all CI tests. Note: our [Travis-CI][build-status] service
  runs only on a CPU backend on Linux. We will run additional tests
  in other environments.
* We will review your contribution and, if any additional fixes or
  modifications are necessary, may provide feedback to guide you. When
  accepted, your pull request will be merged to the repository.


[Architecture and features]:https://ngraph.nervanasys.com/docs/latest/project/about.html
[Documentation]: https://ngraph.nervanasys.com/docs/latest
[build the Library]: https://ngraph.nervanasys.com/docs/latest/buildlb.html
[Getting Started Guides]: Getting-started-guides
[How to contribute]: How-to-contribute
[framework integration guides]: http://ngraph.nervanasys.com/docs/latest/framework-integration-guides.html
[release notes]: https://ngraph.nervanasys.com/docs/latest/project/release-notes.html
[Github issues]: https://github.com/NervanaSystems/ngraph/issues
[contrib guide]: https://ngraph.nervanasys.com/docs/latest/project/code-contributor-README.html
[pull request]: https://github.com/NervanaSystems/ngraph/pulls
[how to import]: http://ngraph.nervanasys.com/docs/latest/howto/import.html
[ngraph-ecosystem]: doc/sphinx/source/graphics/599px-Intel-ngraph-ecosystem.png "nGraph Ecosystem"
[build-status]: https://travis-ci.org/NervanaSystems/ngraph/branches
[build-status-badge]: https://travis-ci.org/NervanaSystems/ngraph.svg?branch=master
[develop-without-lockin]: doc/sphinx/source/graphics/develop-without-lockin.png "Develop on any part of the stack wtihout lockin"
[Movidius™ Myriad™ 2]:https://www.movidius.com/solutions/vision-processing-unit
[PlaidML]: https://github.com/plaidml/plaidml
[nGraph TensorFlow bridge]: https://github.com/NervanaSystems/ngraph-tf
[nGraph-MXNet]: https://github.com/NervanaSystems/ngraph-mxnet/blob/master/NGRAPH_README.md
[ngraph-onnx]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/README.md
[early supporter]: https://ai.intel.com/adaptable-deep-learning-solutions-with-ngraph-compiler-and-onnx/
[nGraph for PyTorch developers]: https://ai.intel.com/investing-in-the-pytorch-developer-community
[Functional]: https://github.com/NervanaSystems/ngraph-onnx/blob/master/14-cwo