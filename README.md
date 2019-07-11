# Graph Neural Networks
This is a PyTorch library to implement graph neural networks. Any questions, comments or suggestions, please e-mail Fernando Gama at fgama@seas.upenn.edu.

* [Introduction](#introduction)
* [Code](#code)
  * [Dependencies](#dependencies)
  * [Datasets](#datasets)
  * [Libraries](#libraries)
  * [Architectures](#architectures)
  * [Examples](#examples)
* [Version](#version)

## Introduction <a class="anchor" id="introduction"></a>

We consider data supported by an underlying graph with _N_ nodes. We describe the graph in terms of an _N x N_ matrix _S_ that respects the sparsity of the graph. That is, the element _(i,j)_ of matrix _S_ can be nonzero, if and only if, _i=j_ or _(j,i)_ is an edge of the graph. Examples of such matrices are the adjacency matrix, the graph Laplacian, the Markov matrix, and many normalized counterparts. In general, we refer to this matrix _S_ as the __graph shift operator__ (GSO). This code supports extension to a tensor GSO whenever we want to assign a vector weight to each edge, instead of a scalar weight.

To describe the _N_-dimensional data _x_ as supported by the graph, we assume that each element of _x_ represents the data value at each node, i.e. the _i_-th element _[x]<sub>i</sub> = x<sub>i</sub>_ represents the data value at node _i_. We thus refer to _x_ as a __graph signal__. To effectively relate the graph signal _x_ (which is an _N_-dimensional vector) to the underlying graph support, we use the GSO matrix _S_. In fact, the linear operation _Sx_ represents an exchange of information with neighbors. When computing _Sx_, each node interacts with its one-hop neighbors and computes a weighted average of the information in these neighbors. More precisely, if we denote by _N<sub>i</sub>_ the set of neighbors of node _i_, we see that the output of the operation _Sx_ at node _i_ is given by

<img src="http://www.sciweavers.org/upload/Tex2Img_1562846744/render.png">

due to the sparsity pattern of the matrix _S_ where the only nonzero elements are those where there is an edge _(j,i)_ connecting the nodes. We note that the use of the GSO allows for a very simple and straightforward way of explicitly relating the information between different nodes, following the support specified by the given graph. We can extend the descriptive power of graph signals by assining an _F_-dimensional vector to each node, instead of a simple scalar. Each element _f_ of this vector is refered to as __feature__. Then, the data can be thought of as a collection of _F_ graph signals _x<sup>f</sup>_, for each _f=1,...,F_, where each graph signal _x<sup>f</sup>_ represents the value of the specific feature _f_ across all nodes. Describing the data as a collection of _F_ graph signals, as opposed to a collection of _N_ vectors of dimension _F_, allows us to exploit the GSO to easily relate the data with the underlying graph support (as discussed for the case of a scalar graph signal).

A graph neural network is an information processing architecture that regularizes the linear transform of neural networks to take into account the support of the graph. In its most general description, we assume that we have a cascade of _L_ layers, where each layer _l_ takes as input the previous signal, which is a graph signal described by _F<sub>l-1</sub>_ features, and process it through a bank of _F<sub>l</sub> F<sub>l-1</sub>_ linear operations that exploit the graph structure _S_ to obtain _F<sub>l</sub>_ output features, which are processed by an activation function _&sigma;<sub>l</sub>_. Namely, for layer _l_, the output is computed as

<img src="http://www.sciweavers.org/upload/Tex2Img_1562847619/render.png">

where the linear operators _H<sub>l</sub><sup>fg</sup>(S)_ represent __graph filters__ which are linear transforms that exploit the underlying graph structure (typically, by means of local exchanges only, and access to partial information). There are several choices of graph filters that give rise to different architectures (the most popular choice being the linear shift-invariant graph filters, which give rise to __graph convolutions__), many of which can be found in the ensuing library. The operation of pooling, and the extension of the activation functions to include local neighborhoods, can also be found in this library.

## Code <a name="code"/>

The library is written in [Python3](http://www.python.org/), drawing heavily from [numpy](http://www.numpy.org/), and with neural network models that are defined and trained within the [PyTorch](http://pytorch.org/) framework.

### Dependencies <a class="anchor" id="dependencies"></a>

The required packages are <code>os</code>, <code>numpy</code>, <code>matplotlib</code>, <code>pickle</code>, <code>datetime</code>, <code>scipy.io</code>, <code>copy</code>, <code>torch</code>, <code>scipy</code>, <code>math</code>, and <code>sklearn</code>. Additionally, to handle specific datasets listed below, the following are also required <code>hdf5storage</code>, <code>urllib</code>, <code>zipfile</code>, <code>gzip</code>, <code>shutil</code>, <code>gensim</code>, and <code>re</code>; and to handle tensorboard visualization, also include <code>glob</code>, <code>torchvision</code>, <code>operator</code> and <code>tensorboardX</code>.

### Datasets <a class="anchor" id="datasets"></a>

The different datasets involved graph data that are available in this library are the following ones.

<p>1. 20NEWS dataset, available through the <code>sklearn.datasets</code> library (credit is due to <a href="http://github.com/mdeff/cnn_graph/blob/master/nips2016/20news.ipynb">M. Defferrard</a> for creating many of the functions that handle this dataset). When using this dataset, please cite</p>

T. Joachims, "Analysis of the Rocchio Algorithm with TFIDF for Text Categorization", in _14th Int. Conf. Mach. Learning_. Nashville, TN, 8-12 July 1997.

<p>2. Authorship attribution dataset, available under <code>datasets/authorshipData</code> (note that the available .rar files have to be uncompressed into the <code>authorshipData.mat</code> to be able to use that dataset with the provided code). When using this dataset, please cite</p>

S. Segarra, M. Eisen, and A. Ribeiro, "[Authorship attribution through function word adjacency networks](https://ieeexplore.ieee.org/document/6638728)," _IEEE Trans. Signal Process._, vol. 63, no. 20, pp. 5464–5478, Oct. 2015.

<p>3. The [MovieLens-100k](https://grouplens.org/datasets/movielens/100k/) dataset. When using this dataset, please cite</p>

F. M. Harper and J. A. Konstan, "[The MovieLens datasets: History and Context](https://dl.acm.org/citation.cfm?id=2827872)", _ACM Trans. Interactive Intell. Syst._, vol. 5, no. 4, pp. 19:(1-19), Jan. 2016.

<p>4. A source localization dataset. This source localization problem generates synthetic data at execution time. This data can be generated on synthetic graphs such as the <a href="https://www.nature.com/articles/30918">Small World</a> graph or the <a href="https://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.066106">Stochastic Block Model</a>. It can also generate synthetic data, on a real <a href="https://snap.stanford.edu/data/ego-Facebook.html">Facebook graph</a>. When using the Facebook graph, please cite</p>

J. McAuley and J. Leskovec, "[Learning to discover social circles in Ego networks](https://papers.nips.cc/paper/4532-learning-to-discover-social-circles-in-ego-networks)," in _26th Neural Inform. Process. Syst._ Stateline, TX: NeurIPS Foundation, 3-8 Dec. 2012.

### Libraries <a class="anchor" id="libraries"></a>

The libraries found here are split into two directories: <code>Modules/</code> and <code>Utils/</code>.

* <code>Modules.architectures</code> contains the implementation of several standard architectures (as <code>nn.Module</code> subclasses) so that they can be readily initialized and trained. Details are provided in the [next section](#architectures).

* <code>Modules.model</code> defines a <code>Model</code> that binds together the three basic elements to construct a machine learning model: the (neural network) architecture, the loss function and the optimizer. It also contains assigns a name to the model and a directory where to save the trained parameters of the architecture. It offers methods to save and load parameters, and even to train and evaluate a model individually.

* <code>Modules.train</code> contains a function that handles the training for several models simulatenously, so that they can be compared under the exact same training conditions.

* <code>Utils.dataTools</code> loads each of the datasets described [above](#datasets) as classes with several functionalities particular to each dataset. All the data classes do have two methods: <code>.getSamples</code> to gather the corresponding samples to training, validation or testing sets, and <code>.evaluate</code> that compute the corresponding evaluation measure.

* <code>Utils.graphML</code> is the main library containing the implementation of all the possible graph neural network layers (as <code>nn.Module</code> subclasses). This library is the analogous of the <code>torch.nn</code> layer, but for graph-based operations. It contains the definition of the basic layers that need to be put together to build a graph neural network. Details are provided in the [next section](#architectures).

* <code>Utils.graphTools</code> defines the <code>Graph</code> class that handles graph-structure information, and offers several other tools to handle graphs.

* <code>Utils.miscTools</code> defines some miscellaneous functions.

* <code>Utils.visualTools</code> contains all the relevant classes and functions to handle visualization in tensorboard.

### Architectures <a class="anchor" id="architectures"></a>

In the library <code>Utils.graphML</code>, we can find several ways of parameterizing the filters _H<sub>l</sub><sup>fg</sup>(S)_.

### Examples <a name="examples"/>

## Version <a class="anchor" id="version"></a>
