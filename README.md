# Graph Neural Networks
This is a PyTorch library to implement graph neural networks and graph recurrent neural networks. Any questions, comments or suggestions, please e-mail Fernando Gama at fgama@seas.upenn.edu and/or Luana Ruiz at rubruiz@seas.upenn.edu. An in-depth tutorial on a source localization example can be found [here](tutorial.ipynb).

* [Introduction](#introduction)
* [Code](#code)
  * [Dependencies](#dependencies)
  * [Datasets](#datasets)
  * [Libraries](#libraries)
  * [Architectures](#architectures)
  * [Examples](#examples) ([tutorial](tutorial.ipynb))
* [Version](#version)

Whenever using any part of this code, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Network Architectures for Signals Supported on Graphs](http://ieeexplore.ieee.org/document/8579589)," _IEEE Trans. Signal Process._, vol. 67, no. 4, pp. 1034–1049, Feb. 2019.

We note that some specific [architectures](#architectures) have specific paper citation to adequately acknowledge the respective contributors.

Other papers on GNNs by the authors are

E. Isufi, F. Gama, and A. Ribeiro, "[EdgeNets: Edge Varying Graph Neural Networks](http://arxiv.org/abs/2001.07620)," submitted to _IEEE Trans. Pattern Analysis and Mach. Intell._

F. Gama, E. Isufi, G. Leus, and A. Ribeiro, "[Graphs, Convolutions, and Neural Networks](http://arxiv.org/abs/2003.03777)," submitted to _IEEE Signal Process. Mag._

L. Ruiz, F. Gama, and A. Ribeiro, "[Gated Graph Recurrent Neural Networks](http://arxiv.org/abs/2002.01038)," submitted to _IEEE Trans. Signal Process._

F. Gama, J. Bruna, and A. Ribeiro, "[Stability Properties of Graph Neural Networks](http://arxiv.org/abs/1905.04497)," submitted to _IEEE Trans. Signal Process._

F. Gama, E. Tolstaya, and A. Ribeiro, "[Graph Neural Networks for Decentralized Controllers](http://arxiv.org/abs/2003.10280)," _arXiv:2003.10280v1 [cs.LG],_ 23 March 2020.

L. Ruiz, F. Gama, A. G. Marques, and A. Ribeiro, "[Invariance-Preserving Localized Activation Functions for Graph Neural Networks](https://ieeexplore.ieee.org/document/8911416)," _IEEE Trans. Signal Process._, vol. 68, no. 1, pp. 127-141, Jan. 2020.

F. Gama, J. Bruna, and A. Ribeiro, "[Stability of Graph Scattering Transforms](http://arxiv.org/abs/1906.04784)," in _33rd Conf. Neural Inform. Process. Syst._ Vancouver, BC: Neural Inform. Process. Syst. Foundation, 8-14 Dec. 2019.

F. Gama, A. G. Marques, A. Ribeiro, and G. Leus, "[MIMO Graph Filters for Convolutional Networks](http://ieeexplore.ieee.org/document/8445934)," in _19th IEEE Int. Workshop Signal Process. Advances in Wireless Commun._ Kalamata, Greece: IEEE, 25-28 June 2018, pp. 1–5.

F. Gama, G. Leus, A. G. Marques, and A. Ribeiro, "[Convolutional Neural Networks via Node-Varying Graph Filters](https://ieeexplore.ieee.org/document/8439899)," in _2018 IEEE Data Sci. Workshop._ Lausanne, Switzerland: IEEE, 4-6 June 2018, pp. 220–224.


## Introduction <a class="anchor" id="introduction"></a>

We consider data supported by an underlying graph with _N_ nodes. We describe the graph in terms of an _N x N_ matrix _S_ that respects the sparsity of the graph. That is, the element _(i,j)_ of matrix _S_ can be nonzero, if and only if, _i=j_ or _(j,i)_ is an edge of the graph. Examples of such matrices are the adjacency matrix, the graph Laplacian, the Markov matrix, and many normalized counterparts. In general, we refer to this matrix _S_ as the __graph shift operator__ (GSO). This code supports extension to a tensor GSO whenever we want to assign a vector weight to each edge, instead of a scalar weight.

To describe the _N_-dimensional data _x_ as supported by the graph, we assume that each element of _x_ represents the data value at each node, i.e. the _i_-th element _[x]<sub>i</sub> = x<sub>i</sub>_ represents the data value at node _i_. We thus refer to _x_ as a __graph signal__. To effectively relate the graph signal _x_ (which is an _N_-dimensional vector) to the underlying graph support, we use the GSO matrix _S_. In fact, the linear operation _Sx_ represents an exchange of information with neighbors. When computing _Sx_, each node interacts with its one-hop neighbors and computes a weighted average of the information in these neighbors. More precisely, if we denote by _N<sub>i</sub>_ the set of neighbors of node _i_, we see that the output of the operation _Sx_ at node _i_ is given by

<img src="https://latex.codecogs.com/gif.latex?%5B%5Cmathbf%7BS%7D%5Cmathbf%7Bx%7D%5D_%7Bi%7D%20%3D%20%5Csum_%7Bj%3D1%7D%5E%7BN%7D%20%5B%5Cmathbf%7BS%7D%5D_%7Bij%7D%20%5B%5Cmathbf%7Bx%7D%5D_%7Bj%7D%20%3D%20%5Csum_%7Bj%20%5Cin%20%5CccalN_%7Bi%7D%7D%20s_%7Bij%7D%20x_%7Bj%7D">

due to the sparsity pattern of the matrix _S_ where the only nonzero elements are those where there is an edge _(j,i)_ connecting the nodes. We note that the use of the GSO allows for a very simple and straightforward way of explicitly relating the information between different nodes, following the support specified by the given graph. We can extend the descriptive power of graph signals by assining an _F_-dimensional vector to each node, instead of a simple scalar. Each element _f_ of this vector is refered to as __feature__. Then, the data can be thought of as a collection of _F_ graph signals _x<sup>f</sup>_, for each _f=1,...,F_, where each graph signal _x<sup>f</sup>_ represents the value of the specific feature _f_ across all nodes. Describing the data as a collection of _F_ graph signals, as opposed to a collection of _N_ vectors of dimension _F_, allows us to exploit the GSO to easily relate the data with the underlying graph support (as discussed for the case of a scalar graph signal).

A graph neural network is an information processing architecture that regularizes the linear transform of neural networks to take into account the support of the graph. In its most general description, we assume that we have a cascade of _L_ layers, where each layer _l_ takes as input the previous signal, which is a graph signal described by _F<sub>l-1</sub>_ features, and process it through a bank of _F<sub>l</sub> F<sub>l-1</sub>_ linear operations that exploit the graph structure _S_ to obtain _F<sub>l</sub>_ output features, which are processed by an activation function _&sigma;<sub>l</sub>_. Namely, for layer _l_, the output is computed as

<img src="https://latex.codecogs.com/gif.latex?%5Cmathbf%7Bx%7D_%7B%5Cell%7D%5E%7Bf%7D%20%3D%20%5Csigma_%7B%5Cell%7D%20%5Cleft%28%5Csum_%7Bg%3D1%7D%5E%7BF_%7B%5Cell-1%7D%7D%20%5Cmathbf%7BH%7D_%7B%5Cell%7D%5E%7Bfg%7D%28%5Cmathbf%7BS%7D%29%20%5Cmathbf%7Bx%7D_%7B%5Cell-1%7D%5E%7Bg%7D%20%5Cright%29">

where the linear operators _H<sub>l</sub><sup>fg</sup>(S)_ represent __graph filters__ which are linear transforms that exploit the underlying graph structure (typically, by means of local exchanges only, and access to partial information). There are several choices of graph filters that give rise to different architectures (the most popular choice being the linear shift-invariant graph filters, which give rise to __graph convolutions__), many of which can be found in the ensuing library. The operation of pooling, and the extension of the activation functions to include local neighborhoods, can also be found in this library.

## Code <a class="anchor" id="code"></a>

The library is written in [Python3](http://www.python.org/), drawing heavily from [numpy](http://www.numpy.org/), and with neural network models that are defined and trained within the [PyTorch](http://pytorch.org/) framework.

### Dependencies <a class="anchor" id="dependencies"></a>

The required packages are <code>os</code>, <code>numpy</code>, <code>matplotlib</code>, <code>pickle</code>, <code>datetime</code>, <code>scipy.io</code>, <code>copy</code>, <code>torch</code>, <code>scipy</code>, <code>math</code>, and <code>sklearn</code>. Additionally, to handle specific datasets listed below, the following are also required <code>hdf5storage</code>, <code>urllib</code>, <code>zipfile</code>, <code>gzip</code> and <code>shutil</code>; and to handle tensorboard visualization, also include <code>glob</code>, <code>torchvision</code>, <code>operator</code> and <code>tensorboardX</code>.

### Datasets <a class="anchor" id="datasets"></a>

The different datasets involved graph data that are available in this library are the following ones.

<p>1. Authorship attribution dataset, available under <code>datasets/authorshipData</code> (note that the available .rar files have to be uncompressed into the <code>authorshipData.mat</code> to be able to use that dataset with the provided code). When using this dataset, please cite</p>

S. Segarra, M. Eisen, and A. Ribeiro, "[Authorship attribution through function word adjacency networks](http://ieeexplore.ieee.org/document/6638728)," _IEEE Trans. Signal Process._, vol. 63, no. 20, pp. 5464–5478, Oct. 2015.

<p>2. The <a href="http://grouplens.org/datasets/movielens/100k/">MovieLens-100k</a> dataset. When using this dataset, please cite</p>

F. M. Harper and J. A. Konstan, "[The MovieLens datasets: History and Context](http://dl.acm.org/citation.cfm?id=2827872)", _ACM Trans. Interactive Intell. Syst._, vol. 5, no. 4, pp. 19:(1-19), Jan. 2016.

<p>3. A source localization dataset. This source localization problem generates synthetic data at execution time. This data can be generated on synthetic graphs such as the <a href="http://www.nature.com/articles/30918">Small World</a> graph or the <a href="http://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.066106">Stochastic Block Model</a>. It can also generate synthetic data, on a real <a href="http://snap.stanford.edu/data/ego-Facebook.html">Facebook graph</a>. When using the Facebook graph, please cite</p>

J. McAuley and J. Leskovec, "[Learning to discover social circles in Ego networks](http://papers.nips.cc/paper/4532-learning-to-discover-social-circles-in-ego-networks)," in _26th Neural Inform. Process. Syst._ Stateline, TX: NeurIPS Foundation, 3-8 Dec. 2012.

<p>4. A flocking dataset. The problem of flocking consists on controlling a robot swarm, initially flying at random, arbitrary velocities, to fly together at the same velocity while avoiding collisions with each other. The task is to do so in a distributed and decentralized manner, where each agent (each robot) can compute its control action at every time instat relying only on information obtained from communications with immediate neighbors. The dataset is synthetic in that it generates different sample trajectories with random initializations. When using this dataset, please cite

F. Gama, E. Tolstaya, and A. Ribeiro, "[Graph Neural Networks for Decentralized Controllers](http://arxiv.org/abs/2003.10280)," _arXiv:2003.10280v1 [cs.LG],_ 23 March 2020.

<p>4. An epidemic dataset. In this problem, we track the spread of an epidemic on a high school friendship network. The epidemic data is generated by using the SIR model to simulate the spread of an infectious disease on the friendship network built from this <a href="http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/">SocioPatterns dataset</a>. When using this dataset, please cite

L. Ruiz, F. Gama, and A. Ribeiro, "[Gated Graph Recurrent Neural Networks](http://arxiv.org/abs/2002.01038)," submitted to _IEEE Trans. Signal Process._


### Libraries <a class="anchor" id="libraries"></a>

The `alelab` package is split up into two sub-package: `alelab.modules` and `alelab.utils`.

* <code>modules.architectures</code> contains the implementation of several standard architectures (as <code>nn.Module</code> subclasses) so that they can be readily initialized and trained. Details are provided in the [next section](#architectures).

* <code>modules.architecturesTime</code> contains the implementation of several standard architectures (as <code>nn.Module</code> subclasses) that handle time-dependent topologies, so that they can be readily initialized and trained. Details are provided in the [next section](#architectures).

* <code>modules.evaluation</code> contains functions that act as intermediaries between the model and the data in order to evaluate a trained architecture.

* <code>modules.loss</code> contains a wrapper for the loss function so that it can adapt to multiple scenarios, and the loss function for the F1 score.

* <code>modules.model</code> defines a <code>Model</code> that binds together the three basic elements to construct a machine learning model: the (neural network) architecture, the loss function and the optimizer. Additionally, it assigns a training handler and an evaluator. It assigns a name to the model and a directory where to save the trained parameters of the architecture, as well. It is the basic class that can train and evaluate a model and also offers methods to save and load parameters.

* <code>modules.training</code> contains classes that handle the training of each model, acting as an intermediary between the data and the specific architecture within the model being trained.

* <code>utils.dataTools</code> loads each of the datasets described [above](#datasets) as classes with several functionalities particular to each dataset. All the data classes do have two methods: <code>.getSamples</code> to gather the corresponding samples to training, validation or testing sets, and <code>.evaluate</code> that compute the corresponding evaluation measure.

* <code>utils.graphML</code> is the main library containing the implementation of all the possible graph neural network layers (as <code>nn.Module</code> subclasses). This library is the analogous of the <code>torch.nn</code> layer, but for graph-based operations. It contains the definition of the basic layers that need to be put together to build a graph neural network. Details are provided in the [next section](#architectures).

* <code>utils.graphTools</code> defines the <code>Graph</code> class that handles graph-structure information, and offers several other tools to handle graphs.

* <code>utils.miscTools</code> defines some miscellaneous functions.

* <code>utils.visualTools</code> contains all the relevant classes and functions to handle visualization in tensorboard.

### Architectures <a class="anchor" id="architectures"></a>

In what follows, we describe several ways of parameterizing the filters _H<sub>l</sub><sup>fg</sup>(S)_ that are implemented in this library.

* ___Convolutional Graph Neural Networks (via Selection)___. The most popular graph neural network (GNN) is that one that parameterizes _H<sub>l</sub><sup>fg</sup>(S)_ by a linear shift-invariant graph filter, giving rise to a __graph convolution__. The <code>nn.Module</code> subclass that implements the graph filter (convolutional) layer can be found in <code>utils.graphML.GraphFilter</code>. This layer is the basic linear layer in the Selection GNN architecture (which also adds the pointwise activation function and the zero-padding pooling operation), which is already implemented in <code>modules.architectures.SelectionGNN</code> and shown in several <a href="#examples">examples</a>. For more details on this graph convolutional layer or its architecture, and whenever using it, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Network Architectures for Signals Supported on Graphs](http://ieeexplore.ieee.org/document/8579589)," _IEEE Trans. Signal Process._, vol. 67, no. 4, pp. 1034–1049, Feb. 2019.

The <code>modules.architectures.SelectionGNN</code> also has a flag called <code>coarsening</code> that allows for the pooling to be done in terms of graph coarsening, following the Graclus algorithm. This part of the code was mainly adapted to PyTorch from <a href="http://github.com/mdeff/cnn_graph">this repository</a>. For more details on graph coarsening, and whenever using the <code>SelectionGNN</code> with graph coarsening pooling, please cite the following [paper](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf). Also note that by setting the number of filter taps (<code>nFilterTaps</code>) to <code>2</code> on every layer leads to this [architecture](http://openreview.net/forum?id=SJU4ayYgl). Finally, this other [architecture](https://openreview.net/forum?id=ryGs6iA5Km) is obtained by setting the number of filter taps to <code>1</code> for each number of designed fully-connected layers, and then setting it to <code>2</code> to complete the corresponding _GIN layer_. There is one further implementation that is entirely local (i.e. it only involves operations exchanging information with one-hop neighbors). This implementation essentially replaces the last fully-connected layer by a readout layer that only operates on the features obtained at the node. The implementation is dubbed <code>LocalGNN</code> and is used in the <code>MovieLens</code> example.

* ___Convolutional Graph Neural Networks (via Spectrum)___. The spectral GNN is an early implementation of the convolutional GNN in the graph frequency domain. It does not scale to large graphs due to the cost of the eigendecomposition of the GSO. The spectral filtering layer is implemented as a <code>nn.Module</code> subclass in <code>utils.graphML.SpectralGF</code> and the corresponding architecture with these linear layers, together with pointwise nonlinearities is implemented in <code>modules.architectures.SpectralGNN</code>. For more details on the spectral graph filtering layer or its architecture, and whenever using it, please cite

J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, "[Spectral networks and deep locally connected networks on graphs](http://openreview.net/forum?id=DQNsQf-UsoDBa)," in _Int. Conf. Learning Representations 2014_. Banff, AB: Assoc. Comput. Linguistics, 14-16 Apr. 2014, pp. 1–14.

* ___Convolutional Graph Neural Networks (via Aggregation)___. An alternative way to implementing a graph convolution is by means of building an aggregation sequence on each node. Instead of thinking of the graph signal as being diffused through the graph and each diffusion being weighed separately (as is the case of a GCNN via Selection), we think of the signal as being aggregated at each node, by means of successive communications with the one-hop neighbors, and each communication is being weighed by a separate filter tap. The key point is that these aggregation sequences exhibit a regular structure that simultaneously take into account the underlying graph support, since each contiguous element in the sequence represents a contiguous neighborhood. Once we have a regular sequence we can go ahead and apply a regular CNN to process its information. This idea is called an Aggregation GNN and is implemented in <code>modules.architectures.AggregationGNN</code>, since it relies on regular convolution and pooling already defined on <code>torch.nn</code>. A more sophisticated and powerful variant of the Aggregation GNN, called the __Multi-Node Aggregation GNN__ is also available on <code>modules.architectures.MultiNodeAggregationGNN</code>. For more details on the Aggregation GNN, and whenever using it, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Network Architectures for Signals Supported on Graphs](http://ieeexplore.ieee.org/document/8579589)," _IEEE Trans. Signal Process._, vol. 67, no. 4, pp. 1034–1049, Feb. 2019.

* ___Node-Variant Graph Neural Networks___. Parameterizing _H<sub>l</sub><sup>fg</sup>(S)_ with a node-variant graph filter (as opposed to a shift-invariant graph filter), a non-convolutional graph neural network architecture can be built. A node-variant graph filter, essentially lets each node learn its own weight for each neighborhood information. In order to allow this architecture to scale (so that the number of learnable parameters does not depend on the size of the graph), we offer a hybrid node-variant GNN approach as well. The graph filtering layer using node-variant graph filters is defined in <code>utils.graphML.NodeVariantGF</code> and an example of an architecture using these filters for the linear operation, combined with pointwise activation functions and zero-padding pooling, is available in <code>modules.architectures.NodeVariantGNN</code>. For more details on node-variant GNNs, and whenever using these filters or architecture, please cite the following paper

E. Isufi, F. Gama, and A. Ribeiro, "[EdgeNets: Edge Varying Graph Neural Networks](http://arxiv.org/abs/2001.07620)," submitted to _IEEE Trans. Pattern Analysis and Mach. Intell._

* ___ARMA Graph Neural Networks___. A convolutional architecture that is very flexible and with enlarged descriptive power. It replaces the graph convolution with a FIR filter (i.e. the use of a polynomial of the shift operator) by an ratio of polynomials. This architecture offers a good trade-off between number of paramters and selectivity of learnable filters. The edge-variant graph filter layer can be found in <code>utils.graphML.EdgeVariantGF</code>. An example of an architecture with ARMA graph filters as the linear layer, and pointwise activation functions and zero-padding pooling is available in <code>modules.architectures.ARMAfilterGNN</code>. A <code>Local</code> version of this architecture is also available. For more details on ARMA GNNs, and whenever using these filters or architecture, please cite the following paper

E. Isufi, F. Gama, and A. Ribeiro, "[EdgeNets: Edge Varying Graph Neural Networks](http://arxiv.org/abs/2001.07620)," submitted to _IEEE Trans. Pattern Analysis and Mach. Intell._

* ___Edge-Variant Graph Neural Networks___. The most general parameterization that we can make of a linear operation that also takes into account the underlying graph support, is to let each node weigh each of their neighbors' information differently. This is achieved by means of an edge-variant graph filter. Certainly, the edge-variant graph filter has a number of parameters that scales with the number of edges, so a hybrid approach is available. The edge-variant graph filter layer can be found in <code>utils.graphML.GraphFilterARMA</code>. An example of an architecture with edge-variant graph filters as the linear layer, and pointwise activation functions and zero-padding pooling is available in <code>modules.architectures.EdgeVariantGNN</code>. A <code>Local</code> version of this architecture is also available. For more details on edge-variant GNNs, and whenever using these filters or architecture, please cite the following paper

E. Isufi, F. Gama, and A. Ribeiro, "[EdgeNets: Edge Varying Graph Neural Networks](http://arxiv.org/abs/2001.07620)," submitted to _IEEE Trans. Pattern Analysis and Mach. Intell._

* ___Graph Attention Networks___. A particular case of edge-variant graph filters (that predates the use of more general edge-variant filters) and that has been shown to be successful is the graph attention network (commonly known as GAT). The original implementation of GATs can be found in <a href="http://github.com/PetarV-/GAT">this repository</a>. In this library, we offer a PyTorch adaptation of this code (which was originally written for TensorFlow). The GAT parameterizes the edge-variant graph filter by taking into account both the graph support and the data, yielding an architecture with a number of parameters that is independent of the size of the graph. The graph attentional layer can be found in <code>utils.graphML.GraphAttentional</code>, and an example of this architecture in <code>modules.architectures.GraphAttentionNetwork</code>. For more details on GATs, and whenever using this code, please cite the following paper

P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "[Graph Attention Networks](http://openreview.net/forum?id=rJXMpikCZ)," in _6th Int. Conf. Learning Representations_. Vancouver, BC: Assoc. Comput. Linguistics, 30 Apr.-3 May 2018, pp. 1–12.

* ___Local Activation Functions___. Local activation functions exploit the irregular neighborhoods that are inherent to arbitrary graphs. Instead of just applying a pointwise (node-wise) activation function, using a local activation function that carries out a nonlinear operation within a neighborhood has been shown to be effective as well. The corresponding architecture is named <code>LocalActivationGNN</code> and is available under <code>modules/architectures.py</code>. In particular, in this code, the __median activation function__ is implemented in <code>utils.graphML.MedianLocalActivation</code> and the __max activation function__ is implemented in <code>utils.graphML.MaxLocalActivation</code>. For more details on local activation function, and whenever using these operational layers, please cite the following papers

L. Ruiz, F. Gama, A. G. Marques, and A. Ribeiro, "[Invariance-Preserving Localized Activation Functions for Graph Neural Networks](https://ieeexplore.ieee.org/document/8911416)," _IEEE Trans. Signal Process._, vol. 68, no. 1, pp. 127-141, Jan. 2020.

* ___Time-Varying Architectures___. The Selection and Aggregation GNNs have a version adapted to handling time-varying graph signals as well as time-varying shift operators, acting with a unit-delay between communication with neighbors. These architectures can be found in <code>architecturesTime.LocalGNN_DB</code> and <code>architecturesTime.AggregationGNN_DB</code>. For more details on these architectures, please see (and if use, please cite)

F. Gama, E. Tolstaya, and A. Ribeiro, "[Graph Neural Networks for Decentralized Controllers](http://arxiv.org/abs/2003.10280)," _arXiv:2003.10280v1 [cs.LG],_ 23 March 2020.

E. Tolstaya, F. Gama, J. Paulos, G. Pappas, V. Kumar, and A. Ribeiro, "[Learning Decentralized COntrollers for Robot Swarms with Graph Neural Networks](http://arxiv.org/abs/1903.10527)," in _Conf. Robot Learning 2019._ Osaka, Japan: Int. Found. Robotics Res., 30 Oct.-1 Nov. 2019.

* ___Graph Recurrent Neural Networks___. A graph RNN approximates a time-varying graph process with a hidden Markov model, where the hidden state is learned from data. In a graph RNN all linear transforms involved are graph filters that respect the graph. This is a highly flexible architecture that exploits the graph structure as well as the time-dependencies present in data. For static graphs, the architecture can be found in <code>architectures.GraphRecurrentNN</code>, and in <code>architectures.GatedGraphRecurrentNN</code> for time, node and edge gated variations. For time-varying graphs, the architecture is <code>architecturesTime.GraphRecurrentNN_DB</code>. For more details please see, and when using this architecture please cite,

L. Ruiz, F. Gama, and A. Ribeiro, "[Gated Graph Recurrent Neural Networks](http://arxiv.org/abs/2002.01038)," submitted to _IEEE Trans. Signal Process._

### Examples <a class="anchor" id="examples"/>

We have included an in-depth [tutorial](tutorial.ipynb) <code>tutorial.ipynb</code> on a [Jupyter Notebook](http://jupyter.org/). We have also included other examples involving all the four datasets presented [above](#datasets), with examples of all the architectures [just](#architectures) discussed.

* [Tutorial](tutorial.ipynb): <code>tutorial.ipynb</code>. The tutorial covers the basic mathematical formulation for the graph neural networks, and considers a small synthetic problem of source localization. It implements the Aggregation and Selection GNN (both zero-padding and graph coarsening). This tutorial explain, in-depth, all the elements intervening in the setup, training and evaluation of the models, that serves as skeleton for all the other examples.

* [Source Localization](examples/sourceLocGNN.py): <code>sourceLocGNN.py</code>. This example deals with the source localization problem on a 100-node, 5-community random-generated SBM graph. It can consider multiple graph and data realizations to account for randomness in data generation. Implementations of Selection and Aggregation GNNs with different node sampling criteria are presented.

* [MovieLens](examples/movieGNN.py): <code>movieGNN.py</code>. This example has the objective of predicting the rating some user would give to a movie, based on the movies it has ranked before (following the <a href="http://grouplens.org/datasets/movielens/100k/">MovieLens-100k</a> dataset). In this case we present a one- and two-layer Selection GNN with no-padding and the one- and two-layer local implementation available at <code>LocalGNN</code>.

* [Authorship Attribution](examples/authorshipGNN.py): <code>authorshipGNN.py</code>. This example addresses the problem of authorship attribution, by which a text has to be assigned to some author according to their styolmetric signature (based on the underlying word adjacency network; details <a href="http://ieeexplore.ieee.org/document/6638728">here</a>). In this case, we test different local activation functions (median, max, and pointwise).

* [Flocking](examples/flockingGNN.py): <code>flockingGNN.py</code>. This is an example of controlling a robot swarm to fly together at the same velocity while avoiding collisions. It is a synthetic dataset where time-dependent architectures can be tested. In particular, we test the use of a linear filter, a Local GNN, an Aggregation GNN and a GRNN, considering, not only samples of the form (S_t, x_t), for each t, but also delayed communications where the information observed from further away neighbors is actually delayed.

* [Epidemic Tracking](examples/epidemicGRNN.py): <code>epidemicGRNN.py</code>. In this example, we compare GRNNs and gated GRNNs in a binary node classification problem modeling the spread of an epidemic on a high school friendship network. The disease is first recorded on day t=0, when each individual node is infected with probability p_seed=0.05. On the days that follow, an infected student can then spread the disease to their susceptible friends with probability p_inf=0.3 each day. Infected students become immune after 4 days, at which point they can no longer spread or contract the disease. Given the state of each node at some point in time (susceptible, infected or recovered), the binary node classification problem is to predict whether each node in the network will have the disease (i.e., be infected) 8 days ahead.

## Version <a class="anchor" id="version"></a>

* ___0.4 (March 5, 2021):___ Added the main file for the epidemic tracking experiment, <code>epidemicGRNN.py</code>. Added the edge list from which the graph used in this experiment is built. <code>dataTools.py</code> now has an <code>Epidemics</code> class which handles the abovementioned graph and the epidemic data. <code>loss.py</code> now has a new loss function, which computes the loss corresponding to the F1 score (1-F1 score). <code>graphML.py</code> now has the functional <code>GatedGRNN</code> and the layers <code>HiddenState</code>, <code>TimeGatedHiddenState</code>, <code>NodeGatedHiddenState</code>, <code>EdgeGatedHiddenState</code>, which are used to calculate the hidden state of (gated) GRNNs. <code>architectures.py</code> now has the architectures <code>GraphRecurrentNN</code> and <code>GatedGraphRecurrentNN</code>.

* ___0.3 (May 2, 2020):___ Added the time-dependent architectures that handle (graph, graph signal) batch data as well as delayed communications. These architectures can be found in <code>architecturesTime.py</code>. A new synthetic dataset has also been added, namely, that used in the Flocking problem. Made the <code>Model</code> class to be the central handler of all the machine learning model. Training multiple models has been dropped in favor of training through the method offered in the <code>Model</code> class. Trainers and evaluators had to been added to be effective intermediaries between the architectures and the data, especially in problems that are not classification ones (i.e. regression -interpolation- in the movie recommendation setting, and imitation learning in the flocking problem). This should give flexibility to carry over these architectures to new problems, as well as make prototyping easier since training and evaluating has been greatly simplified. Minor modifications and eventual bug fixes have been made here and there.

* ___0.2 (Dec 16, 2019):___ Added new architecture: <code>LocalActivationGNN</code> and <code>LocalGNN</code>. Added new loss module to handle the logic that gives flexibility to the loss function. Moved the ordering from external to the architecture, to internal to it. Added two new methods: <code>.splitForward()</code> and <code>.changeGSO()</code> to separate the output from the graph layers and the MLP, and to change the GSO from training to test time, respectively. Class <code>Model</code> does not keep track of the order anymore. Got rid of <code>MATLAB(R)</code> support. Better memory management (do not move the entire dataset to memory, only the batch). Created methods to normalize dat aand change data type. Deleted the 20News dataset which is not supported anymore. Added the method <code>.expandDims()</code> to the <code>data</code> for increased flexibility. Changed the evaluate method so that it is always a decreasing function. Totally revamped the <code>MovieLens</code> class. Corrected a bug on the <code>computeNeighborhood()</code> function (thanks to Bianca Iancu, A (dot) Iancu-1 (at) student (dot) tudelft (dot) nl and Gabriele Mazzola, G (dot) Mazzola (at) student (dot) tudelft (dot) nl for spotting it). Corrected bugs on device handling of local activation functions. Updated tutorial.

* ___0.1 (Jul 12, 2019):___ First released (beta) version of this graph neural network library. Includes the basic convolutional graph neural networks (selection -zero-padding and graph coarsening-, spectral, aggregation), and some non-convolutional graph neural networks as well (node-variant, edge-variant and graph attention networks). It also inlcudes local activation functions (max and median). In terms of examples, it considers the source localization problem (both in the tutorial and in a separate example), the movie recommendation problem, the authorship attribution problem and the text categorization problems. In terms of structure, it sets the basis for data handling and training of multiple models.
