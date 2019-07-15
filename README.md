# Graph Neural Networks
This is a PyTorch library to implement graph neural networks. Any questions, comments or suggestions, please e-mail Fernando Gama at fgama@seas.upenn.edu and/or Luana Ruiz at rubruiz@seas.upenn.edu. An in-depth tutorial on a source localization example can be found [here](tutorial.ipynb).

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

## Introduction <a class="anchor" id="introduction"></a>

We consider data supported by an underlying graph with _N_ nodes. We describe the graph in terms of an _N x N_ matrix _S_ that respects the sparsity of the graph. That is, the element _(i,j)_ of matrix _S_ can be nonzero, if and only if, _i=j_ or _(j,i)_ is an edge of the graph. Examples of such matrices are the adjacency matrix, the graph Laplacian, the Markov matrix, and many normalized counterparts. In general, we refer to this matrix _S_ as the __graph shift operator__ (GSO). This code supports extension to a tensor GSO whenever we want to assign a vector weight to each edge, instead of a scalar weight.

To describe the _N_-dimensional data _x_ as supported by the graph, we assume that each element of _x_ represents the data value at each node, i.e. the _i_-th element _[x]<sub>i</sub> = x<sub>i</sub>_ represents the data value at node _i_. We thus refer to _x_ as a __graph signal__. To effectively relate the graph signal _x_ (which is an _N_-dimensional vector) to the underlying graph support, we use the GSO matrix _S_. In fact, the linear operation _Sx_ represents an exchange of information with neighbors. When computing _Sx_, each node interacts with its one-hop neighbors and computes a weighted average of the information in these neighbors. More precisely, if we denote by _N<sub>i</sub>_ the set of neighbors of node _i_, we see that the output of the operation _Sx_ at node _i_ is given by

<img src="http://www.sciweavers.org/upload/Tex2Img_1562846744/render.png">

due to the sparsity pattern of the matrix _S_ where the only nonzero elements are those where there is an edge _(j,i)_ connecting the nodes. We note that the use of the GSO allows for a very simple and straightforward way of explicitly relating the information between different nodes, following the support specified by the given graph. We can extend the descriptive power of graph signals by assining an _F_-dimensional vector to each node, instead of a simple scalar. Each element _f_ of this vector is refered to as __feature__. Then, the data can be thought of as a collection of _F_ graph signals _x<sup>f</sup>_, for each _f=1,...,F_, where each graph signal _x<sup>f</sup>_ represents the value of the specific feature _f_ across all nodes. Describing the data as a collection of _F_ graph signals, as opposed to a collection of _N_ vectors of dimension _F_, allows us to exploit the GSO to easily relate the data with the underlying graph support (as discussed for the case of a scalar graph signal).

A graph neural network is an information processing architecture that regularizes the linear transform of neural networks to take into account the support of the graph. In its most general description, we assume that we have a cascade of _L_ layers, where each layer _l_ takes as input the previous signal, which is a graph signal described by _F<sub>l-1</sub>_ features, and process it through a bank of _F<sub>l</sub> F<sub>l-1</sub>_ linear operations that exploit the graph structure _S_ to obtain _F<sub>l</sub>_ output features, which are processed by an activation function _&sigma;<sub>l</sub>_. Namely, for layer _l_, the output is computed as

<img src="http://www.sciweavers.org/upload/Tex2Img_1562847619/render.png">

where the linear operators _H<sub>l</sub><sup>fg</sup>(S)_ represent __graph filters__ which are linear transforms that exploit the underlying graph structure (typically, by means of local exchanges only, and access to partial information). There are several choices of graph filters that give rise to different architectures (the most popular choice being the linear shift-invariant graph filters, which give rise to __graph convolutions__), many of which can be found in the ensuing library. The operation of pooling, and the extension of the activation functions to include local neighborhoods, can also be found in this library.

## Code <a class="anchor" id="code"></a>

The library is written in [Python3](http://www.python.org/), drawing heavily from [numpy](http://www.numpy.org/), and with neural network models that are defined and trained within the [PyTorch](http://pytorch.org/) framework.

### Dependencies <a class="anchor" id="dependencies"></a>

The required packages are <code>os</code>, <code>numpy</code>, <code>matplotlib</code>, <code>pickle</code>, <code>datetime</code>, <code>scipy.io</code>, <code>copy</code>, <code>torch</code>, <code>scipy</code>, <code>math</code>, and <code>sklearn</code>. Additionally, to handle specific datasets listed below, the following are also required <code>hdf5storage</code>, <code>urllib</code>, <code>zipfile</code>, <code>gzip</code>, <code>shutil</code>, <code>gensim</code>, and <code>re</code>; and to handle tensorboard visualization, also include <code>glob</code>, <code>torchvision</code>, <code>operator</code> and <code>tensorboardX</code>.

### Datasets <a class="anchor" id="datasets"></a>

The different datasets involved graph data that are available in this library are the following ones.

<p>1. 20NEWS dataset, available through the <code>sklearn.datasets</code> library (credit is due to <a href="http://github.com/mdeff/cnn_graph/blob/master/nips2016/20news.ipynb">M. Defferrard</a> for creating many of the functions that handle this dataset). When using this dataset, please cite</p>

T. Joachims, "Analysis of the Rocchio Algorithm with TFIDF for Text Categorization", in _14th Int. Conf. Mach. Learning_. Nashville, TN, 8-12 July 1997.

<p>2. Authorship attribution dataset, available under <code>datasets/authorshipData</code> (note that the available .rar files have to be uncompressed into the <code>authorshipData.mat</code> to be able to use that dataset with the provided code). When using this dataset, please cite</p>

S. Segarra, M. Eisen, and A. Ribeiro, "[Authorship attribution through function word adjacency networks](http://ieeexplore.ieee.org/document/6638728)," _IEEE Trans. Signal Process._, vol. 63, no. 20, pp. 5464–5478, Oct. 2015.

<p>3. The <a href="http://grouplens.org/datasets/movielens/100k/">MovieLens-100k</a> dataset. When using this dataset, please cite</p>

F. M. Harper and J. A. Konstan, "[The MovieLens datasets: History and Context](http://dl.acm.org/citation.cfm?id=2827872)", _ACM Trans. Interactive Intell. Syst._, vol. 5, no. 4, pp. 19:(1-19), Jan. 2016.

<p>4. A source localization dataset. This source localization problem generates synthetic data at execution time. This data can be generated on synthetic graphs such as the <a href="http://www.nature.com/articles/30918">Small World</a> graph or the <a href="http://journals.aps.org/pre/abstract/10.1103/PhysRevE.84.066106">Stochastic Block Model</a>. It can also generate synthetic data, on a real <a href="http://snap.stanford.edu/data/ego-Facebook.html">Facebook graph</a>. When using the Facebook graph, please cite</p>

J. McAuley and J. Leskovec, "[Learning to discover social circles in Ego networks](http://papers.nips.cc/paper/4532-learning-to-discover-social-circles-in-ego-networks)," in _26th Neural Inform. Process. Syst._ Stateline, TX: NeurIPS Foundation, 3-8 Dec. 2012.

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

In what follows, we describe several ways of parameterizing the filters _H<sub>l</sub><sup>fg</sup>(S)_ that are implemented in this library.

* ___Convolutional Graph Neural Networks (via Selection)___. The most popular graph neural network (GNN) is that one that parameterizes _H<sub>l</sub><sup>fg</sup>(S)_ by a linear shift-invariant graph filter, giving rise to a __graph convolution__. The <code>nn.Module</code> subclass that implements the graph filter (convolutional) layer can be found in <code>Utils.graphML.GraphFilter</code>. This layer is the basic linear layer in the Selection GNN architecture (which also adds the pointwise activation function and the zero-padding pooling operation), which is already implemented in <code>Modules.architectures.SelectionGNN</code> and shown in several <a href="#examples">examples</a>. For more details on this graph convolutional layer or its architecture, and whenever using it, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Network Architectures for Signals Supported on Graphs](http://ieeexplore.ieee.org/document/8579589)," _IEEE Trans. Signal Process._, vol. 67, no. 4, pp. 1034–1049, Feb. 2019.

The <code>Modules.architectures.SelectionGNN</code> also has a flag called <code>coarsening</code> that allows for the pooling to be done in terms of graph coarsening, following the Graclus algorithm. This part of the code was mainly adapted to PyTorch from <a href="http://github.com/mdeff/cnn_graph">this repository</a>. For more details on graph coarsening, and whenever using the <code>SelectionGNN</code> with graph coarsening pooling, please cite the following paper

M. Defferrard, X. Bresson, and P. Vandergheynst, "[Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](http://papers.nips.cc/paper/6081-convolutional-neural-networks-on-graphs-with-fast-localized-spectral-filtering.pdf),” in _30th Annu. Conf. Neural Inform. Process. Syst._ Barcelona, Spain: Neural Inform. Process. Foundation, 5-10 Dec. 2016, pp. 3844–3858.

* ___Convolutional Graph Neural Networks (via Spectrum)___. The spectral GNN is an early implementation of the convolutional GNN in the graph frequency domain. It does not scale to large graphs due to the cost of the eigendecomposition of the GSO. The spectral filtering layer is implemented as a <code>nn.Module</code> subclass in <code>Utils.graphML.SpectralGF</code> and the corresponding architecture with these linear layers, together with pointwise nonlinearities is implemented in <code>Modules.architectures.SpectralGNN</code>. For more details on the spectral graph filtering layer or its architecture, and whenever using it, please cite

J. Bruna, W. Zaremba, A. Szlam, and Y. LeCun, "[Spectral networks and deep locally connected networks on graphs](http://openreview.net/forum?id=DQNsQf-UsoDBa)," in _Int. Conf. Learning Representations 2014_. Banff, AB: Assoc. Comput. Linguistics, 14-16 Apr. 2014, pp. 1–14.

* ___Convolutional Graph Neural Networks (via Aggregation)___. An alternative way to implementing a graph convolution is by means of building an aggregation sequence on each node. Instead of thinking of the graph signal as being diffused through the graph and each diffusion being weighed separately (as is the case of a GCNN via Selection), we think of the signal as being aggregated at each node, by means of successive communications with the one-hop neighbors, and each communication is being weighed by a separate filter tap. The key point is that these aggregation sequences exhibit a regular structure that simultaneously take into account the underlying graph support, since each contiguous element in the sequence represents a contiguous neighborhood. Once we have a regular sequence we can go ahead and apply a regular CNN to process its information. This idea is called an Aggregation GNN and is implemented in <code>Modules.architectures.AggregationGNN</code>, since it relies on regular convolution and pooling already defined on <code>torch.nn</code>. A more sophisticated and powerful variant of the Aggregation GNN, called the __Multi-Node Aggregation GNN__ is also available on <code>Modules.architectures.MultiNodeAggregationGNN</code>. For more details on the Aggregation GNN, and whenever using it, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Network Architectures for Signals Supported on Graphs](http://ieeexplore.ieee.org/document/8579589)," _IEEE Trans. Signal Process._, vol. 67, no. 4, pp. 1034–1049, Feb. 2019.

* ___Node-Variant Graph Neural Networks___. Parameterizing _H<sub>l</sub><sup>fg</sup>(S)_ with a node-variant graph filter (as opposed to a shift-invariant graph filter), a non-convolutional graph neural network architecture can be built. A node-variant graph filter, essentially lets each node learn its own weight for each neighborhood information. In order to allow this architecture to scale (so that the number of learnable parameters does not depend on the size of the graph), we offer a hybrid node-variant GNN approach as well. The graph filtering layer using node-variant graph filters is defined in <code>Utils.graphML.NodeVariantGF</code> and an example of an architecture using these filters for the linear operation, combined with pointwise activation functions and zero-padding pooling, is available in <code>Modules.architectures.NodeVariantGNN</code>. For more details on node-variant GNNs, and whenever using these filters or architecture, please cite the following paper

F. Gama, A. G. Marques, G. Leus, and A. Ribeiro, "[Convolutional Neural Networks via Node-Varying Graph Filters](http://ieeexplore.ieee.org/document/8439899)," in _2018 IEEE Data Sci. Workshop_. Lausanne, Switzerland: IEEE, 4-6 June 2018, pp. 220-224.

* ___Edge-Variant Graph Neural Networks___. The most general parameterization that we can make of a linear operation that also takes into account the underlying graph support, is to let each node weigh each of their neighbors' information differently. This is achieved by means of an edge-variant graph filter. Certainly, the edge-variant graph filter has a number of parameters that scales with the number of edges, so a hybrid approach is available. The edge-variant graph filter layer cane be found in <code>Utils.graphML.EdgeVariantGF</code>. An example of an architecture with edge-variant graph filters as the linear layer, and pointwise activation functions and zero-padding pooling is available in <code>Modules.architectures.EdgeVariantGNN</code>. For more details on edge-variant GNNs, and whenever using these filters or architecture, please cite the following paper

E. Isufi, F. Gama, and A. Ribeiro, "[Generalizing Graph Convolutional Neural Networks with Edge-Variant Recursions on Graphs](http://arxiv.org/abs/1903.01298)," in _27th Eur. Signal Process. Conf._ A Coruña, Spain: EURASIP, 2-6 Sep. 2019.

* ___Graph Attention Networks___. A particular case of edge-variant graph filters (that predates the use of more general edge-variant filters) and that has been shown to be successful is the graph attention network (commonly known as GAT). The original implementation of GATs can be found in <a href="http://github.com/PetarV-/GAT">this repository</a>. In this library, we offer a PyTorch adaptation of this code (which was originally written for TensorFlow). The GAT parameterizes the edge-variant graph filter by taking into account both the graph support and the data, yielding an architecture with a number of parameters that is independent of the size of the graph. The graph attentional layer can be found in <code>Utils.graphML.GraphAttentional</code>, and an example of this architecture in <code>Modules.architectures.GraphAttentionNetwork</code>. For more details on GATs, and whenever using this code, please cite the following paper

P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, "[Graph Attention Networks](http://openreview.net/forum?id=rJXMpikCZ)," in _6th Int. Conf. Learning Representations_. Vancouver, BC: Assoc. Comput. Linguistics, 30 Apr.-3 May 2018, pp. 1–12.

* ___Local Activation Functions___. Besides all the different implementations of graph-based linear operations that have just been discussed, local activation functions are also implemented to use in the activation function operation within each layer. Local activation functions exploit the irregular neighborhoods that are inherent to arbitrary graphs. Instead of just applying a pointwise (node-wise) activation function, using a local activation function that carries out a nonlinear operation within a neighborhood has been shown to be effective as well. In particular, in this code, the __median activation function__ is implemented in <code>Utils.graphML.MedianLocalActivation</code> and the __max activation function__ is implemented in <code>Utils.graphML.MaxLocalActivation</code>. For more details on local activation function, and whenever using these operational layers, please cite the following papers

L. Ruiz, F. Gama, A. G. Marques, and A. Ribeiro, "[Invariance-Preserving Localized Activation Functions for Graph Neural Networks](http://arxiv.org/abs/1903.12575)," _arXiv:1903.12575v1 \[eess.SP\]_, 29 March 2019. \[Online\]. Available: http://arxiv.org/abs/1903.12575

L. Ruiz, F. Gama, A. G. Marques, and A. Ribeiro, "[Median Activation Functions for Graph Neural Networks](http://ieeexplore.ieee.org/document/8683472)," in _44th IEEE Int. Conf. Acoust., Speech and Signal Process._ Brighton, UK: IEEE, 12-17 May 2019, pp. 7440–7447.

### Examples <a class="anchor" id="examples"/>

We have included an in-depth [tutorial](tutorial.ipynb) <code>tutorial.ipynb</code> on a [Jupyter Notebook](http://jupyter.org/). We have also included other examples involving all the four datasets presented [above](#datasets), with examples of all the architectures [just](#architectures) discussed.

* [Tutorial](tutorial.ipynb): <code>tutorial.ipynb</code>. The tutorial covers the basic mathematical formulation for the graph neural networks, and considers a small synthetic problem of source localization. It implements the Aggregation and Selection GNN (both zero-padding and graph coarsening). This tutorial explain, in-depth, all the elements intervining in the setup, training and evaluation of the models, that serves as skeleton for all the other examples.

* [Source Localization](sourceLocGNN.py): <code>sourceLocGNN.py</code>. This example deals with the source localization problem on a 40-node, 2-community random-generated SBM graph. It considers multiple graph and data realizations to account for randomness in data generation. The trained architectures in this example are the Edge-Variant, the Node-Variant and the GAT.

* [MovieLens](movieGNN.py): <code>movieGNN.py</code>. This example has the objective of predicting the rating a given user would give to a movie, based on the movies it has ranked before (following the <a href="http://grouplens.org/datasets/movielens/100k/">MovieLens-100k</a> dataset). The Selection GNN, both with zero-padding and graph coarsening, are implemented.

* [Authorship Attribution](authorshipGNN.py): <code>authorshipGNN.py</code>. This example addresses the problem of authorship attribution, by which a text has to be assigned to some author according to their styolemtric signature (based on the underlying word adjacency network; details <a href="http://ieeexplore.ieee.org/document/6638728">here</a>). In this case, we test the Aggregation GNN and the Multi-Node variant of the Aggregation GNN.

* [20NEWS](20newsGNN.py): <code>20newsGNN.py</code>. The last example deals with the problem of text categorization in the 20NEWS dataset. The objective is to determine whether a given text belongs to one of 20 different categories, based on the word frequency (the words likeness is given by a <a href="http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf">word2vec</a> embedding of the training set). In this last example, we test two Selection GNNs: one with graph coarsening and one with no pooling at all. 

## Version <a class="anchor" id="version"></a>

* ___0.1 (Jul 12, 2019):___ First released (beta) version of this graph neural network library. Includes the basic convolutional graph neural networks (selection -zero-padding and graph coarsening-, spectral, aggregation), and some non-convolutional graph neural networks as well (node-variant, edge-variant and graph attention networks). It also inlcudes local activation functions (max and median). In terms of examples, it considers the source localization problem (both in the tutorial and in a separate example), the movie recommendation problem, the authorship attribution problem and the text categorization problems. In terms of structure, it sets the basis for data handling and training of multiple models.
