# Awesome Topic Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)


> A curated list of amazing topic modelling libraries.


## Contents

- [Libraries & Toolkits](#libraries--toolkits)
- [Models](#models)
- [Techniques](#techniques)
- [Research Implementations](#research-implementations)
- [Visualizations](#visualizations)
- [Resources](#resources)
- [Related awesome lists](#related-awesome-lists)


## Libraries & Toolkits

- [gensim](https://github.com/RaRe-Technologies/gensim) - Python library for topic modelling ![GitHub Repo stars](https://img.shields.io/github/stars/RaRe-Technologies/gensim?style=social)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Python library for machine learning ![GitHub Repo stars](https://img.shields.io/github/stars/scikit-learn/scikit-learn?style=social)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for Gibbs sampling based *tomoto* which is written in C++ ![GitHub Repo stars](https://img.shields.io/github/stars/bab2min/tomotopy?style=social)
- [tomoto](https://github.com/ankane/tomoto) - Ruby extension for Gibbs sampling based *tomoto* which is written in C++ ![GitHub Repo stars](https://img.shields.io/github/stars/ankane/tomoto?style=social)
- [OCTIS](https://github.com/MIND-Lab/OCTIS) - Python package to integrate, optimize and evaluate topic models ![GitHub Repo stars](https://img.shields.io/github/stars/MIND-Lab/OCTIS?style=social)
- [tmtoolkit](https://github.com/WZBSocialScienceCenter/tmtoolkit) -  Python topic modeling toolkit with parallel processing power ![GitHub Repo stars](https://img.shields.io/github/stars/WZBSocialScienceCenter/tmtoolkit?style=social)
- [Mallet](https://github.com/mimno/Mallet) - Java-based package for topic modeling ![GitHub Repo stars](https://img.shields.io/github/stars/mimno/mallet?style=social)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java-based package for topic modeling ![GitHub Repo stars](https://img.shields.io/github/stars/soberqian/TopicModel4J?style=social)
- [BIDMach](https://github.com/BIDData/BIDMach) - CPU and GPU-accelerated machine learning library ![GitHub Repo stars](https://img.shields.io/github/stars/BIDData/BIDMach?style=social)
- [BigARTM](https://github.com/bigartm/bigartm) - Fast topic modeling platform ![GitHub Repo stars](https://img.shields.io/github/stars/bigartm/bigartm?style=social)
- [TopicNet](https://github.com/machine-intelligence-laboratory/TopicNet) - A high-level Python interface for BigARTM library ![GitHub Repo stars](https://img.shields.io/github/stars/machine-intelligence-laboratory/TopicNet?style=social)
- [stm](https://github.com/bstewart/stm) - R package for the Structural Topic Model ![GitHub Repo stars](https://img.shields.io/github/stars/bstewart/stm?style=social)
- [RMallet](https://github.com/mimno/RMallet) -  R package to interface with the Java machine learning tool MALLET ![GitHub Repo stars](https://img.shields.io/github/stars/mimno/RMallet?style=social)
- [R-lda](https://github.com/slycoder/R-lda) - R package for topic modelling (LDA, sLDA, corrLDA, etc.) ![GitHub Repo stars](https://img.shields.io/github/stars/slycoder/R-lda?style=social)
- [topicmodels](https://github.com/cran/topicmodels) - R package with interface to C code for LDA and CTM ![GitHub Repo stars](https://img.shields.io/github/stars/cran/topicmodels?style=social)
- [lda++](https://github.com/angeloskath/supervised-lda) - C++ library for LDA and (fast) supervised LDA (sLDA/fsLDA) using variational inference ![GitHub Repo stars](https://img.shields.io/github/stars/angeloskath/supervised-lda?style=social)


## Models

There are huge differences in performance and scalability as well as the support of advanced features as hyperparameter tuning or evaluation capabilities.

### Truncated Singular Value Decomposition (SVD) / Latent Semantic Analysis (LSA) / Latent Semantic Indexing (LSI)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) - Python implementation using fast [randomized SVD solver](https://arxiv.org/pdf/0909.4061.pdf) or a “naive” algorithm that uses [ARPACK](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html)
- [gensim](https://radimrehurek.com/gensim/models/lsimodel.html) - Python implementation using multi-pass [randomized SVD solver](https://arxiv.org/pdf/0909.4061.pdf) or a [one-pass merge algorithm](https://rdcu.be/cghAi)
- [SVDlibc](https://github.com/lucasmaystre/svdlibc) - C implementation of SVD by Doug Rohde
- [sparsesvd](https://github.com/RaRe-Technologies/sparsesvd) - Python wrapper for SVDlibc
- [BIDMach](https://github.com/BIDData/BIDMach/blob/master/src/main/scala/BIDMach/models/SVD.scala) - Scala implementation of a scalable approximate SVD using subspace iteration

### Non-Negative Matrix Factorization (NMF or NNMF)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) - Python implementation using a [coordinate descent](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.214.6398&rep=rep1&type=pdf) or a [multiplicative update](https://arxiv.org/pdf/1010.1763.pdf) solver
- [gensim](https://radimrehurek.com/gensim/models/nmf.html) - Python implementation of [online NMF](https://arxiv.org/pdf/1604.02634.pdf)
- [BIDMach](https://github.com/BIDData/BIDMach/blob/master/src/main/scala/BIDMach/models/NMF.scala) - CPU and GPU-accelerated Scala implementation with L2 loss

### Latent Dirichlet Allocation (LDA) [:page_facing_up:](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) - Python implementation using online variational Bayes inference [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [lda](https://github.com/lda-project/lda) - Python implementation using collapsed Gibbs sampling which follows scikit-learn interface [:page_facing_up:](https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf)
- [lda-gensim](https://radimrehurek.com/gensim/models/ldamodel.html) - Python implementation using online variational inference [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [ldamulticore-gensim](https://radimrehurek.com/gensim/models/ldamulticore.html) - Parallelized Python implementation using online variational inference [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [GibbsSamplingLDA-TopicModel4J](https://github.com/soberqian/TopicModel4J/blob/master/src/main/java/com/topic/model/CVBLDA.java) - Java implementation using collapsed Gibbs sampling [:page_facing_up:](https://www.pnas.org/content/pnas/101/suppl_1/5228.full.pdf)
- [CVBLDA-TopicModel4J](https://github.com/soberqian/TopicModel4J/blob/master/src/main/java/com/topic/model/CVBLDA.java) - Java implementation using collapsed variational Bayesian (CVB) inference [:page_facing_up:](https://papers.nips.cc/paper/2006/file/532b7cbe070a3579f424988a040752f2-Paper.pdf)
- [Mallet](https://github.com/mimno/Mallet/blob/master/src/cc/mallet/topics/ParallelTopicModel.java) - Parallelized Java implementation using Gibbs sampling [:page_facing_up:](https://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)[:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1557019.1557121)
- [gensim-wrapper-Mallet](https://github.com/mimno/Mallet/blob/master/src/cc/mallet/topics/ParallelTopicModel.java) - Python wrapper for Mallet's implementation [:page_facing_up:](https://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)[:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1557019.1557121)
- [PartiallyCollapsedLDA](https://github.com/lejon/PartiallyCollapsedLDA) - Various fast parallelized samplers for LDA, including Partially Collapsed LDA, LightLDA, Partially Collapsed Light LDA and a very efficient Polya-Urn LDA
- [Vowpal Wabbit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Latent-Dirichlet-Allocation) - C++ implementaion using online variational Bayes inference [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python binding for C++ implementation using Gibbs sampling and different [term-weighting](https://www.aclweb.org/anthology/N10-1070.pdf) options [:page_facing_up:](https://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)
- [topicmodel-lib](https://github.com/hncuong/topicmodel-lib) - Cython library for online/streaming LDA (Online VB, Online CVB0, Online CGS, Online OPE, Online FW, Streaming VB, Streaming OPE, Streaming FW, ML-OPE, ML-CGS, ML-FW)
- [jsLDA](https://github.com/mimno/jsLDA) - JavaScript implementation of LDA topic modeling in the browser
- [lda-nodejs](https://github.com/primaryobjects/lda) - Node.js implementation of LDA topic modeling
- [lda-purescript](https://github.com/lettier/lda-topic-modeling) - PureScript, browser-based implementation of LDA topic modeling
- [TopicModels.jl](https://github.com/slycoder/TopicModels.jl) - Julia implementation of LDA
- [turicreate](https://github.com/apple/turicreate) - C++ [LDA](https://github.com/apple/turicreate/blob/master/userguide/text/README.md) and [aliasLDA](https://apple.github.io/turicreate/docs/api/generated/turicreate.topic_model.create.html) implementation with export to Apple's Core ML for use in iOS, macOS, watchOS, and tvOS apps
- [MeTA](https://github.com/meta-toolkit/meta) - C++ implementation of (parallel) collapsed [Gibbs sampling, CVB0 and SCVB](https://meta-toolkit.org/topic-models-tutorial.html)
- [Fugue](https://github.com/PuzaTech/Fugue) - Java implementation of collapsed Gibbs sampling with slice sampling for hyper-parameter optimization

#### Hyperparameter optimization
- [GA-LDA](https://github.com/GESAD-MSR/GA-LDA) - R scripts using Genetic Algorithms (GA) for hyper-paramenter optimization, based on Panichella [:page_facing_up:](https://doi.org/10.1016/j.infsof.2020.106411)
- [Search-Based-LDA](https://github.com/apanichella/Search-Based-LDA) - R scripts using Genetic Algorithms (GA) for hyper-paramenter optimization by Panichella [:page_facing_up:](https://doi.org/10.1016/j.infsof.2020.106411)
- [Dodge](https://github.com/amritbhanu/Dodge) - Python tuning tool that ignores redundant tunings [:page_facing_up:](https://arxiv.org/pdf/1902.01838.pdf)
- [LDADE](https://github.com/amritbhanu/LDADE-package) - Python tuning tool using differential evolution [:page_facing_up:](https://arxiv.org/pdf/1608.08176.pdf)
- [ldatuning](https://github.com/nikita-moor/ldatuning) - R package to find optimal number of topics for LDA [:page_facing_up:](https://rpubs.com/siri/ldatuning)
- [Scalable](https://www.tandfonline.com/doi/suppl/10.1080/10618600.2020.1741378) - Scalable Hyperparameter Selection for LDA [:page_facing_up:](https://www.tandfonline.com/doi/full/10.1080/10618600.2020.1741378)
- [topic_interpretability](https://github.com/jhlau/topic_interpretability) - 
- [topic-coherence-sensitivity](https://github.com/jhlau/topic-coherence-sensitivity) - 
- [topic-model-diversity](https://github.com/silviatti/topic-model-diversity) - 

#### CPU-based high performance implementations
- [LDA\*](https://github.com/Angel-ML/angel/blob/master/docs/algo/lda_on_angel_en.md) - Tencent's hybrid sampler that uses different samplers for different types of documents in combination with an asymmetric parameter server [:page_facing_up:](http://www.vldb.org/pvldb/vol10/p1406-yu.pdf)
- [FastLDA](https://github.com/Arnie0426/FastLDA) - C++ implementation of LDA [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1401890.1401960)
- [dmlc](https://github.com/dmlc/experimental-lda) - Single-and multi-threaded C++ implementations of [lightLDA](https://arxiv.org/pdf/1412.1576.pdf), [F+LDA](https://arxiv.org/pdf/1412.4986v1.pdf), [AliasLDA](https://dl.acm.org/doi/pdf/10.1145/2623330.2623756), forestLDA and many more
- [SparseLDA](https://github.com/mimno/Mallet/blob/master/src/cc/mallet/topics/ParallelTopicModel.java) - Java algorithm and data structure for evaluating Gibbs sampling distributions used in Mallet [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1557019.1557121)
- [warpLDA](https://github.com/thu-ml/warplda) - C++ cache efficient LDA implementation which samples each token in O(1) [:page_facing_up:](https://arxiv.org/pdf/1510.08628.pdf)
- [lightLDA](https://github.com/microsoft/LightLDA) - C++ implementation using O(1) Metropolis-Hastings sampling [:page_facing_up:](https://arxiv.org/pdf/1412.1576.pdf)
- [F+LDA](https://bigdata.oden.utexas.edu/software/nomad) - C++ implementation of F+LDA using an appropriately modified Fenwick tree [:page_facing_up:](https://arxiv.org/pdf/1412.4986v1.pdf)
- [AliasLDA](https://github.com/polymorpher/aliaslda) - C++ implemenation using Metropolis-Hastings and *alias* method[:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/2623330.2623756)
- [Yahoo-LDA](https://github.com/sudar/Yahoo_LDA) - Yahoo!'s topic modelling framework [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/2124295.2124312)
- [PLDA+](https://github.com/openbigdatagroup/plda) - Google's C++ implementation using data placement and pipeline processing [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1961189.1961198)
- [Familia](https://github.com/baidu/Familia) - A toolkit for industrial topic modeling (LDA, SentenceLDA and Topical Word Embedding) [:warning:](https://github.com/baidu/Familia/issues/111) [:page_facing_up:](https://arxiv.org/pdf/1707.09823.pdf)

#### GPU-based high performance implementations
- [SaberLDA](https://dl.acm.org/doi/pdf/10.1145/3093336.3037740) - GPU-based system that implements a sparsity-aware algorithm to achieve sublinear time complexity
- [GS-LDA-BIDMach](https://github.com/BIDData/BIDMach/blob/master/src/main/scala/BIDMach/models/LDAgibbs.scala) - CPU and GPU-accelerated Scala implementation using Gibbs sampling
- [VB-LDA-BIDMach](https://github.com/BIDData/BIDMach/blob/master/src/main/scala/BIDMach/models/LDA.scala) - CPU and GPU-accelerated Scala implementation using online variational Bayes inference

### Hierarchical Dirichlet Process (HDP) [:page_facing_up:](https://papers.nips.cc/paper/2004/file/fb4ab556bc42d6f0ee0f9e24ec4d1af0-Paper.pdf)
- [gensim](https://radimrehurek.com/gensim/models/hdpmodel.html) - Python implementation using online variational inference [:page_facing_up:](http://proceedings.mlr.press/v15/wang11a/wang11a.pdf)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling [:page_facing_up:](https://www.jmlr.org/papers/volume10/newman09a/newman09a.pdf)
- [Mallet](https://github.com/mimno/Mallet) - Java-based package for topic modeling using Gibbs sampling
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using Gibbs sampling based on Chinese restaurant franchise metaphor
- [hca](https://github.com/wbuntine/topic-models) - C implementation using Gibbs sampling with/without burstiness modelling 
- [bnp](https://github.com/chyikwei/bnp) - Cython reimplementation based on *online-hdp* following scikit-learn's API.
- [Scalable HDP](http://www.vldb.org/pvldb/vol11/p826-chen.pdf) - interesting paper

### Hierarchical LDA (hLDA) [:page_facing_up:](https://dl.acm.org/doi/10.5555/2981345.2981348)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [Mallet](https://github.com/mimno/Mallet/blob/master/src/cc/mallet/topics/HierarchicalLDA.java) - Java implementation using Gibbs sampling
- [hlda](https://github.com/joewandy/hlda) - Python package based on *Mallet's* Gibbs sampler having a fixed depth on the nCRP tree
- [hLDA](https://github.com/blei-lab/hlda) - C implementation of hierarchical LDA by David Blei


### Dynamic Topic Model (DTM) [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1143844.1143859)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling based on FastDTM
- [FastDTM](https://github.com/Arnie0426/FastDTM) - Scalable C++ implementation using Gibbs sampling with Stochastic Gradient Langevin Dynamics (MCMC-based) [:page_facing_up:](https://arxiv.org/pdf/1602.06049.pdf)
- [ldaseqmodel-gensim](https://radimrehurek.com/gensim_3.8.3/models/ldaseqmodel.html) - Python implementation using online variational inference [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [dtm-BigTopicModel](https://github.com/thu-ml/BigTopicModel) - C++ engine for running large-scale topic models
- [tca](https://github.com/wbuntine/topic-models/blob/master/HCA/doc/tcaman.pdf) - C implementation using Gibbs sampling with/without burstiness modelling [:page_facing_up:](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.705.1649&rep=rep1&type=pdf)
- [DETM](https://github.com/adjidieng/DETM) - Python implementation of the Dynamic Embedded Topic Model [:page_facing_up:](https://arxiv.org/pdf/1907.05545.pdf)


### Author-topic Model (ATM) [:page_facing_up:](https://arxiv.org/pdf/1207.4169.pdf)
- [gensim](https://radimrehurek.com/gensim/models/atmodel.html) - Python implementation with online training (constant in memory w.r.t. the number of documents)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation
- [Matlab Topic Modeling Toolbox](https://github.com/jonaschn/Matlab-Topic-Modeling-Toolbox) - Matlab and C++ implementation using Gibbs sampling
- [Topic-Model](https://github.com/Ward-nju/Topic-Model) - Simple Python implementation using Gibbs sampling

### Labeled Latent Dirichlet Allocation (LLDA, Labeled-LDA, L-LDA) [:page_facing_up:](https://www.aclweb.org/anthology/D09-1026.pdf)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation
- [Mallet](https://github.com/mimno/Mallet/blob/master/src/cc/mallet/topics/LabeledLDA.java) - Java implementation using Gibbs sampling [:page_facing_up:](http://www.mimno.org/articles/labelsandpatterns)
- [gensims_mallet_wrapper](https://github.com/jonaschn/gensim/tree/labeled-lda) - Python wrapper for Mallet using gensim interface
- [STMT](https://nlp.stanford.edu/software/tmt/tmt-0.4/) - Scala implementation by Daniel Ramage
- [topbox](https://github.com/jonaschn/topbox) - Python wrapper for labeled LDA implementation of *Stanford TMT*
- [Labeled-LDA-Python](https://github.com/JoeZJH/Labeled-LDA-Python) - Python implementation (easy to use, does not scale)
- [JGibbLabeledLDA](https://github.com/myleott/JGibbLabeledLDA) - Java implementation based on the popular [JGibbLDA](jgibblda.sourceforge.net) package


### Partially Labeled Dirichlet Allocation (PLDA) / Dirichlet Process (PLDP) [:page_facing_up:](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/08/KDD2011-pldp-final.pdf)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling
- [STMT](https://nlp.stanford.edu/software/tmt/tmt-0.4/) - Scala implementation of PLDA & PLDP by Daniel Ramage

### Dirichlet Multinomial Regression (DMR) topic model [:page_facing_up:](https://dl.acm.org/doi/10.5555/3023476.3023525)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [Mallet](https://github.com/mimno/Mallet) - Java-based package for topic modeling 

### Generalized Dirichlet Multinomial Regression (g-DMR) topic model [:page_facing_up:](https://dl.acm.org/doi/10.1007/s11192-020-03508-3)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling


### Link LDA
- [PTM](https://github.com/yao8839836/PTM) - implemented as benchmark [:page_facing_up:](https://ieeexplore.ieee.org/abstract/document/8242679)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling 

### Correlated Topic Model (CTM) a.k.a. logistic-normal topic models
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling [:page_facing_up:](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.149.922)
- [ctm-c](https://github.com/blei-lab/ctm-c) - Original C implementation of the correlated topic model by David Blei [:page_facing_up:](https://proceedings.nips.cc/paper/2005/file/9e82757e9a1c12cb710ad680db11f6f1-Paper.pdf)
- [BigTopicModel](https://github.com/thu-ml/BigTopicModel) - C++ engine for running large-scale DTM [:page_facing_up:](https://papers.nips.cc/paper/2013/file/285f89b802bcb2651801455c86d78f2a-Paper.pdf)
 - [stm](https://github.com/bstewart/stm) - R package for the Structural Topic Model (CTM in case of no covariates) [:page_facing_up:](https://github.com/bstewart/stm/blob/master/vignettes/stmVignette.pdf?raw=true)

### Relational Topic Model (RTM)
- [BigTopicModel](https://github.com/thu-ml/BigTopicModel) - C++ engine for running large-scale topic models
- [Constrained-RTM](https://github.com/MIND-Lab/Constrained-RTM) - Java implementation of Contrained RTM [:page_facing_up:](https://doi.org/10.1016/j.ins.2019.09.039)
- [R-lda](https://github.com/slycoder/R-lda) - R implementation using collapsed Gibbs sampling


### Supervised LDA (sLDA) [:page_facing_up:](https://papers.nips.cc/paper/2007/file/d56b9fc4b0f1be8871f5e1c40c0067e7-Paper.pdf)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [R-lda](https://github.com/slycoder/R-lda) - R implementation using collapsed Gibbs sampling
- [slda](https://github.com/Savvysherpa/slda) -  Cython implementation of Gibbs sampling for LDA and various sLDA variants
	- supervised LDA (linear regression)
    - binary logistic supervised LDA (logistic regression)
    - binary logistic hierarchical supervised LDA (trees)
    - generalized relational topic models (graphs)
- [YWWTools](https://github.com/ywwbill/YWWTools-v2#slda-supervised-lda) - Java implementation using Gibbs sampling for LDA and various sLDA variants:
	- BS-LDA: Binary SLDA
    - Lex-WSB-BS-LDA: BS-LDA with Lexcial Weights and Weighted Stochastic Block Priors
    - Lex-WSB-Med-LDA: Lex-WSB-BS-LDA with Hinge Loss
- [sLDA](https://github.com/blei-lab/class-slda) - C++ implementation of supervised topic models with a categorical response


### Topic Models for short documents
#### Sentence-LDA / SentenceLDA / Sentence LDA [:page_facing_up:](https://dl.acm.org/doi/10.1145/1935826.1935932)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation of Sentence-LDA using collapsed Gibbs sampling
- [Familia](https://github.com/baidu/Familia) - Apply inference on pre-trained SentenceLDA models [:warning:](https://github.com/baidu/Familia/issues/111) [:page_facing_up:](https://arxiv.org/pdf/1707.09823.pdf)

#### Dirichlet Multinomial Mixture Model (DMM) [:page_facing_up:](https://link.springer.com/content/pdf/10.1023/A:1007692713085.pdf)
- [GPyM_TM](https://github.com/jrmazarura/GPM) - Python implementation of DMM and Poisson model
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling [:page_facing_up:](https://dl.acm.org/doi/10.1145/2623330.2623715)
- [jLDADMM](https://github.com/datquocnguyen/jLDADMM) - Java implementation using collapsed Gibbs sampling [:page_facing_up:](https://arxiv.org/pdf/1808.03835.pdf)

#### Dirichlet Process Multinomial Mixture Model (DPMM)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling [:page_facing_up:](https://ieeexplore.ieee.org/document/7498276)

#### Pseudo-document-based Topic Model (PTM) [:page_facing_up:](https://dl.acm.org/doi/10.1145/2939672.2939880)
- [tomotopy](https://github.com/bab2min/tomotopy) - Python extension for C++ implementation using Gibbs sampling
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling

#### Biterm topic model (BTM)
- [TopicModel4J](https://github.com/soberqian/TopicModel4J) - Java implementation using collapsed Gibbs sampling
- [BTM](https://github.com/xiaohuiyan/BTM) - Original C++ implementation using collapsed Gibbs sampling [:page_facing_up:](https://raw.githubusercontent.com/xiaohuiyan/xiaohuiyan.github.io/master/paper/BTM-WWW13.pdf)
- [BurstyBTM](https://github.com/xiaohuiyan/BurstyBTM) - Original C++ implementation of the Bursty BTM (BBTM) [:page_facing_up:](https://raw.githubusercontent.com/xiaohuiyan/xiaohuiyan.github.io/master/paper/BBTM-AAAI15.pdf)
- [OnlineBTM]() - Original C++ implementation of online BTM (oBTM) and incremental BTM (iBTM) [:page_facing_up](https://raw.githubusercontent.com/xiaohuiyan/xiaohuiyan.github.io/master/paper/BTM-TKDE.pdf)
- [R-BTM](https://github.com/bnosac/BTM) - R package wrapping the C++ code from BTM

#### Others
- [STTM](https://github.com/qiang2100/STTM) - Java implementation and evaluation of DMM, WNTM, PTM, ETM, GPU-DMM, GPU-DPMM, LF-DMM [:page_facing_up:](https://arxiv.org/pdf/1904.07695.pdf)
- [SATM](https://github.com/WHUIR/SATM) - Java implementation of Self-Aggregation Topic Model [:page_facing_up:](https://dl.acm.org/doi/10.5555/2832415.2832564)
- [shorttext](https://github.com/stephenhky/PyShortTextCategorization) -  Python implementation of various algorithms for Short Text Mining


### Miscellaneous topic models
- [trLDA](https://github.com/lucastheis/trlda/) - Python implementation of streaming LDA based on trust-regions [:page_facing_up:](http://proceedings.mlr.press/v37/theis15.pdf)
- [Logistic LDA](https://github.com/lucastheis/logistic_lda) - Tensorflow implementation of Discriminative Topic Modeling with Logistic LDA [:page_facing_up:](https://proceedings.neurips.cc/paper/2019/file/54ebdfbbfe6c31c39aaba9a1ee83860a-Paper.pdf)
- [EnsTop](https://github.com/lmcinnes/enstop) - Python implementation of *ENS*emble *TOP*ic modelling with pLSA
- [Dual-Sparse Topic Model](https://github.com/soberqian/TopicModel4J/blob/master/src/main/java/com/topic/model/DualSparseLDA.java) - implemented in TopicModel4J using collapsed variational Bayes inference [:page_facing_up:](https://dl.acm.org/doi/10.1145/2566486.2567980)
- [Multi-Grain-LDA](https://github.com/bab2min/tomotopy) - MG-LDA implemented in tomotopy using collapsed Gibbs sampling [:page_facing_up:](https://dl.acm.org/doi/10.1145/1367497.1367513)
- [lda++](https://github.com/angeloskath/supervised-lda) - C++ library for LDA and (fast) supervised LDA (sLDA/fsLDA) using variational inference [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/2964284.2967237) [:page_facing_up:](http://www.cs.columbia.edu/~blei/papers/WangBleiFeiFei2009.pdf)
- [discLDA](https://github.com/anthonylife/discLDA) - C++ implementation of discLDA based on GibbsLDA++ [:page_facing_up:](https://papers.nips.cc/paper/2008/file/7b13b2203029ed80337f27127a9f1d28-Paper.pdf)
- [GuidedLDA](https://github.com/vi3k6i5/GuidedLDA) - Python implementation that can be guided by setting some seed words per topic (using Gibbs sampling) [:page_facing_up:](https://www.aclweb.org/anthology/E12-1021.pdf)
- [seededLDA](https://github.com/koheiw/seededlda) - R package that implements seeded-LDA for semi-supervised topic modeling
- [keyATM](https://github.com/keyATM/keyATM) - R package for Keyword Assisted Topic Models.
- [hca](https://github.com/wbuntine/topic-models) - C implementation of non-parametric topic models (HDP, HPYP-LDA, etc.) with focus on hyperparameter tuning
- [BayesPA](https://github.com/strin/BayesPA) - Python interface for streaming implementation of MedLDA, maximum entropy discrimination LDA (max-margin supervised topic model) [:page_facing_up:](http://proceedings.mlr.press/v32/shi14.pdf)	
- [sailing-pmls](https://pmls.readthedocs.io/en/latest/med-lda.html) - Parallel LDA and medLDA implementation
- [BigTopicModel](https://github.com/thu-ml/BigTopicModel) - C++ engine for running large-scale MedLDA models [:page_facing_up:](https://dl.acm.org/doi/10.1145/2487575.2487658)
- [DAPPER](https://github.com/robert-giaquinto/dapper) - Python implementation of Dynamic Author Persona (DAP) topic model [:page_facing_up:](https://arxiv.org/pdf/1811.01931.pdf)
- [ToT](https://github.com/ahmaurya/topics_over_time) - Python implementation of Topics Over Time (A Non-Markov Continuous-Time Model of Topical Trends) [:page_facing_up:](https://dl.acm.org/doi/10.1145/1150402.1150450)
- [MLTM](https://github.com/hsoleimani/MLTM) - C implementation of multilabel topic model (MLTM) [:page_facing_up:](https://www.mitpressjournals.org/doi/pdf/10.1162/NECO_a_00939)
- [sequence-models](https://github.com/michaeljpaul/sequence-models) - Java implementation of block HMM and the mixed membership Markov model (M4)
- [Entropy-Based Topic Modeling](https://github.com/julian-risch/JCDL2018/) - Java implementation of Entropy-Based Topic Modeling on Multiple Domain-Specific Text Collections
- [ST-LDA](https://github.com/ywwbill/YWWTools-v2#st-lda-single-topic-lda) - ST-LDA: Single Topic LDA [:page_facing_up:](https://ywwbill.github.io/files/2016_socinfo_topicDynamic.pdf)
- [MTM](https://github.com/ywwbill/YWWTools-v2#mtm-in-command-line) - Java implementation of Multilingual Topic Model [:page_facing_up:](https://www.aclweb.org/anthology/D19-1120.pdf)
- [YWWTools](https://github.com/ywwbill/YWWTools-v2) - Java-based package for various topic models by Weiwei Yang


### Exotic models
- [TEM](https://github.com/jonaschn/TopicExpertiseModel) - Topic Expertise Model [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/2505515.2505720)
- [PTM](https://github.com/yao8839836/PTM) - Prescription Topic Model for Traditional Chinese Medicine Prescriptions [:page_facing_up:](https://ieeexplore.ieee.org/abstract/document/8242679) (interesting benchmark models)
- [KGE-LDA](https://github.com/yao8839836/KGE-LDA) - Knowledge Graph Embedding LDA [:page_facing_up:](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14170/14086) 
- [LDA-SP](https://github.com/aritter/LDA-SP) - A Latent Dirichlet Allocation Method for Selectional Preferences [:page_facing_up:](https://www.aclweb.org/anthology/P10-1044.pdf)
- [LDA+FFT](https://github.com/ai-se/LDA_FFT) - LDA and FFTs (Fast and Frugal Trees) for better comprehensibility [:page_facing_up:](https://arxiv.org/pdf/1804.10657.pdf)

### Neural Topic Models

### Embedding based

- [CTM](https://github.com/MilaNLProc/contextualized-topic-models) - CTMs combine contextualized embeddings (e.g., BERT) with topic models
- [ETM](https://github.com/adjidieng/ETM) - Embedded Topic Model [:page_facing_up:](https://arxiv.org/pdf/1907.04907.pdf)
- [D-ETM](https://github.com/adjidieng/DETM) - Dynamic Embedded Topic Model [:page_facing_up:](https://arxiv.org/pdf/1907.05545.pdf)
- [ProdLDA](https://github.com/akashgit/autoencoding_vi_for_topic_models) - Original TensorFlow implementation of Autoencoding Variational Inference (AEVI) for Topic Models [:page_facing_up:](https://arxiv.org/pdf/1703.01488.pdf)
- [pytorch-ProdLDA](https://github.com/hyqneuron/pytorch-avitm) - PyTorch implementation of ProdLDA [:page_facing_up:](https://arxiv.org/pdf/1703.01488.pdf)
- [CatE](https://github.com/yumeng5/CatE) -  Discriminative Topic Mining via Category-Name Guided Text Embedding [:page_facing_up:](https://arxiv.org/pdf/1908.07162.pdf)
- [Top2Vec](https://github.com/ddangelov/Top2Vec) - Python implementation that learns jointly embedded topic, document and word vectors [:page_facing_up:](https://arxiv.org/pdf/2008.09470.pdf)
- [lda2vec](https://github.com/cemoody/lda2vec) - Mixing dirichlet topic models and word embeddings to make lda2vec [:page_facing_up:](https://arxiv.org/pdf/1605.02019.pdf)
- [lda2vec-pytorch](https://github.com/TropComplique/lda2vec-pytorch) - PyTorch implementation of lda2vec
- [G-LDA](https://github.com/rajarshd/Gaussian_LDA) - Java implementation of Gaussian LDA using word embeddings [:page_facing_up:](https://www.aclweb.org/anthology/P15-1077.pdf)
- [MG-LDA](https://github.com/EliasKB/Multilingual-Gaussian-Latent-Dirichlet-Allocation-MGLDA) - Python implementation of (Multi-lingual) Gaussian LDA [:page_facing_up:](https://raw.githubusercontent.com/EliasKB/Multilingual-Gaussian-Latent-Dirichlet-Allocation-MGLDA/master/MGLDA.pdf)
- [MetaLDA](https://github.com/ethanhezhao/MetaLDA) - Java implementation using Gibbs sampling that leverages document metadata and word embeddings [:page_facing_up:](https://arxiv.org/pdf/1709.06365.pdf)
- [LFTM](https://github.com/datquocnguyen/LFTM) - Java implementation of latent feature topic models (improving LDA and DMM with word embeddings) [:page_facing_up:](https://www.aclweb.org/anthology/Q15-1022.pdf)

- [CorEx](https://github.com/gregversteeg/bio_corex) - Recover latent factors with Correlation Explanation (CorEx) [:page_facing_up:](https://arxiv.org/pdf/1406.1222.pdf)
- [Anchored CorEx](https://github.com/gregversteeg/corex_topic) - Hierarchical Topic Modeling with Minimal Domain Knowledge [:page_facing_up:](https://arxiv.org/pdf/1611.10277.pdf)
- [Linear CorEx](https://github.com/gregversteeg/LinearCorex) - Latent Factor Models Based on Linear Total CorEx [:page_facing_up:](https://arxiv.org/pdf/1706.03353v3.pdf)


## Probabilistic Programming Languages (PPL) (a.k.a. Build your own Topic Model)
- [Stan](https://github.com/stan-dev/stan) - Platform for statistical modeling and high-performance statistical computation, e.g., [LDA](https://mc-stan.org/docs/2_26/stan-users-guide/latent-dirichlet-allocation.html) [:page_facing_up:](https://files.eric.ed.gov/fulltext/ED590311.pdf)
- [PyMC3](https://github.com/pymc-devs/pymc3) - Python package for Bayesian statistical modeling and probabilistic machine learning, e.g., [LDA](http://docs.pymc.io/notebooks/lda-advi-aevb.html) [:page_facing_up:](https://peerj.com/articles/cs-55.pdf)
- [TFP](https://github.com/tensorflow/probability) - Probabilistic reasoning and statistical analysis in TensorFlow, e.g., [LDA](https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/latent_dirichlet_allocation_distributions.py) [:page_facing_up:](https://arxiv.org/pdf/2001.11819.pdf)
- [edward2](https://github.com/google/edward2) - Simple PPL with core utilities in the NumPy and TensorFlow ecosystem [:page_facing_up:](https://arxiv.org/pdf/1811.02091.pdf)
- [pyro](https://github.com/pyro-ppl/pyro) - PPL built on PyTorch, e.g., [prodLDA](http://pyro.ai/examples/prodlda.html) [:page_facing_up:](https://www.jmlr.org/papers/volume20/18-403/18-403.pdf)
- [edward](https://github.com/blei-lab/edward) - A PPL built on TensorFlow, e.g., [LDA](http://edwardlib.org/iclr2017?Figure%2011.%20Latent%20Dirichlet%20allocation) [:page_facing_up:](https://arxiv.org/pdf/1610.09787.pdf)
- [ZhuSuan](https://github.com/thu-ml/zhusuan) - A PPL for Bayesian deep learning, generative models, built on Tensorflow, e.g., [LDA](https://zhusuan.readthedocs.io/en/latest/tutorials/lntm.html) [:page_facing_up:](https://arxiv.org/pdf/1709.05870.pdf)


## Research Implementations

- [lda-c](https://github.com/blei-lab/lda-c) - C implementation using variational EM by David Blei
- [sLDA](https://github.com/blei-lab/class-slda) - C++ implementation of supervised topic models with a categorical response.
- [onlineldavb](https://github.com/blei-lab/onlineldavb) - Python online variational Bayes implementation by Matthew Hoffman [:page_facing_up:](https://proceedings.neurips.cc/paper/2010/file/71f6278d140af599e06ad9bf1ba03cb0-Paper.pdf)
- [HDP](https://github.com/blei-lab/hdp) - C++ implementation of hierarchical Dirichlet processes by Chong Wang
- [online-hdp](https://github.com/blei-lab/online-hdp) - Python implementation of online hierarchical Dirichlet processes by Chong Wang
- [ctr](https://github.com/blei-lab/ctr) - C++ implementation of collaborative topic models by Chong Wang
- [dtm](https://github.com/blei-lab/dtm) - C implementation of dynamic topic models by David Blei & Sean Gerrish
- [ctm-c](https://github.com/blei-lab/ctm-c) - C implementation of the correlated topic model by David Blei
- [diln](https://github.com/blei-lab/diln) - C implementation of Discrete Infinite Logistic Normal (with HDP option) by John Paisley
- [hLDA](https://github.com/blei-lab/hlda) - C implementation of hierarchical LDA by David Blei
- [turbotopics](https://github.com/blei-lab/turbotopics) - Python implementation that finds significant multiword phrases in topics by David Blei
- [Stanford Topic Modeling Toolbox](https://nlp.stanford.edu/software/tmt/tmt-0.4/) - Scala implementation of LDA, labeledLDA, PLDA, PLDP by Daniel Ramage and Evan Rosen
- [LDAGibbs](https://github.com/yangliuy/LDAGibbsSampling) - Java implementation of LDA using Gibbs sampling by Liu Yang
- [Matlab Topic Modeling Toolbox](https://github.com/jonaschn/Matlab-Topic-Modeling-Toolbox) - Matlab implementations of LDA, ATM, HMM-LDA, LDA-COL (Collocation) models by Mark Steyvers and Tom Griffiths
- [cvbLDA](https://github.com/davidandrzej/cvbLDA) -  Python C extension implementation of collapsed variational Bayesian inference for LDA
- [fast](https://www.ics.uci.edu/~asuncion/software/fast.htm) - A Fast And Scalable Topic-Modeling Toolbox (Fast-LDA, CVB0) by Arthur Asuncion and colleagues [:page_facing_up:](https://arxiv.org/pdf/1205.2662.pdf)


## Popular Implementations (but not maintained anymore)
- [Stanford Topic Modeling Toolbox](https://nlp.stanford.edu/software/tmt/tmt-0.4/) - Scala implementation of LDA, labeledLDA, PLDA, PLDP by Daniel Ramage and Evan Rosen
- [Matlab Topic Modeling Toolbox](https://github.com/jonaschn/Matlab-Topic-Modeling-Toolbox) - Matlab implementations of LDA, ATM, HMM-LDA, LDA-COL (Collocation) models by Mark Steyvers and Tom Griffiths
- [GibbsLDA++](http://gibbslda.sourceforge.net) - C++ implementation using Gibbs sampling [:page_facing_up:](https://dl.acm.org/doi/pdf/10.1145/1367497.1367510)
[:fork:](https://github.com/mrquincle/gibbs-lda)
- [JGibbLDA](http://jgibblda.sourceforge.net) - Java implementation using Gibbs sampling
- [Mr.LDA](https://github.com/lintool/Mr.LDA) - Scalable Topic Modeling using Variational Inference in MapReduce [:page_facing_up:](https://dl.acm.org/doi/10.1145/2187836.2187955)


## Toy Implementations (hopefully easy to understand)

- [topic_models](https://github.com/laserwave/topic_models) - Python implementation of LSA, PLSA and LDA
- [Topic-Model](https://github.com/Ward-nju/Topic-Model) - Python implementation of LDA, Labeled LDA, ATM, Temporal Author-Topic Model using Gibbs sampling


## Visualizations

- [LDAvis](https://github.com/cpsievert/LDAvis) - R package for interactive topic model visualization
- [pyLDAvis](https://github.com/bmabey/pyLDAvis) - Python library for interactive topic model visualization
- [scalaLDAvis](https://github.com/iaja/scalaLDAvis) - Scala port of pyLDAvis
- [dtmvisual](https://github.com/GSukr/dtmvisual) - Python package for visualizing DTM (trained with gensim)
- [TMVE online](https://github.com/ajbc/tmv) - Online Django variant of topic model visualization engine (*TMVE*) 
- [TMVE](https://github.com/ajbc/tmve-original) - Original topic model visualization engine (LDA trained with *lda-c*) [:page_facing_up:](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM12/paper/viewFile/4645/5021)
- [topicmodel-lib](https://test-dslab.readthedocs.io/en/latest/visualization.html) - Python wrapper for TMVE for visualizing LDA (trained with topicmodel-lib)
- [wordcloud](http://amueller.github.io/word_cloud/) - Python package for visualizing topics via word_cloud
- [Mallet-GUI](https://github.com/senderle/topic-modeling-tool) - GUI for creating and analyzing topic models produced by MALLET
- [TWiC](https://github.com/jarmoza/twic) - Topic Words in Context is a highly-interactive, browser-based visualization for MALLET topic models
- [Topics](https://github.com/DARIAH-DE/Topics) - Python library for topic modeling and visualization
- [TopicsExplorer](https://github.com/DARIAH-DE/TopicsExplorer) - Explore your own text collection with a topic model – without prior knowledge [:page_facing_up:](https://dh2018.adho.org/a-graphical-user-interface-for-lda-topic-modeling)
- [topicApp](https://github.com/wesslen/topicApp) - A Simple Shiny App for Topic Modeling
- [stminsights](https://github.com/cschwem2er/stminsights) - A Shiny Application for Inspecting Structural Topic Models

## Dirichlet hyperparameter optimization techniques
- [Slice sampling](https://people.cs.umass.edu/~cxl/cs691bm/lec08.html)
- [Minka](https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf)
    - [fastfit](https://github.com/tminka/fastfit) 
    - [dirichlet](https://github.com/ericsuh/dirichlet) Python port of fastfit
    - [lightspeed](https://github.com/tminka/lightspeed)
    - [lecture-notes](https://people.cs.umass.edu/~cxl/cs691bm/lec09.html)
- [Newton-Raphson Method](http://jonathan-huang.org/research/dirichlet/dirichlet.pdf)
- [fixed-point iteration](https://people.cs.umass.edu/~wallach/theses/wallach_phd_thesis.pdf) - Wallach's PhD thesis, chapter 2.3
## Resources

- [David Blei](http://www.cs.columbia.edu/~blei/topicmodeling.html) - David Blei's Homepage with introductory materials


## Related awesome lists

- [awesome-machine-learning](https://github.com/josephmisiti/awesome-machine-learning)
- [awesome-datascience](https://github.com/academic/awesome-datascience)
- [awesome-python-data-science](https://github.com/krzjoa/awesome-python-data-science)

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.


## License

[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](http://creativecommons.org/publicdomain/zero/1.0)

To the extent possible under law, Jonathan Schneider has waived all copyright and
related or neighboring rights to this work.
