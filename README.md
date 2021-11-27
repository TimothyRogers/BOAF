# Birds Of A Feather (BOAF)
 
 BOAF is a python module which aims to provide a simple interface for solving clustering problems.

 ## Available Algorithms

 Currently available techniques:
 - KMeans/Kmeans++  - `boaf.KMeans`
 - Mixture Models `boaf.Mixture`:
   - Gaussian Mixture Model `boaf.GMM`
 
## Usage

 Every algorithm for clustering is built around an abstract class `Cluster`. Every approach must provide two methods:
 - `Cluster.learn` - where the parameters of the algorithm are learnt from data
 - `Cluster.predict` - where data are assigned indices indicating which cluster they belong to

### Example Usage

To start with let's look at a simple KMeans clustering problem. First we will define some options for the method, for KMeans we need to know the number of clusters and a maximum number of iterations to learn for. Options are specified as a `dict` which is passed to the `__init__` method of the `KMeans` class.

```
opts={
    'nclusters':nclusters,
    'niters':100
    }
```
The `KMeans` class defaults to using KMeans++ as the initialisation technique but this can be overridden with:
```
opts={
    'nclusters':nclusters,
    'niters':100,
    'init':'rand'
    }
```
Next, we need to initialise the clustering algorithm.
```
model = KMeans(opts_kmeans)
```
The object `model` now provides the interface for performing clustering. It will keep track of the mean locations of the clusters to assess any new data. To learn those cluster centers we will need to have some data from which they can be obtained.

All models expect data in `NDArray` objects with a size `(N,D)` where there are `N` observations (data points) in `D` dimensions. Let's imagine there is an array of data `X` available. To learn the cluster centers one can simply call:
```
model.learn(X)
```
To make a prediction, which returns the indices of the most likely cluster associated with each data point, as an `NDArray` of size `(N,)`. We can all the predict method passing in the data to be assessed, e.g.
```
inds = model.predict(X)
```
This would give us the indices of the clusters for each point in `X` used to learn the model. We could also for a new test dataset `Xt` get the indices of the most likely clusters for the test points without changing the parameters of the model (in this case the cluster centers).
```
inds_test = model.predict(Xt)
```

#### A Slightly More Complicated Example
Let's see how to fit a GMM using *maximum a posteriori* (MAP) expectation-maximisation (EM).

As before we specify the options relevant to the GMM clustering method. We will explicitly use a Normal-Inverse-Wishart base distribution through the `MixtureModel` interface but using `GMM` simply abstracts this step.

The options dictionary now contains the number of clusters (`nclusters`), number of EM iterations (`niters`) and the specification of the prior distribution (`prior`).

```
opts = {
        'nclusters': nclusters,
        'niters': 200,
        'prior':{
            'alpha': np.ones((nclusters))/nclusters,
            'cluster':{
                'nu':2,
                'kappa':1,
                'mu':np.zeros((1,2)),
                'sigma':np.eye(2)
            }
        }
    }
```
We now initialise the mixture model with our options dictionary and passing the base distribution class in (`NIW`).
```
model = MixtureModel(opts, base_distribution=NIW)
```
Out of interest we may wish to manually initialise the clusters and inspect the starting point of the model prior to performing EM. 
```
model._init_clusters(X)
```
We can also check what the log posterior predictive likelihood is for each data point immediately following initialisation. 
```
ll = np.hstack([c.logpredpdf(X)[:,None] for c in model.clusters]) + model.mixing_proportions
lln = ll - logsumexp(ll.T).T
```
Then finally, we can learn the model (run the EM procedure) and predict the indices for the training data.
```
model.learn(X)
inds_pred = model.predict(X)
```
