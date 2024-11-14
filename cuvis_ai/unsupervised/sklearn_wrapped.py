from ..node.wrap import make_node
import sklearn.cluster
import sklearn.mixture


@make_node
class KMeans(sklearn.cluster.KMeans):
    pass


@make_node
class MeanShift(sklearn.cluster.MeanShift):
    pass


@make_node
class GMM(sklearn.mixture.GaussianMixture):
    pass
