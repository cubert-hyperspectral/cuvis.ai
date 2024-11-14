from ..node.wrap import make_node
import sklearn.decomposition


@make_node
class PCA(sklearn.decomposition.PCA):
    pass


@make_node
class NMF(sklearn.decomposition.NMF):
    pass
