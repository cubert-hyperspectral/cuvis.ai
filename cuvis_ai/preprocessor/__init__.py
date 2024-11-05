# from .pca import *
# from .nmf import *

import sklearn.decomposition
from ..node.wrap import node
import sklearn


@node
class PCA(sklearn.decomposition.PCA):
    pass


@node
class NMF(sklearn.decomposition.NMF):
    pass
