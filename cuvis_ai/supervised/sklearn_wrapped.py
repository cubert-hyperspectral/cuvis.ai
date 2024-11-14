from ..node.wrap import make_node
import sklearn.discriminant_analysis
import sklearn.svm
import sklearn.neural_network


@make_node
class LDA(sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
    pass


@make_node
class QDA(sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis):
    pass


@make_node
class SVM(sklearn.svm.SVC):
    pass


@make_node
class MLP(sklearn.neural_network.MLPClassifier):
    pass
