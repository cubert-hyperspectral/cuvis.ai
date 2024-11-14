

# from sklearn.decomposition import PCA, NMF
# from sklearn.cluster import KMeans, MeanShift

from cuvis_ai.preprocessor import PCA, NMF
from cuvis_ai.unsupervised import KMeans, GMM, MeanShift
from cuvis_ai.supervised import SVM, LDA, QDA

from cuvis_ai.utils.test import get_np_dummy_data

import numpy as np

TEST_DIR = 'test/tmp'

if __name__ == '__main__':
    testPCA = PCA(n_components=3)

    testNMF = NMF(n_components=3)

    testSVM = SVM()

    testQDA = QDA()
    params = testNMF.serialize(TEST_DIR)

    data = get_np_dummy_data((10, 10, 10, 10))
    labels = np.where(get_np_dummy_data((10, 10, 10)) > 0.5, 1, 0)

    testNMF.fit(data)

    testSVM.fit(data, labels)

    testQDA.fit(data, labels)

    svmParams = testSVM.serialize(TEST_DIR)

    qdaParams = testQDA.serialize(TEST_DIR)

    kmeansInstance = KMeans(n_clusters=3)
