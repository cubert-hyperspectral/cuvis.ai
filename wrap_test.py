from cuvis_ai.node.wrap import node

from sklearn.decomposition import PCA, NMF

from cuvis_ai.utils.test import get_np_dummy_data

import numpy as np

WrappedPCA = node(PCA)

WrappedNMF = node(NMF)

if __name__ == '__main__':
    testPCA = PCA(n_components=3)

    testNMF = NMF(n_components=3)
    params = testPCA.get_params()

    wrappedPCAInstance = node(testPCA)

    wrappedPCAInstance2 = WrappedPCA(n_components=3)

    data = get_np_dummy_data((10, 10))

    moreData = get_np_dummy_data((10, 10, 10, 10))

    wrappedPCAInstance2.fit(moreData)

    testPCA.fit(data)

    testNMF.fit(data)

    serializedData = wrappedPCAInstance2.serialize()

    newPCA = WrappedPCA(n_components=3)

    newPCA.load(serializedData)

    oldTransformed = wrappedPCAInstance2.forward(moreData)

    newTransformed = newPCA.forward(moreData)

    is_same = np.all(oldTransformed == newTransformed)

    print(params)
