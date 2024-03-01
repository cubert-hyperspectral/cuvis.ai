
import numpy as np

class MetaData:
    pass


class Data:
    pass

class Node:
    '''
    Base Node Class

    '''
    pass

class PreprocessingNode(Node):
    """This represents a Preprocessing Nodes, example use case Dimensionality reduction

    """


    def __init__(self) -> None:
        super().__init__()


    
    def apply(data: np.array) -> np.array:
        pass


    
    def apply_check(test_data: MetaData) -> MetaData:
        pass


class ClassificationNode(Node):
    """Classification Node, used to classify an input Image pixel by Pixel

    """

    def __init__(self) -> None:
        super().__init__()


    def apply(data: np.array) -> np.array:
        pass

    def apply_check(test_data: MetaData) -> MetaData:
        pass



class ImageClassificationNode(Node):
    """Image Classification Node, used to classify an input Image and a assign a whole class to the image

    """
    def __init__(self) -> None:
        super().__init__()


    def apply(data: np.array) -> int:
        pass