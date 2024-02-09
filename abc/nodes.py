
import numpy as np

class MetaData:
    pass


class Data:
    pass

class Node:
    pass

class PreprocessingNode(Node):



    def __init__(self) -> None:
        super().__init__()


    
    def apply(data: np.array) -> np.array:
        pass


    
    def apply_check(test_data: MetaData) -> MetaData:
        pass


class ClassificationNode(Node):

    def __init__(self) -> None:
        super().__init__()


    def apply(data: np.array) -> np.array:
        pass

        def apply_check(test_data: MetaData) -> MetaData:
        pass



class ImageClassificationNode(Node):
    def __init__(self) -> None:
        super().__init__()


    def apply(data: np.array) -> int:
        pass