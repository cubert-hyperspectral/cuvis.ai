import networkx as nx
from ..node.node import Node
import numpy as np
from typing import Optional, Union, List, Dict, Tuple
import torch
from ..node.Consumers import LabelConsumerInference, MetadataConsumerInference, CubeConsumer, LabelConsumer, MetadataConsumer


class MemoryExecutor:

    def __init__(self, graph: nx.DiGraph, nodes: dict[str, Node], entry_point: str):
        self.graph = graph
        self.nodes = nodes
        self.entry_point = entry_point
        self.sorted_nodes = list(nx.topological_sort(self.graph))

    def forward(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pass data through the graph by starting at the root node and flowing through all
        intermediary stages.

        Parameters
        ----------
        X : np.ndarray
            Input data
        Y : Optional[Union[np.ndarray, List]], optional
            Label data
        M : Optional[Union[np.ndarray, List]], optional
            Metadata by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Residuals of processed X, Y, and M
        """
        self.sorted_graph = list(nx.topological_sort(self.graph))
        assert (self.sorted_graph[0] == self.entry_point)

        xs = self._flatten_to_4dim(X)
        xs = np.split(xs, indices_or_sections=xs.shape[0], axis=0)
        ys = None
        if Y is not None:
            if (isinstance(Y, List) and isinstance(Y[0], np.ndarray)) or isinstance(Y, np.ndarray):
                ys = self._flatten_to_4dim(Y)
                ys = np.split(ys, indices_or_sections=ys.shape[0], axis=0)
            else:
                ys = Y

        if ys is None:
            ys = [None]*len(xs)
        if M is None:
            ms = [None]*len(xs)
        else:
            ms = M

        results = []
        for x, y, m in zip(xs, ys, ms):

            intermediary = {}
            intermediary_labels = {}
            intermediary_metas = {}
            intermediary[self.entry_point], intermediary_labels[self.entry_point], intermediary_metas[self.entry_point] = self._forward_node(
                self.nodes[self.entry_point], x, y, m)

            for node in self.sorted_graph[1:]:
                self._forward_helper(node, intermediary,
                                     intermediary_labels, intermediary_metas)

            results.append((intermediary[self.sorted_graph[-1]],
                           intermediary_labels[self.sorted_graph[-1]], intermediary_metas[self.sorted_graph[-1]]))
        zr = tuple(zip(*results))
        rxs = zr[0]
        rys = zr[1]
        rms = zr[2]

        rxs = np.concatenate(rxs, axis=0)
        if isinstance(rys[0], np.ndarray):
            rys = np.concatenate(rys, axis=0)

        return (rxs, rys, rms)

    def _forward_helper(self, current: str, intermediary: dict, intermediary_labels: dict, intermediary_metas: dict):
        """Helper function to aggregate inputs and calculate products from a given node.

        Parameters
        ----------
        current : str
            id for current node
        intermediary : dict
            Dictionary containing intermediary products with key as id of node
        intermediary_labels : np.ndarray
            Dictionary containing intermediary labels with key as id of node
        intermediary_metas : np.ndarray
            Dictionary containing intermediary metadata with key as id of node
        """
        p_nodes = list(self.graph.predecessors(current))
        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)

        no_labels = intermediary_labels[p_nodes[0]] is None
        if not no_labels:
            if isinstance(intermediary_labels[p_nodes[0]], np.ndarray):
                use_labels = np.concatenate(
                    [intermediary_labels[p] for p in p_nodes], axis=-1)
            else:
                use_labels = [intermediary_labels[p] for p in p_nodes]
        else:
            use_labels = []

        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = [intermediary_metas[p] for p in p_nodes]
        else:
            use_metas = []

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = self._forward_node(
            self.nodes[current], use_prods, use_labels, use_metas)

        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def _forward_node(self, node: Node, data: np.ndarray, labels: np.ndarray, metadata: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pass data through a node which has already been trained/fit.

        Parameters
        ----------
        node : Node
            Node within the graph
        data : np.ndarray
            Data to pass through the nodes
        labels : np.ndarray
            Labels associated with input data
        metadata : np.ndarray
            Metadata needed for forward pass

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Output data, output labels, output metadata
        """
        node_input = [data]

        if isinstance(node, LabelConsumerInference):
            node_input.append(labels)
        if isinstance(node, MetadataConsumerInference):
            node_input.append(metadata)

        if len(node_input) == 1:
            out = node.forward(node_input[0])
        else:
            out = node.forward(tuple(node_input))
        if isinstance(out, Tuple):
            return out
        else:
            return out, labels, metadata

    def _not_needed_anymore(self, id: str, intermediary: list[Node]) -> bool:
        """Private function to determine if a node products are still needed or can be safely removed.

        Parameters
        ----------
        id : str
            Alphanumeric identifier for node to check
        intermediary : list[Node]
            List node nodes for which the current node's data is an intermediary

        Returns
        -------
        bool
           If all successors are already present in intermediary, it will return True
        """
        return all([succs in intermediary for succs in self.graph.successors(id)]) and \
            len(list(self.graph.successors(id))
                ) > 0  # Do not remove a terminal nodes data

    def train(self, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader):
        """Train a graph use a dataloader to iteratively pass data through the graph

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training dataloader
        test_dataloader : torch.utils.data.DataLoader
            Test dataloader

        Raises
        ------
        TypeError
            Raises error if dataloaders passed to train function are not pytorch dataloaders
        """
        if not isinstance(train_dataloader, torch.utils.data.DataLoader) or not isinstance(test_dataloader, torch.utils.data.DataLoader):
            raise TypeError(
                "train or test dataloader argument is not a pytorch DataLoader!")

        xs = []
        ys = []
        ms = []
        for x, y, m in iter(train_dataloader):
            xs.append(x)
            ys.append(y)
            ms.append(m)

        xs = self._flatten_to_4dim(xs)

        if isinstance(ys[0], np.ndarray):
            ys = self._flatten_to_4dim(ys)

        self.fit(xs, ys, ms)

        # test stage
        for x, y, m in iter(test_dataloader):
            test_results = self.forward(x, y, m)
            # do some metrics

    def fit(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None):
        """Take a graph of uninitialized nodes and fit then given a set of inputs and outputs

        Parameters
        ----------
        X : np.ndarray
            Input data
        Y : Optional[Union[np.ndarray, List]], optional
            Input labels, by default None
        M : Optional[Union[np.ndarray, List]], optional
            Input metadata, by default None
        """
        # training stage
        self.sorted_graph = list(nx.topological_sort(self.graph))
        assert (self.sorted_graph[0] == self.entry_point)

        intermediary = {}
        intermediary_labels = {}
        intermediary_metas = {}

        result = self._fit_node(self.nodes[self.entry_point], X, Y, M)

        intermediary[self.entry_point], intermediary_labels[self.entry_point], intermediary_metas[self.entry_point] = self._fit_node(
            self.nodes[self.entry_point], X, Y, M)

        for node in self.sorted_graph[1:]:
            self._fit_helper(node, intermediary,
                             intermediary_labels, intermediary_metas)

    def _fit_helper(self, current: str, intermediary: dict, intermediary_labels: dict, intermediary_metas: dict):
        """Private helper function to fit an individual node.

        Parameters
        ----------
        current : str
            id of current node in graph
        intermediary : str
            Dictionary containing intermediary products
        intermediary_labels : np.ndarray
            Dictionary containing intermediary labels
        intermediary_metas : np.ndarray
            Dictionary containing intermediary metadata
        """
        p_nodes = list(self.graph.predecessors(current))

        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)

        no_labels = intermediary_labels[p_nodes[0]] is None
        if not no_labels:
            use_labels = np.concatenate(
                [intermediary_labels[p] for p in p_nodes], axis=-1)
        else:
            use_labels = None

        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = np.concatenate(
                [intermediary_metas[p] for p in p_nodes], axis=-1)
        else:
            use_metas = None

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = self._fit_node(
            self.nodes[current], use_prods, use_labels, use_metas)

        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def _fit_node(self, node: Node, data: np.ndarray, labels: np.ndarray, metadata: np.ndarray) -> np.ndarray:
        """Private function wrapper to call the fit function for an individual node

        Parameters
        ----------
        node : Node
            Graph node that will be fit
        data : np.ndarray
            Training data
        labels : np.ndarray
            Training labels
        metadata : np.ndarray
            Training metadata

        Returns
        -------
        np.ndarray
            Results of passing data through the fit node

        Raises
        ------
        RuntimeError
            Data is empty (length 0)
        """
        node_input = []

        if isinstance(node, CubeConsumer):
            node_input.append(data)
        if isinstance(node, LabelConsumer):
            node_input.append(labels)
        if isinstance(node, MetadataConsumer):
            node_input.append(metadata)

        if len(node_input) == 0:
            raise RuntimeError(
                F"Node {node} invalid, does not indicate input data type!")
        elif len(node_input) == 1:
            node.fit(node_input[0])
        else:
            node.fit(*node_input)

        return self._forward_node(node, data, labels, metadata)

    @staticmethod
    def _flatten_to_4dim(x: list | np.ndarray) -> np.ndarray:
        """Private method to flatten

        Parameters
        ----------
        x : list | np.ndarray
            Make sure all array like data has 4 dimensions

        Returns
        -------
        np.ndarray
            Flattened array
        """
        if isinstance(x, List):
            if len(x[0].shape) == 5:
                x = np.concatenate(x, axis=0)
            else:
                x = np.stack(x, axis=0)
        while len(x.shape) >= 5:
            x = x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))
        return x