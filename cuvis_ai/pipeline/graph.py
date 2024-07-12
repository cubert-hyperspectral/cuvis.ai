import os
import yaml
import typing
import shutil
import torch
from typing import Any
from datetime import datetime
from os.path import expanduser
from typing import Optional, Dict, List, Tuple, Any, Union
from cuvis_ai.preprocessor import *
from cuvis_ai.pipeline import *
from cuvis_ai.unsupervised import *
from cuvis_ai.supervised import *
from cuvis_ai.distance import *
from cuvis_ai.deciders import *
import networkx as nx
from typing import List, Union
from collections import defaultdict
import pkg_resources  # part of setuptools
from ..node import Node
from ..node.Consumers import *
from ..data.OutputFormat import OutputFormat
from ..utils.numpy_utils import get_shape_without_batch, check_array_shape


class Graph():
    """Main class for connecting nodes in a CUVIS.AI processing graph
    """
    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.sorted_graph = None
        self.nodes: dict[str,Node] = {}
        self.entry_point = None
        self.name = name


    def add_node(self, node: Node, parent: list[Node] | Node = None) -> None:
        """Add a new node into the graph structure

        Parameters
        ----------
        node : Node
            CUVIS.AI type node
        parent : list[Node] | Node, optional
           Node(s) that the child node should be connected to,
           with data flowing from parent(s) to child, by default None.

        Raises
        ------
        ValueError
            If no parent is provided, node is assumed to be the base node of the graph.
            This event will raise an error to prevent base from being overwritten.
        ValueError
            If parent(s) do not already belong to the graph.
        ValueError
            If parent(s) and child nodes are mismatched in expected data size.
            
        """
        if parent is None:
            # this is the first Node of the graph
            if self.entry_point is not None:
                raise ValueError("Graph already has base node")
            self.entry_point = node.id
            parent = []

        if isinstance(parent, Node):
            parent = [parent]

        # Check if operation is valid
        if not all([self.graph.has_node(p.id) for p in parent]):
            raise ValueError("Not all parents are part of the Graph")
        
        if not all([check_array_shape(p.output_dim, node.input_dim) for p in parent]):
            raise ValueError('Unsatisfied dimensionality constraint!')

        self.graph.add_node(node.id)

        for p in parent:
            self.graph.add_edge(p.id, node.id)

        self.nodes[node.id] = node

        # Remove if verify fails
        if not self._verify():
            self.delete_node(node)

    def add_base_node(self, node: Node) -> None:
        """Adds new node into the graph by creating the first entry point.

        Parameters
        ----------
        node : Node
            CUVIS.AI node to add to the graph
        """
        self.graph.add_node(node.id)
        self.nodes[node.id] = node
        self.entry_point = node.id

    def add_edge(self, node: Node, node2: Node) -> None:
        """Adds sequential nodes to create a directed edge.
        At least one of the nodes should already be in the graph.

        Parameters
        ----------
        node : Node
            Parent node.
        node2 : Node
            Child node.
        """
        self.graph.add_edge(node.id, node2.id)
        self.nodes[node.id] = node
        self.nodes[node2.id] = node2
        if not self._verify():
            # TODO Issue: This could potentially leave the graph in an invalid state 
            # Delete nodes and connection
            del self.nodes[node.id]
            del self.nodes[node2.id]
            # Remove the nodes from the graph as a whole
            self.graph.remove_nodes_from([node.id, node2.id])

    def _verify_input_outputs(self) -> bool:
        """Private function to validate the integrity of data passed between nodes.

        Returns
        -------
        bool
            Inputs and outputs of all nodes are congruent.
        """
        all_edges = list(self.graph.edges)
        for start, end in all_edges:
            # TODO: Issue what if multiple Nodes feed into the same successor Node, how would the shape look like?
            if not check_array_shape(self.nodes[start].output_dim, self.nodes[end].input_dim):
                print('Unsatisfied dimensionality constraint!')
                return False
        return True

    def _verify(self) -> bool:
        """Private function to verify the integrity of the processing graph.

        Returns
        -------
        bool
            Graph meets/does not meet requirements for ordered and error-free flow of data.
        """
        if len(self.nodes.keys()) == 0:
            print('Empty graph!')
            return True
        elif len(self.nodes.keys()) == 1:
            print('Single stage graph!')
            return True
        # Check that no cycles exist
        if len(list(nx.simple_cycles(self.graph))) > 0:
            return False
        # Get all edges in the graph
        if not self._verify_input_outputs():
            return False
        
        return True
    
    def delete_node(self, id: Node | str) -> None:
        """Removes a node from the graph.
        To successfully remove a node, it must not have successors.


        Parameters
        ----------
        id : Node | str
            UUID for target node to delete, or a copy of the node itself.

        Raises
        ------
        ValueError
            Node to delete contains successors in the graph.
        ValueError
            Node does not exist in the graph.
        """
        if isinstance(id, Node):
            id = id.id

        # Check if operation is valid
        if not len(list(self.graph.successors(id))) == 0:
            raise ValueError("The node does have successors, removing it would invalidate the Graph structure")

        if not id in self.nodes:
            raise ValueError("Cannot remove node, it no longer exists")
        
        self.graph.remove_edges_from([id])
        del self.nodes[id]

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
        assert(self.sorted_graph[0] == self.entry_point)
        
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
            intermediary[self.entry_point], intermediary_labels[self.entry_point], intermediary_metas[self.entry_point] = self._forward_node(self.nodes[self.entry_point], x, y, m)

            for node in self.sorted_graph[1:]:
                self._forward_helper(node, intermediary, intermediary_labels, intermediary_metas)

            results.append((intermediary[self.sorted_graph[-1]], intermediary_labels[self.sorted_graph[-1]], intermediary_metas[self.sorted_graph[-1]]))
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
                use_labels = np.concatenate([intermediary_labels[p] for p in p_nodes], axis=-1)
            else:
                use_labels = [intermediary_labels[p] for p in p_nodes]
        else:
            use_labels = []
        
        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = [intermediary_metas[p] for p in p_nodes]
        else:
            use_metas = []

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = self._forward_node(self.nodes[current], use_prods, use_labels, use_metas)
        
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
            len(list(self.graph.successors(id))) > 0 # Do not remove a terminal nodes data

    def train(self, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader):
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
            raise TypeError("train or test dataloader argument is not a pytorch DataLoader!")
        
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
        assert(self.sorted_graph[0] == self.entry_point)

        intermediary = {}
        intermediary_labels = {}
        intermediary_metas = {}
        
        result = self._fit_node(self.nodes[self.entry_point], X, Y, M)
        
        
        intermediary[self.entry_point], intermediary_labels[self.entry_point], intermediary_metas[self.entry_point] = self._fit_node(self.nodes[self.entry_point], X, Y, M)

        for node in self.sorted_graph[1:]:
            self._fit_helper(node, intermediary, intermediary_labels, intermediary_metas)
        # do some metrics

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
            use_labels = np.concatenate([intermediary_labels[p] for p in p_nodes], axis=-1)
        else:
            use_labels = None
        
        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = np.concatenate([intermediary_metas[p] for p in p_nodes], axis=-1)
        else:
            use_metas = None

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = self._fit_node(self.nodes[current], use_prods, use_labels, use_metas)
        
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
            raise RuntimeError(F"Node {node} invalid, does not indicate input data type!")
        elif len(node_input) == 1:
            node.fit(node_input[0])
        else:
            node.fit(tuple(node_input))
        
        return self._forward_node(node, data, labels, metadata)

    def serialize(self) -> None:
        """Convert graph structure and all contained nodes to a serializable YAML format.
        Numeric data and fit models will be stored in zipped directory named with current time.
        """

        from cuvis_ai import __version__
        output = {
            'edges': [],
            'nodes': [],
            'name': self.name,
            'entry_point': self.entry_point,
            'version': __version__
        }
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        working_dir = f'{expanduser("~")}/{self.name}_{now}'
        os.mkdir(working_dir)
        # Step through all the stages of the pipeline and serialize
        for node in self.nodes.values():
            output['nodes'].append(
                yaml.full_load(node.serialize(working_dir))
            )
        # Grab the connections and write as plain text
        output['edges'] = [list(z) for z in list(self.graph.edges)]
        # Create main .yml file
        with open(f'{working_dir}/main.yml', 'w') as f:
            f.write(yaml.dump(output, default_flow_style=False))
        # Create a portable zip archive
        shutil.make_archive(f'{expanduser("~")}/{self.name}_{now}', 'zip', working_dir)
        print(f'Project saved to ~/{self.name}_{now}.zip')
    
    def load(self, filepath: str) -> None:
        """Reconstruct the graph from a file path defining the location of a zip archive.

        Parameters
        ----------
        filepath : str
            Location of zip archive
        """
        self.now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        shutil.unpack_archive(filepath, f'/tmp/cuvis_{self.now}')
        # Read the pipeline structure from the location
        # Create main .yml file
        self.pipeline = []
        self.reconstruct_from_yaml(f'/tmp/cuvis_{self.now}')

    def reconstruct_from_yaml(self, root_path: str) -> None:
        """_summary_

        Parameters
        ----------
        root_path : str
            Path to unzipped graph archive directory.

        Raises
        ------
        Exception
            Currently installed version of CUVIS.AI does not match version model was saved with
        """
        with open(f'{root_path}/main.yml') as f:
            structure = yaml.safe_load(f)
        # We now have a dictionary defining the pipeline
        self.name = structure.get('name')
        # Check the version of serialization matches currently installed version
        print(structure)
        if pkg_resources.require('cuvis_ai')[0].version != structure.get('version'):
            print(pkg_resources.require('cuvis_ai')[0].version)
            print(structure.get('version'))
            raise Exception('Incorrect version of cuvis_ai package')
        if not structure.get('nodes'):
            print('No node information available!')
        for stage in structure.get('nodes'):
            t = self.reconstruct_stage(stage, root_path)
            self.nodes[t.id] = t
        # Set the entry point
        self.entry_point = structure.get('entry_point')
        # Create the graph instance
        self.graph = nx.DiGraph()
        # Handle base case where there is only one node
        if len(structure.get('nodes')) > 1:
            # Graph has at least one valid edge
            self.graph.add_edges_from(structure.get('edges'))
        else:
            # Only single node exists, add it into the graph
            self.add_base_node(list(self.nodes.values())[0])

    def reconstruct_stage(self, data: dict, filepath: str) -> Node:
        """Function to rebuild each node in the graph.

        Parameters
        ----------
        data : dict
            Parameters defining the values to initialize the node
        filepath : str
            Path to any associated models or numeric data to initialize the node

        Returns
        -------
        Node
            Fully initialized node
        """
        stage = globals()[data.get('type')]()
        stage.load(data, filepath)
        return stage

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