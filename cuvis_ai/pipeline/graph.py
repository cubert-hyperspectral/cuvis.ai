import os
import yaml
import typing
import shutil
from typing import Any
from datetime import datetime
from os.path import expanduser
from typing import Optional
from cuvis_ai.preprocessor import *
from cuvis_ai.pipeline import *
from cuvis_ai.unsupervised import *
from cuvis_ai.supervised import *
from cuvis_ai.distance import *
from cuvis_ai.deciders import *
import networkx as nx
from collections import defaultdict
import pkg_resources  # part of setuptools
from ..node import Node
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

    def forward(self, data: np.ndarray) -> Any:
        """Pass data through the graph by starting at the root node and flowing through all
        intermediary stages.

        Parameters
        ----------
        data : np.ndarray
            Hyperspectral data to pass through all processing stages.

        Returns
        -------
        Any
            Returns the output from the leaf nodes of the graph.
        """
        self.sorted_graph = list(nx.topological_sort(self.graph))
        assert(self.sorted_graph[0] == self.entry_point)

        intermediary = {}
        intermediary[self.entry_point] = self._forward_node(self.nodes[self.entry_point], data)

        for node in self.sorted_graph[1:]:
            self._forward_helper(node, intermediary)

        return intermediary[self.sorted_graph[-1]]

    def _forward_helper(self, current: str, intermediary: list) -> None:
        """Pass data through the current nodes

        Parameters
        ----------
        current : str
            Current node in processing graph.
        intermediary : list
            List of node ids that define which nodes generate data for the current node
        """
        p_nodes = self.graph.predecessors(current)
        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)

        intermediary[current] = self._forward_node(self.nodes[current], use_prods)
        
        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            del intermediary[current]

    def _forward_node(self, node: Node, data: np.ndarray) -> Any:
        """Private wrapper to call the forward method on a graph node.

        Parameters
        ----------
        node : Node
            Node which will apply some operation to the data.
        data : np.ndarray
            Data that will be transformed by the node.

        Returns
        -------
        Any
            Given the variety of nodes, the return type may vary, but will generally be an np.ndarray.
        """
        return node.forward(data)
    
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

    def fit(self, X: np.ndarray, Y: Optional[np.ndarray] = None):
        """For all nodes on the graph apply the training data to fit the nodes which require training.

        Parameters
        ----------
        X : np.ndarray
            Input data
        Y : Optional[np.ndarray], optional
            Labels or records corresponding to input data, by default None
        """
        # training stage
        self.sorted_graph = list(nx.topological_sort(self.graph))
        assert(self.sorted_graph[0] == self.entry_point)

        intermediary = {}
        intermediary_labels = {}
        intermediary[self.entry_point], intermediary_labels[self.entry_point] = self._fit_node(self.nodes[self.entry_point], X,Y)

        for node in self.sorted_graph[1:]:
            self._fit_helper(node, intermediary, intermediary_labels)
        # do some metrics


        
    def train(self, train_dataloader: Any, test_dataloader: Any):
        """Similar to fit method but works with dataloader

        Parameters
        ----------
        train_dataloader : Any
            Dataloader for training data
        test_dataloader : Any
            Dataloader for testing data
        """
        x, y = zip(*[train_dataloader[i] for i in range(0,10)])
        x = np.array(x)
        y = np.array(y)

        self.fit(x,y)

        # test stage
        test_x, test_y = zip(*[train_dataloader[i] for i in range(10,20)])
        test_x = np.array(test_x)

        test_results = self.forward(test_x)
        # do some metrics

    def _fit_helper(self, current: str, intermediary: dict, intermediary_labels: dict):
        """Fit the node and consider the products from other nodes that are needed to fit it.

        Parameters
        ----------
        current : str
            ID of the current node to fit
        intermediary : dict
            Dict of intermediary products keyed by id
        intermediary_labels : dict
            Labels associated with the intermediary products
        """
        p_nodes = list(self.graph.predecessors(current))

        no_labels = intermediary_labels[p_nodes[0]] is None
        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)
        if not no_labels:
            use_labels = np.concatenate([intermediary_labels[p] for p in p_nodes], axis=-1)
        else:
            use_labels = None

        intermediary[current], intermediary_labels[current] = self._fit_node(self.nodes[current], use_prods, use_labels)
        
        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            del intermediary[current]
            del intermediary_labels[current]

    def _fit_node(self, node: Node, input: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit a node of the graph.

        Parameters
        ----------
        node : Node
            Node object to fit
        input : np.ndarray
            Training data
        labels : np.ndarray
            Labels to evaluate performance against

        Returns
        -------
        np.ndarray
            Output from passing data through the fit node

        Raises
        ------
        NotImplementedError
           Node does not inherit from one of the predefined base classes
        """
        if isinstance(node,BaseUnsupervised) or isinstance(node,Preprocessor):
            node.fit(input)
        elif isinstance(node,BaseSupervised):
            node.fit(input,labels)
        else:
            raise NotImplementedError("Invalid class type")
        
        return node.forward(input), labels


    def serialize(self) -> None:
        """Convert graph structure and all contained nodes to a serializable YAML format.
        Numeric data and fit models will be stored in zipped directory named with current time.
        """
        output = {
            'edges': [],
            'nodes': [],
            'name': self.name,
            'entry_point': self.entry_point,
            'version': pkg_resources.require('cuvis_ai')[0].version
        }
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        working_dir = f'{expanduser("~")}/{self.name}_{now}'
        os.mkdir(working_dir)
        # Step through all the stages of the pipeline and serialize
        for node in self.nodes.values():
            output['nodes'].append(
                yaml.safe_load(node.serialize(working_dir))
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