import os
import shutil
import torch
from typing import Any
from datetime import datetime
from os.path import expanduser
from typing import Optional, Dict, List, Tuple, Any, Union
import networkx as nx
from typing import List, Union
from collections import defaultdict
import pkg_resources  # part of setuptools
from ..node import Node
from ..node.wrap import make_node
from ..node.Consumers import *
from ..data.OutputFormat import OutputFormat
from ..utils.numpy import get_shape_without_batch, check_array_shape
from ..utils.filesystem import change_working_dir
from ..utils.serializer import YamlSerializer
from ..utils.dependencies import get_installed_packages_str
import numpy as np
import tempfile
from pathlib import Path
from importlib import import_module
from .executor import MemoryExecutor


class Graph():
    """Main class for connecting nodes in a CUVIS.AI processing graph
    """

    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.sorted_graph = None
        self.nodes: dict[str, Node] = {}
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
                # TODO reenable this, for now skip
                print('Unsatisfied dimensionality constraint!')
                # return True
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
            raise ValueError(
                "The node does have successors, removing it would invalidate the Graph structure")

        if not id in self.nodes:
            raise ValueError("Cannot remove node, it no longer exists")

        self.graph.remove_edges_from([id])
        del self.nodes[id]

    def serialize(self, data_dir: Path) -> dict:
        """Convert graph structure and all contained nodes to a serializable YAML format.
        Numeric data and fit models will be stored in zipped directory named with current time.
        """
        from importlib.metadata import version

        nodes_data = {
            key: {'__node_module__': str(node.__module__),
                  '__node_class__': str(node.__class__.__name__),
                  **node.serialize(data_dir)}
            for key, node in self.nodes.items()}
        edges_data = [{'from': start, 'to': end}
                      for start, end in list(self.graph.edges)]

        output = {
            'edges': edges_data,
            'nodes': nodes_data,
            'name': self.name,
            'entry_point': self.entry_point,
            'version': version('cuvis_ai'),
            'packages': get_installed_packages_str()
        }

        return output

    def load(self, structure: dict, data_dir: Path) -> None:
        self.name = structure.get('name')

        installed_cuvis_version = pkg_resources.require('cuvis_ai')[0].version
        serialized_cuvis_version = structure.get('version')

        if installed_cuvis_version != serialized_cuvis_version:
            raise ValueError(f'Incorrect version of cuvis_ai package. Installed {installed_cuvis_version} but serialized with {serialized_cuvis_version}')  # nopep8
        if not structure.get('nodes'):
            print('No node information available!')
        for key, params in structure.get('nodes').items():

            node_module = params.get('__node_module__')
            node_class = params.get('__node_class__')

            cls = getattr(import_module(node_module), node_class)
            if not issubclass(cls, Node):
                cls = make_node(cls)
            stage = cls()
            stage.load(params, data_dir)
            self.nodes[key] = stage

        # Set the entry point
        self.entry_point = structure.get('entry_point')
        # Create the graph instance
        self.graph = nx.DiGraph()
        # Handle base case where there is only one node
        if len(structure.get('nodes')) > 1:
            # Graph has at least one valid edge
            for edge in structure.get('edges'):
                self.graph.add_edge(edge.get('from'), edge.get('to'))
        else:
            # Only single node exists, add it into the graph
            self.add_base_node(list(self.nodes.values())[0])

    def save_to_file(self, filepath) -> None:
        filepath = Path(filepath)

        os.makedirs(filepath.parent, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpDir:
            with change_working_dir(tmpDir):
                graph_data = self.serialize('.')

                serial = YamlSerializer(tmpDir, 'main')
                serial.serialize(graph_data)

            shutil.make_archive(
                f'{str(filepath)}', 'zip', tmpDir)
            print(f'Project saved to {str(filepath)}')

    @classmethod
    def load_from_file(cls, filepath: str) -> None:
        """Reconstruct the graph from a file path defining the location of a zip archive.

        Parameters
        ----------
        filepath : str
            Location of zip archive
        """
        new_graph = cls('Loaded')
        with tempfile.TemporaryDirectory() as tmpDir:
            shutil.unpack_archive(filepath, tmpDir)

            with change_working_dir(tmpDir):
                serial = YamlSerializer(tmpDir, 'main')
                graph_data = serial.load()

                new_graph.load(graph_data, '.')

    def forward(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None, backend: str = 'memory') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if backend == 'memory':
            executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        elif backend == 'hummingbird':
        else:
            raise ValueError("Unknown Backend")
        return executor.forward(X, Y, M)

    def fit(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None):
        executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        executor.fit(X, Y, M)
