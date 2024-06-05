import os
import yaml
import typing
import shutil
import torch
from datetime import datetime
from os.path import expanduser
from typing import Optional, Dict, List, Tuple, Any, Union
from cuvis_ai.preprocessor import *
from cuvis_ai.pipeline import *
from cuvis_ai.unsupervised import *
from cuvis_ai.supervised import *
import networkx as nx
from collections import defaultdict
import pkg_resources  # part of setuptools
from ..node import Node
from ..node.Consumers import *
from ..data.OutputFormat import OutputFormat
from ..utils.numpy_utils import get_shape_without_batch, check_array_shape


class Graph():
    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.sorted_graph = None
        self.nodes: dict[str,Node] = {}
        self.entry_point = None
        self.name = name


    def add_node(self, node: Node, parent: Optional[list[Node] | Node] = None):
        '''
        Alternative proposal to add Nodes to the Network
        '''
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
        '''
        Adds new node into the network

        This function creates the first entry
        '''
        self.graph.add_node(node.id)
        self.nodes[node.id] = node
        self.entry_point = node.id

    def add_edge(self, node: Node, node2: Node) -> None:
        '''
        Adds sequential nodes to create a directed edge
        '''
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
        all_edges = list(self.graph.edges)
        for start, end in all_edges:
            # TODO: Issue what if multiple Nodes feed into the same sucessor Node, how would the shape look like
            if not check_array_shape(self.nodes[start].output_dim, self.nodes[end].input_dim):
                print('Unsatisfied dimensionality constraint!')
                return False
        return True

    def _verify(self) -> bool:
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
        '''
        Removes a node by its id. To Sucessfully remove a node it need to have no sucessors.
        '''
        if isinstance(id, Node):
            id = id.id

        # Check if operation is valid
        if not len(list(self.graph.successors(id))) == 0:
            raise ValueError("The node does have successors, removing it would invalidate the Graph structure")

        if not id in self.nodes:
            raise ValueError("Cannot remove node, it no longer exists")
        
        self.graph.remove_edges_from([id])
        del self.nodes[id]

    def forward(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None):
        self.sorted_graph = list(nx.topological_sort(self.graph))
        assert(self.sorted_graph[0] == self.entry_point)

        intermediary = {}
        intermediary_labels = {}
        intermediary_metas = {}
        intermediary[self.entry_point], intermediary_labels[self.entry_point], intermediary_metas[self.entry_point] = self._forward_node(self.nodes[self.entry_point], X, Y, M)

        for node in self.sorted_graph[1:]:
            self._forward_helper(node, intermediary, intermediary_labels, intermediary_metas)

        return intermediary[self.sorted_graph[-1]]

    def _forward_helper(self, current, intermediary, intermediary_labels, intermediary_metas):
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

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = self._forward_node(self.nodes[current], use_prods, use_labels, use_metas)
        
        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)
    

    def _forward_node(self, node: Node, data, labels, metadata):
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
    
    def _not_needed_anymore(self, id: str, intermediary) -> bool:
        '''
        Checks if the intermediary results of a node are needed again,
        if all successors are already present in intermediary, it will return True
        '''
        return all([succs in intermediary for succs in self.graph.successors(id)]) and \
            len(list(self.graph.successors(id))) > 0 # Do not remove a terminal nodes data

    def train(self, train_dataloader:torch.utils.data.DataLoader, test_dataloader:torch.utils.data.DataLoader):
        if not isinstance(train_dataloader, torch.utils.data.DataLoader) or not isinstance(test_dataloader, torch.utils.data.DataLoader):
            raise TypeError("train or test dataloader argument is not a pytorch DataLoader!")

        x, y, m = next(iter(train_dataloader))
        
        self.fit(x, y, m)

        # test stage
        x, y, m = next(iter(train_dataloader))

        test_results = self.forward(x, y, m)
        # do some metrics

    def fit(self, X: np.ndarray, Y: Optional[Union[np.ndarray, List]] = None, M: Optional[Union[np.ndarray, List]] = None):
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

    def _fit_helper(self, current, intermediary, intermediary_labels, intermediary_metas):
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

    def _fit_node(self, node: Node, data, labels, metadata):
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
        output = {
            'edges': [],
            'nodes': [],
            'name': self.name,
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
        self.now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        shutil.unpack_archive(filepath, f'/tmp/cuvis_{self.now}')
        # Read the pipeline structure from the location
        # Create main .yml file
        self.pipeline = []
        self.reconstruct_from_yaml(f'/tmp/cuvis_{self.now}')

    def reconstruct_from_yaml(self, root_path: str) -> None:
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
        # Create the graph instance
        self.graph = nx.DiGraph()
        self.graph.add_edges_from(structure.get('edges'))

    def reconstruct_stage(self, data: dict, filepath: str) -> Node:
        stage = globals()[data.get('type')]()
        stage.load(data, filepath)
        return stage