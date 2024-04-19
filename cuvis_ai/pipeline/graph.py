import os
import yaml
import typing
from typing import List, Dict, Any
import shutil
from datetime import datetime
from os.path import expanduser
from typing import Any, Optional
from cuvis_ai.preprocessor import *
from cuvis_ai.pipeline import *
from cuvis_ai.unsupervised import *
from cuvis_ai.supervised import *
import networkx as nx
from collections import defaultdict
import pkg_resources  # part of setuptools
from ..node import Node

class Graph():
    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.sorted_graph = None
        self.nodes = {}
        self.entry_point = None
        self.name = name


    def add_node(self, node: Node, parent: Optional[list[Node] | Node]):
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

        self.graph.add_node(node.id)

        for p in parent:
            self.graph.add_edge(p.id, node.id)

    def add_base_node(self, node: Any) -> None:
        '''
        Adds new node into the network

        This function creates the first entry
        '''
        self.graph.add_node(node.id)
        self.nodes[node.id] = node
        self.entry_point = node.id

    def add_edge(self, node: Any, node2: Any) -> None:
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
        for edge in all_edges:
            # TODO: Issue what if multiple Nodes feed into the same sucessor Node, how would the shape look like
            if self.nodes[edge[0]].output_size != self.nodes[edge[1]].input_size:
                print('Unsatisfied dimensionality constraint!')
                return False
        # If we succeed in reaching this stage, all constraints are satisfied
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
        Remove node by its id
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

    def forward(self, data):
        # This yields all the nodes in sorted topological order
        self.sorted_graph = nx.topological_sort(self.graph)
        return self.forward_helper(list(self.sorted_graph), data)

    def forward_helper(self, sorted_nodes, data):
        intermediary_products = {}
        # Calculate the first forward pass
        start_node = sorted_nodes[0]
        intermediary_products[start_node] = self.nodes[start_node].forward(data)
        for node in sorted_nodes[1:]:
            # Grab the intermediary products for the node
            p_nodes = self.graph.predecessors(node)
            # Take the intermediary results and pass forward
            use_prods = [intermediary_products[p] for p in p_nodes]
            intermediary_products[node] = self.nodes[node].forward(*use_prods)
        return intermediary_products            
        
    def train(self, train_dataloader, test_dataloader):

        x, y = zip(*[train_dataloader[i] for i in range(0,10)])
        x = np.array(x)
        y = np.array(y)

        # training stage
        for stage in self.pipeline:

            if isinstance(stage,BaseUnsupervised) or isinstance(stage,Preprocessor):
                stage.fit(x)
            elif isinstance(stage,BaseSupervised):
                stage.fit(x,y)
            else:
                raise NotImplementedError("Invalid class type")

            x = stage.forward(x)

        # test stage
        test_x, test_y = zip(*[train_dataloader[i] for i in range(10,20)])
        test_x = np.array(test_x)
        

        for stage in self.pipeline:
            test_x = stage.forward(test_x)

        # do some metrics



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

    def reconstruct_stage(self, data: dict, filepath: str) -> Any:
        stage = globals()[data.get('type')]()
        stage.load(data, filepath)
        return stage