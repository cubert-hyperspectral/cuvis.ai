import os
import yaml
import typing
import shutil
from datetime import datetime
from os.path import expanduser
from typing import Any
from cuvis_ai.preprocessor import *
from cuvis_ai.pipeline import *
from cuvis_ai.unsupervised import *
import networkx as nx
from collections import defaultdict

class Graph():
    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.sorted_graph = None
        self.nodes = {}
        self.entry_point = None
        self.name = name

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
        # TODO this needs a method to verify if the edge is validly constructed
        self.graph.add_edge(node.id, node2.id)
        self.nodes[node.id] = node
        self.nodes[node2.id] = node2
        if not self._verify():
            # Delete nodes and connection
            del self.nodes[node.id]
            del self.nodes[node2.id]
            # Remove the nodes from the graph as a whole
            self.graph.remove_nodes_from([node.id, node2.id])


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
        # Check that all the edges maintain correct constraints
        # Non-empty stage, check all connections are satisfiable
        #for stage in range(len(self.pipeline)-1):
        #    # Check if adjacent elements work
        #    start_elem = self.pipeline[stage]
        #    end_elem = self.pipeline[stage+1]
        #    if start_elem.output_size != end_elem.input_size:
        #        return False
        return True
    
    def delete_node(self, id) -> None:
        '''
        Remove node by its id
        '''
        if self.nodes[id]:
            # Delete the node
            del self.nodes[id]
            # Delete any edges from go to/from that node
            self.graph.remove_edges_from([id])
        else:
            print('Cannot remove node, it no longer exists')

    def forward(self, data):
        # This yields all the nodes in sorted topological order
        self.sorted_graph = nx.topological_sort(self.graph)
        return self.forward_helper(list(self.sorted_graph), data)

    def forward_helper(self, sorted_nodes, data):
        # If we don't have a start point, assume it is the global graph entry point

        # Mark all the vertices as not visited
        # visited = defaultdict(lambda: False)
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
            'stages': [],
            'name': self.name
        }
        now = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        working_dir = f'{expanduser("~")}/{self.name}_{now}'
        os.mkdir(working_dir)
        # Step through all the stages of the pipeline and serialize
        for stage in self.pipeline:
            output['stages'].append(
                yaml.safe_load(stage.serialize(working_dir))
            )
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
        if not structure.get('stages'):
            print('No pipeline information available!')
        for stage in structure.get('stages'):
            self.add_stage(self.reconstruct_stage(stage, root_path))

    def reconstruct_stage(self, data: dict, filepath: str) -> Any:
        stage = globals()[data.get('type')]()
        stage.load(data, filepath)
        return stage