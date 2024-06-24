import yaml
import numpy as np
import pickle as pk
import os
from pathlib import Path


class Serializer:

    def __init__(self, serial_dir, *, pickle_inner=True) -> None:
        self.serial_dir = Path(serial_dir)
        self.pickle_inner = pickle_inner

    def serialize_node(self, node, name, *, inner_module: str, params_list: list[str]) -> str:

        pickle_name = f"{hash(node.__dict__[inner_module])}_{name}.pkl"
        if self.pickle_inner:
            pk.dump(node.__dict__[inner_module], open(self.serial_dir /
                    pickle_name, "wb"))

        data = {
            'type': type(node).__name__,
            'id': node.id
        }
        if self.pickle_inner:
            data[f'{name}_object'] = pickle_name

        additional_params = {p: node.__dict__[p] for p in params_list}
        data = data | additional_params
        return yaml.dump(data, default_flow_style=False)

    def load_node(self, node, name, *, params: dict, inner_module: str, params_list: list[str]):

        node.id = params['id']

        for p in params_list:
            node.__dict__[p] = params[p]

        if self.pickle_inner:
            pickle_name = params[f'{name}_object']
            node.__dict__[inner_module] = pk.load(
                open(self.serial_dir / pickle_name, 'rb'))

        node.initialized = True
