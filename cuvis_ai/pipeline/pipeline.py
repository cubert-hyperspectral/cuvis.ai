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
from cuvis_ai.supervised import *

class Pipeline():
    def __init__(self, name: str) -> None:
        self.pipeline = []
        self.name = name
        
    def _verify(self) -> bool:
        if len(self.pipeline) == 0:
            print('Empty pipeline!')
            return True
        elif len(self.pipeline) == 1:
            print('Single stage pipeline!')
            return True
        # Non-empty stage
        for stage in range(len(self.pipeline)-1):
            # Check if adjacent elements work
            start_elem = self.pipeline[stage]
            end_elem = self.pipeline[stage+1]
            if start_elem.output_size != end_elem.input_size:
                return False
        return True
    
    def add_stage(self, stage: Any, idx: int=-1) -> None:
        self.pipeline.append(stage)
        # Check if the pipeline is still valid
        if not self._verify():
            print('Invalid pipeline configuration!\nUndoing add stage!')
            self.delete_stage(idx)

    def delete_stage(self, idx: int=-1) -> None:
        if len(self.pipeline) > 0:
            self.pipeline.pop(idx)
        else:
            print('Cannot delete stage from pipeline! Not enough stages')

    def forward(self, data: Any) -> None:
        for stage in self.pipeline:
            data = stage.forward(data)
        return data
    
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