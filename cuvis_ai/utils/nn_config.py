from abc import ABC, abstractmethod
from dataclasses import dataclass

class Optimizer(ABC):

    @property
    @abstractmethod

    def name(self):
        pass

    @property
    @abstractmethod
    def args(self):
        pass


@dataclass
class Adam(Optimizer):
    lr: float = 0.001
    alpha: float = 0.0001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8
    
    @Optimizer.name.getter
    def name(self):
        return 'Adam Optimizer'
    
    @Optimizer.args.getter
    def args(self):
        return {
            'learning_rate_init' : self.lr,
            'alpha' : self.alpha,
            'beta_1' : self.beta_1,
            'beta_2' : self.beta_2,
            'epsilon' : self.epsilon
        }

@dataclass
class SGD(Optimizer):
    lr: float = 0.001
    alpha: float = 0.0001
    power_t: float = 0.5
    momentum: float = 0.9
    nesterov_momentum: bool = True

    @Optimizer.name.getter
    def name(self):
        return 'SGD Optimizer'
    
    @Optimizer.args.getter
    def args(self):
        return {
            'learning_rate_init' : self.lr,
            'alpha' : self.alpha,
            'power_t' : self.power_t,
            'momentum' : self.momentum,
            'nesterov_momentum' : self.nesterov_momentum
         }