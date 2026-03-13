from typing import Callable

from romtools.model import Model


class Poisson2D(Model):

    forcing: Callable[[dict], int]

    def evaluate(self, inputs: dict) -> int:
        return self.forcing(inputs)

    def solve(self):
        pass
    