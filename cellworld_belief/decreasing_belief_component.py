import typing
import cellworld_game as cg
import torch
from .belief_state_component import BeliefStateComponent


class DecreasingBeliefComponent(BeliefStateComponent):

    def __init__(self, rate: float = .01):
        self.rate = rate
        self.active = False
        BeliefStateComponent.__init__(self)

    def on_reset(self):
        self.active = True

    def on_other_location_update(self,
                                 other_location: cg.Point.type,
                                 other_indices: typing.Tuple[int, int],
                                 time_step: int) -> None:
        self.active = False

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        if self.active:
            self.belief_state.probability = self.belief_state.probability * (1 - self.rate)
