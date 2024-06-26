import typing
import cellworld_game as cg
from .utils import gaussian_tensor
from .belief_state_component import BeliefStateComponent
import torch


class ProximityComponent(BeliefStateComponent):
    def __init__(self,
                 distance_threshold: float = .1,
                 other_scale: float = 1.0):
        BeliefStateComponent.__init__(self)
        self.other_scale = other_scale
        self.distance_threshold = distance_threshold
        self.other_size = None
        self.other_sigma = None
        self.other_location = None
        self.other_indices = None
        self.other_location_update_time_step = None
        self.self_location = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.other_size = belief_state.other.size * self.other_scale
        self.other_sigma = int(self.other_size * belief_state.definition) + 1

    def on_reset(self) -> None:
        self.other_location = None
        self.other_indices = None
        self.other_location_update_time_step = None

    def on_self_location_update(self,
                                self_location: cg.Point.type,
                                self_indices: typing.Tuple[int, int],
                                time_step: int) -> None:
        self.self_location = self_location

    def on_other_location_update(self,
                                 other_location: cg.Point.type,
                                 other_indices: tuple,
                                 time_step: int):
        if cg.Point.distance(self.self_location, other_location) < self.distance_threshold:
            self.other_location = other_location
            self.other_indices = other_indices
            self.other_location_update_time_step = time_step

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        if self.other_location_update_time_step == time_step:
            other_distribution = gaussian_tensor(dimensions=self.belief_state.shape,
                                                 sigma=self.other_sigma,
                                                 center=self.other_indices,
                                                 device=self.belief_state.device)
            probability_distribution.copy_(other_distribution)
            self.belief_state.probability = 1.0
