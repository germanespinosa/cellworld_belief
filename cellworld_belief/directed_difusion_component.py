from .belief_state_component import BeliefStateComponent
from .utils import shift_tensor
import math
import torch
import typing
import cellworld_game as cg


class DirectedDiffusionComponent(BeliefStateComponent):
    def __init__(self, diffusion_scale: float = 1.0):
        BeliefStateComponent.__init__(self)
        self.diffusion_scale = diffusion_scale
        self.start_time_step = 0
        self.end_time_step = 0
        self.step_distance = None
        self.source = None
        self.step = None
        self.speed_rate = None

    def on_reset(self):
        self.start_time_step = 0
        self.end_time_step = 0

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.speed_rate = belief_state.other.max_forward_speed * self.diffusion_scale
        self.step_distance = self.speed_rate / belief_state.granularity

    def on_other_location_update(self,
                                 other_location: cg.Point.type,
                                 other_indices: typing.Tuple[int, int],
                                 time_step: int) -> None:
        self.source = self.belief_state.other_indices
        target = self.belief_state.self_indices
        distance = math.dist(self.source, target)
        step_count = distance / self.step_distance
        if step_count > 0:
            self.start_time_step = time_step + 1
            self.step = tuple((ti - si) / step_count for si, ti in zip(self.source, target))
            self.end_time_step = time_step + int(step_count)
        else:
            self.start_time_step = 0
            self.end_time_step = 0

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        if self.start_time_step < time_step <= self.end_time_step:
            shifted_probability_distribution = shift_tensor(tensor=probability_distribution,
                                                            displacement=self.step)
            probability_distribution.copy_(shifted_probability_distribution)

