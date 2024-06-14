import typing

import cellworld_game as cg
from .belief_state_component import BeliefStateComponent
import torch


class VisibilityComponent(BeliefStateComponent):
    def __init__(self):
        BeliefStateComponent.__init__(self)
        self.visibility_polygon = None
        self.visibility_update_time_step = None
        self.other_location_update_time_step = None
        self.visibility_map = None

    def on_reset(self) -> None:
        self.visibility_update_time_step = None

    def on_visibility_update(self,
                             visibility_polygon: cg.Polygon,
                             time_step: int) -> None:
        self.visibility_polygon = visibility_polygon
        self.visibility_update_time_step = time_step

    def on_other_location_update(self,
                                 other_location: cg.Point.type,
                                 other_indices: typing.Tuple[int, int],
                                 time_step: int) -> None:
        self.other_location_update_time_step = time_step

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        if self.visibility_update_time_step == time_step and self.other_location_update_time_step != time_step:
            in_view = self.visibility_polygon.contains(self.belief_state.points)
            in_view_matrix = torch.reshape(in_view, self.belief_state.shape)
            probability_distribution[in_view_matrix] = 0
