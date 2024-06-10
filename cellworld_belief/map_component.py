from .belief_state_component import BeliefStateComponent
import torch


class MapComponent(BeliefStateComponent):
    def __init__(self):
        BeliefStateComponent.__init__(self)
        self.valid_map = None
        self.invalid_map = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        inside_arena = belief_state.model.arena.contains(belief_state.points)
        inside_arena_matrix = torch.reshape(inside_arena, belief_state.shape)
        for occlusion in belief_state.model.occlusions:
            inside_occlusion = occlusion.contains(belief_state.points)
            inside_occlusion_matrix = torch.reshape(inside_occlusion, belief_state.shape)
            inside_arena_matrix[inside_occlusion_matrix] = False

        self.invalid_map = torch.logical_not(inside_arena_matrix)

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution[self.invalid_map] = 0
