from .belief_state_component import BeliefStateComponent
import torch


class OcclusionsComponent(BeliefStateComponent):
    def __init__(self):
        BeliefStateComponent.__init__(self)
        self.invalid_map = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.invalid_map = torch.zeros(belief_state.shape,
                                       dtype=torch.bool,
                                       device=belief_state.device)

        for occlusion in belief_state.model.occlusions:
            inside_occlusion = occlusion.contains(belief_state.points)
            inside_occlusion_matrix = torch.reshape(inside_occlusion, belief_state.shape)
            self.invalid_map[inside_occlusion_matrix] = True

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution[self.invalid_map] = 0
