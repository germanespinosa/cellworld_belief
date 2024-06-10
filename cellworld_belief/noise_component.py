from .belief_state_component import BeliefStateComponent
import torch


class NoiseComponent(BeliefStateComponent):
    def __init__(self, noise_level: float = .1):
        BeliefStateComponent.__init__(self)
        self.noise_level = noise_level

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution *= torch.rand(self.belief_state.shape) * self.noise_level
