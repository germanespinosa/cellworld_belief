from .belief_state_component import BeliefStateComponent
import torch


class NoBeliefComponent(BeliefStateComponent):
    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution.fill_(0)
        self.belief_state.probability = 0
