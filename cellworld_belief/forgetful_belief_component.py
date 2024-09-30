import torch
from .belief_state_component import BeliefStateComponent


class ForgetfulBeliefComponent(BeliefStateComponent):

    def __init__(self):
        BeliefStateComponent.__init__(self)

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        probability_distribution.fill_(1)
