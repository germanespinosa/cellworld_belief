import typing
import cellworld_game as cg
import torch


class BeliefStateComponent(object):

    def __init__(self):
        self.belief_state = None

    def set_belief_state(self, belief_state: "BeliefState"):
        if self.belief_state is None:
            self.belief_state = belief_state
        else:
            raise ValueError("Belief state has already been set.")
        self.on_belief_state_set(belief_state=belief_state)

    def on_belief_state_set(self,
                            belief_state: "BeliefState") -> None:
        pass

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        raise NotImplementedError

    def on_reset(self) -> None:
        pass

    def on_self_location_update(self,
                                self_location: cg.Point.type,
                                self_indices: typing.Tuple[int, int],
                                time_step: int) -> None:
        pass

    def on_other_location_update(self,
                                 other_location: cg.Point.type,
                                 other_indices: typing.Tuple[int, int],
                                 time_step: int) -> None:
        pass

    def on_visibility_update(self,
                             visibility_polygon: cg.Polygon,
                             time_step: int) -> None:
        pass

    def on_tick(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        self.predict(probability_distribution=probability_distribution,
                     time_step=time_step)
