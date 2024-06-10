from .belief_state_component import BeliefStateComponent
import math
import torch
import torch.nn.functional as F


class DiffusionComponent(BeliefStateComponent):
    def __init__(self, diffusion_scale: float = 1.0):
        BeliefStateComponent.__init__(self)
        self.diffusion_scale = diffusion_scale
        self.radius = None
        self.diffusion_kernel = None
        self.padding = None
        self.kernel_dimensions = None
        self.diffusion_rate = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.diffusion_rate = belief_state.other.max_forward_speed * self.diffusion_scale
        self.radius = math.ceil(self.diffusion_rate / belief_state.granularity)
        size = self.radius * 2 + 1
        self.kernel_dimensions = (size, size)
        kernel = torch.zeros(self.kernel_dimensions, dtype=torch.float32, device=belief_state.device)
        x_grid, y_grid = torch.meshgrid(torch.arange(size, device=belief_state.device),
                                        torch.arange(size, device=belief_state.device), indexing='xy')
        distance = (x_grid - self.radius) ** 2 + (y_grid - self.radius) ** 2
        within_radius = distance <= self.radius
        kernel[within_radius] = 1
        self.diffusion_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        convoluted = F.conv2d(probability_distribution.unsqueeze(0).unsqueeze(0),
                              self.diffusion_kernel,
                              padding=self.padding).squeeze(0).squeeze(0)

        probability_distribution.copy_(convoluted)
