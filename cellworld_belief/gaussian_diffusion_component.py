from .belief_state_component import BeliefStateComponent
import math
from .utils import gaussian_tensor
import torch
import torch.nn.functional as F


class GaussianDiffusionComponent(BeliefStateComponent):
    def __init__(self, diffusion_scale: float = 1.0):
        BeliefStateComponent.__init__(self)
        self.diffusion_scale = diffusion_scale
        self.stencil_size = None
        self.diffusion_kernel = None
        self.padding = None
        self.kernel_dimensions = None
        self.diffusion_rate = None

    def on_belief_state_set(self,
                            belief_state: "BeliefState"):
        self.diffusion_rate = belief_state.other.max_forward_speed * self.diffusion_scale
        self.stencil_size = math.ceil(self.diffusion_rate / belief_state.granularity)
        self.kernel_dimensions = (self.stencil_size * 2 + 1, self.stencil_size * 2 + 1)
        kernel = gaussian_tensor(self.kernel_dimensions, self.stencil_size, device=belief_state.device)
        self.padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
        self.diffusion_kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.diffusion_kernel.to(belief_state.device)

    def predict(self,
                probability_distribution: torch.tensor,
                time_step: int) -> None:
        convoluted = F.conv2d(probability_distribution.unsqueeze(0).unsqueeze(0),
                              self.diffusion_kernel,
                              padding=self.padding).squeeze(0).squeeze(0)

        probability_distribution.copy_(convoluted)
