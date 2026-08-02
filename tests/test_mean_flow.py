import unittest

import torch
from torch import nn

from mean_flow import (
    improved_velocity_from_mean_velocity,
    imf_loss,
    mean_flow_loss,
    sample_ordered_times,
)


class TinyMeanVelocity(nn.Module):
    """Small differentiable network used to smoke-test the iMF autograd path."""

    null_class_idx = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, image, r, t, w, class_labels):
        return self.scale * image + (t - r)[:, None, None, None] * self.scale


class ImprovedMeanFlowTests(unittest.TestCase):
    def test_noise_to_data_identity_uses_minus_derivative(self):
        u = torch.tensor([[[[4.0]]]])
        du_dr = torch.tensor([[[[3.0]]]])
        interval = torch.tensor([[[[0.25]]]])

        actual = improved_velocity_from_mean_velocity(u, du_dr, interval)

        self.assertTrue(torch.equal(actual, torch.tensor([[[[3.25]]]])))

    def test_imf_loss_and_gradients_are_finite(self):
        torch.manual_seed(0)
        model = TinyMeanVelocity()
        images = torch.randn(4, 1, 3, 3)
        labels = torch.tensor([0, 1, 0, 1])

        loss, raw_mse = imf_loss(
            model,
            images,
            labels,
            p_uncond=0.0,
            w_min=1.0,
            w_max=1.0,
            return_raw_mse=True,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw_mse))
        self.assertTrue(torch.isfinite(model.scale.grad))

    def test_mean_flow_adaptive_loss_and_logit_normal_times_are_finite(self):
        torch.manual_seed(0)
        model = TinyMeanVelocity()
        images = torch.randn(4, 1, 3, 3)
        labels = torch.tensor([0, 1, 0, 1])

        r, t = sample_ordered_times(128, images.device, time_sampler="logit_normal")
        loss, raw_mse = mean_flow_loss(
            model,
            images,
            labels,
            p_uncond=0.0,
            w_min=1.0,
            w_max=1.0,
            time_sampler="logit_normal",
            return_raw_mse=True,
        )
        loss.backward()

        self.assertTrue(torch.all((0.0 <= r) & (r <= t) & (t <= 1.0)))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(raw_mse))
        self.assertTrue(torch.isfinite(model.scale.grad))


if __name__ == "__main__":
    unittest.main()
