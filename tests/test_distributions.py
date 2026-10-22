import numpy as np
import pytest

from joistrogen.distributions.copula import GaussianCopulaSampler
from joistrogen.distributions.normal import JointNormalSampler


def test_joint_normal_sampler_has_correct_shape_and_bounds():
    sampler = JointNormalSampler([0.0, 0.0], [1.0, 0.0, 1.0])
    samples = sampler.sample(512)
    assert samples.shape == (512, 2)
    assert np.allclose(np.var(samples, axis=0, ddof=1), np.array([1.0, 1.0]), atol=0.4)


def test_gaussian_copula_sampler_stays_in_interval():
    sampler = GaussianCopulaSampler([0.0, 0.0], [1.0, 1.0], [0.5])
    samples = sampler.sample(256)
    assert samples.shape == (256, 2)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_joint_normal_sampler_rejects_invalid_covariance():
    with pytest.raises(ValueError):
        JointNormalSampler([0.0, 0.0], [1.0, 0.0, -2.0])
