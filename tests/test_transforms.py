import numpy as np

from joistrogen.transforms.affine import AffineTransformer
from joistrogen.transforms.identity import IdentityTransformer


def test_identity_transform_leaves_samples_unchanged():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    transformed = IdentityTransformer().transform(samples)
    assert np.allclose(transformed, samples)


def test_affine_transform_applies_homogeneous_matrix():
    samples = np.array([[1.0, 2.0], [3.0, 4.0]])
    transform = AffineTransformer.from_spec("(1, 0, 2, 0, 1, 3)", dimension=2)
    transformed = transform.transform(samples)
    assert transformed.shape == (2, 2)
    expected = np.array([
        [1.0 * 1 + 2.0 * 0 + 2.0, 1.0 * 0 + 2.0 * 1 + 3.0],
        [3.0 * 1 + 4.0 * 0 + 2.0, 3.0 * 0 + 4.0 * 1 + 3.0],
    ])
    assert np.allclose(transformed, expected)
