import numpy as np

from mantrap_guard.flow import FlowEstimator


def scene(shift):
    rng = np.random.default_rng(0)
    img = np.full((240, 320, 3), 90, np.uint8)
    patch = rng.integers(0, 255, (120, 60, 3), dtype=np.uint8)
    patch = np.kron(patch[::4, ::4], np.ones((4, 4, 1), np.uint8))  # blocky texture
    x = 100 + shift
    img[60:180, x:x + 60] = patch
    return img, (x, 60, x + 60, 180)


def test_flow_direction_and_magnitude():
    for method in ("dis", "farneback"):
        f = FlowEstimator(method=method, width=320)
        v = np.zeros(2)
        for i in range(4):
            img, box = scene(4 * i)
            f.update(img)
            v = f.vector(box)
        assert v[0] > 2.0 and abs(v[1]) < 1.0, (method, v)


def test_static_scene_gives_zero():
    f = FlowEstimator(width=320)
    img, box = scene(0)
    f.update(img)
    f.update(img)
    assert np.allclose(f.vector(box), 0)
