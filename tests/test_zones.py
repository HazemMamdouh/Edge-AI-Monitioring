import numpy as np

from mantrap_guard.annotate import parse_points
from mantrap_guard.zones import Zones, crosses_inward, nearest_on_segment


def make():
    return Zones(np.array([[0, 0], [100, 0], [100, 100], [0, 100]], np.float32),
                 np.array([[10, 10], [90, 10]], np.float32),
                 np.array([[0, 95], [100, 95]], np.float32), (100, 100))


def test_contains():
    z = make()
    assert z.contains((50, 50))
    assert not z.contains((150, 50))


def test_scale_and_roundtrip(tmp_path):
    z = make()
    z.save(tmp_path / "z.json")
    z2 = Zones.load(tmp_path / "z.json", (200, 400))
    assert z2.contains((150, 300))
    assert np.allclose(z2.entry_line, [[20, 40], [180, 40]])


def test_inward_normal_points_into_mantrap():
    z = make()
    assert np.allclose(z.inward_normal(z.exit_line), [0, -1], atol=1e-5)


def test_crossing_direction():
    z = make()
    n = z.inward_normal(z.exit_line)
    assert crosses_inward((50, 99), (50, 90), z.exit_line, n)       # outside -> inside
    assert not crosses_inward((50, 90), (50, 99), z.exit_line, n)   # leaving
    assert not crosses_inward((300, 99), (300, 90), z.exit_line, n)  # far beside the line


def test_nearest_on_segment_clamps():
    assert np.allclose(nearest_on_segment((200, 50), np.array([[0, 0], [100, 0]], np.float32)), [100, 0])


def test_parse_points():
    assert parse_points("1,2 3,4") == [[1.0, 2.0], [3.0, 4.0]]
