import pytest

from elastic_constants.crystal import Crystal


@pytest.mark.parametrize(
    ("lattice", "positions", "numbers", "err_msg"),
    [
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [1, 1],
            "too close distance between atoms",
        ),
        (
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [1],
            "spacegroup search failed",
        ),
    ],
)
def test_crystal_invalid_input(
    lattice: list[list[float]], positions: list[list[float]], numbers: list[int], err_msg: str
) -> None:
    with pytest.raises(ValueError, match=err_msg):
        _ = Crystal(lattice, positions, numbers)
