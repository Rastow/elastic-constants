import pytest

from elastic_constants.crystal import Crystal


@pytest.mark.parametrize(
    ("lattice", "positions", "numbers"),
    [
        (
            # overlapping atoms
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [1, 1],
        ),
        (
            # zero-length lattice vector
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
            [1],
        ),
    ],
)
def test_crystal_invalid_input(
    lattice: list[list[float]], positions: list[list[float]], numbers: list[int]
) -> None:
    err_msg = "cannot instantiate due to failed symmetry search."
    with pytest.raises(ValueError, match=err_msg):
        _ = Crystal(lattice, positions, numbers)
