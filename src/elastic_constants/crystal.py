"""Crystallography module."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


type Coordinate = Sequence[float]


__all__ = ["Crystal"]


class Crystal:
    """Crystal class.

    Periodic arrangement of atoms.

    Parameters
    ----------
    lattice : list[list[float]]
        Matrix containing the three crystal lattice vectors as its rows.
    numbers : list[int]
        Atomic numbers to distinguish the atomic species.
    positions : list[list[float]]
        Atomic positions given as fractional coordinates.

    Examples
    --------
    Create a body-centered caesium chloride crystal using:

    >>> lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> numbers = [55, 17]
    >>> positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    >>> crystal = Crystal(lattice, numbers, positions)
    """

    def __init__(
        self, lattice: list[list[float]], numbers: list[int], positions: list[list[float]]
    ) -> None:
        self.lattice: NDArray[np.float64] = np.asarray(lattice, dtype=np.float64)
        self.numbers: NDArray[np.uint8] = np.asarray(numbers, dtype=np.uint8)
        self.positions: NDArray[np.float64] = np.asarray(positions, dtype=np.float64)
