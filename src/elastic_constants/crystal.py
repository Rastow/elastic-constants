"""Crystallography module."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import spglib


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spglib.spglib import SpglibDataset


type Coordinate = Sequence[float]


__all__ = ["Crystal"]


class Crystal:
    """Crystal class.

    Periodic arrangement of atoms.

    Attributes
    ----------
    lattice: numpy.ndarray[tuple[typing.Any, ...], numpy.double]
        Matrix containing the three crystal lattice vectors as its rows.
    positions : numpy.ndarray[tuple[typing.Any, ...], numpy.double]
        Atomic positions given as fractional coordinates.
    numbers : numpy.ndarray[tuple[typing.Any, ...], numpy.ubyte]
        Arbitrary numbers to distinguish the different atomic species.
    symmetry_dataset : spglib.spglib.SpglibDataset
        Dataclass containing symmetry related information.

    Examples
    --------
    Create a body-centered caesium chloride crystal using:

    >>> lattice = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    >>> positions = [[0, 0, 0], [0.5, 0.5, 0.5]]
    >>> numbers = [55, 17]
    >>> crystal = Crystal(lattice, positions, numbers)
    >>> crystal.symmetry_dataset.number
    221
    """

    def __init__(
        self,
        lattice: list[list[float]],
        positions: list[list[float]],
        numbers: list[int],
        *,
        symmetry_precision: float = 1e-5,
        angle_tolerance: float = -1.0,
    ) -> None:
        """Initialize the crystal class.

        Parameters
        ----------
        lattice : list[list[float]]
        positions : list[list[float]]
        numbers : list[int]

        Other Parameters
        ----------------
        symmetry_precision : float
            Distance tolerance in cartesian coordinates during symmetry search.
            See `symprec <https://spglib.readthedocs.io/en/stable/variable.html#symprec>`_.
        angle_tolerance : float
            Tolerance of angle between basis vectors in degrees during symmetry
            search. See `angle_tolerance <https://spglib.readthedocs.io/en/stable/variable.html#angle-tolerance>`_.

        Raises
        ------
        ValueError
            If the symmetry search failed.
        """
        cell = (lattice, positions, numbers)
        symmetry_dataset = spglib.spglib.get_symmetry_dataset(
            cell, symprec=symmetry_precision, angle_tolerance=angle_tolerance
        )
        if symmetry_dataset is None:
            msg = f"Spglib failed due to {spglib.spglib.get_error_message()}."
            raise ValueError(msg)
        self.symmetry_dataset: SpglibDataset = symmetry_dataset
        self.lattice: NDArray[np.double] = np.asarray(lattice, dtype=np.double)
        self.positions: NDArray[np.double] = np.asarray(positions, dtype=np.double)
        self.numbers: NDArray[np.ubyte] = np.asarray(numbers, dtype=np.ubyte)
