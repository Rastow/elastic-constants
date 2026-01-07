"""Crystallography module."""

import contextlib

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import spglib

from spglib import SpglibError


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from spglib.spg import SpglibDataset


type Coordinate = Sequence[float]


__all__ = ["Crystal"]


with contextlib.suppress(AttributeError):
    spglib.error.OLD_ERROR_HANDLING = False


class Crystal:
    """Crystal class.

    Periodic arrangement of atoms.

    Attributes
    ----------
    symmetry_dataset : spglib.spg.SpglibDataset
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
        """Initialize an instance of the crystal class.

        Parameters
        ----------
        lattice: numpy.typing.ArrayLike
            Matrix containing the three crystal lattice vectors as its rows.
        positions : numpy.typing.ArrayLike
            List of atomic positions given as fractional coordinates.
        numbers : numpy.typing.ArrayLike
            List of (arbitrary) numbers to distinguish the different atomic
            species.

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
        try:
            symmetry_dataset = spglib.spg.get_symmetry_dataset(
                cell, symprec=symmetry_precision, angle_tolerance=angle_tolerance
            )
        except SpglibError as exc:
            msg = "cannot instantiate due to failed symmetry search."
            raise ValueError(msg) from exc

        self.symmetry_dataset: SpglibDataset = symmetry_dataset  # type: ignore [reportAttributeAccessIssue]
        self.lattice: NDArray[np.double] = np.asarray(lattice, dtype=np.double)
        self.positions: NDArray[np.double] = np.asarray(positions, dtype=np.double)
        self.numbers: NDArray[np.ubyte] = np.asarray(numbers, dtype=np.ubyte)
