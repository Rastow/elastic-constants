"""Tensor module."""

import itertools
import math
import string

from abc import ABC
from typing import Any
from typing import ClassVar
from typing import Self

import numpy as np

from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from numpy.typing import NDArray


__all__ = ["Tensor"]


class Tensor(ABC):
    """Abstract tensor class.

    Base class for tensors of arbitrary rank in three-dimensional Euclidean
    space. This class provides methods for initializing tensors, converting
    to and from Voigt notation, and applying orthogonal transformations.
    """

    dimension: ClassVar[int] = 3

    #: Rank of the tensor. Must be defined in subclasses.
    rank: ClassVar[int]

    #: Whether the tensor is Voigt symmetric. Intended to be used by subclasses.
    #: Every index pair in the tensor that maps to a single Voigt index must be
    #: symmetric for the tensor to be Voigt symmetric. For odd rank tensors, the
    #: first index is not part of any index pair.
    voigt_symmetric: ClassVar[bool] = False

    #: Mapping from Voigt index to tensor index pair and its permutations.
    #: Uses zero-based indexing.
    voigt_to_tensor_index: ClassVar[dict[int, tuple[tuple[int, int], ...]]] = {
        0: ((0, 0),),
        1: ((1, 1),),
        2: ((2, 2),),
        3: ((1, 2), (2, 1)),
        4: ((0, 2), (2, 0)),
        5: ((0, 1), (1, 0)),
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:  # noqa: ARG004
        """Create a new tensor instance.

        Raises
        ------
        TypeError
            If attempting to instantiate the abstract tensor class.
        """
        if cls is Tensor:
            msg = "only tensor subclasses may be instantiated."
            raise TypeError(msg)

        return super().__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Initialize a tensor subclass.

        Raises
        ------
        AttributeError
            If the tensor subclass does not define its rank.
        ValueError
            If the Voigt notation scaling factor does not have the correct
            shape.
        ValueError
            If the Voigt notation scaling factor contains zero values.
        """
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "rank"):
            msg = f"{cls.__name__.lower()} tensor must define its rank as a class variable."
            raise AttributeError(msg)

        cls.voigt_shape: tuple[int, ...] = (cls.dimension,) * (cls.rank % 2) + (
            math.comb(cls.dimension + 2 - 1, 2),
        ) * (cls.rank // 2)

        cls.voigt_scale: NDArray[np.number] = (
            np.asarray(cls.voigt_scale)
            if hasattr(cls, "voigt_scale")
            else np.ones(cls.voigt_shape, dtype=np.double)
        )

        if cls.voigt_scale.shape != cls.voigt_shape:
            msg = (
                f"voigt scale for {cls.__name__.lower()} tensor must be of shape "
                f"{cls.voigt_shape}."
            )
            raise ValueError(msg)

        if np.any(np.isclose(cls.voigt_scale, 0)):
            msg = f"voigt scale for {cls.__name__.lower()} tensor must not contain zero."
            raise ValueError(msg)

    def __init__(self, array_like: ArrayLike) -> None:
        """Initialize an instance of the tensor class.

        Parameters
        ----------
        array_like : numpy.typing.ArrayLike
            Input array.

        Raises
        ------
        TypeError
            If the input array does not have the correct data type.
        ValueError
            If the input array does not have the correct rank.
        ValueError
            If the input array does not have the correct shape.
        ValueError
            If the input array is not Voigt symmetric when required.
        """
        array = np.asarray(array_like)

        if not np.issubdtype(array.dtype, np.number):
            msg = "array must only contain numeric data types."
            raise TypeError(msg)

        if array.ndim != self.rank:
            msg = f"input for {self.__class__.__name__.lower()} tensor must be rank {self.rank}."
            raise ValueError(msg)

        if any(dimension != self.dimension for dimension in array.shape):
            msg = (
                f"input for {self.__class__.__name__.lower()} tensor must be "
                f"{self.dimension}-dimensional."
            )
            raise ValueError(msg)

        self._array: NDArray[np.number] = array

        if self.voigt_symmetric and not self.is_voigt_symmetric():
            msg = f"input for {self.__class__.__name__.lower()} tensor must be voigt symmetric."
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return the string representation of the tensor.

        Returns
        -------
        str
            The string representation of the tensor.
        """
        return f"{self.__class__.__name__} tensor\n{self._array}\n"

    def __array__(
        self,
        dtype: DTypeLike | None = None,
        copy: bool | None = None,  # noqa: FBT001
    ) -> NDArray[np.number]:
        """Return the array representation of the tensor.

        Parameters
        ----------
        dtype : numpy.typing.DTypeLike | None
            Desired data type of the array. By default, the data-type is
            inferred from the input data.
        copy : bool | None
            Whether to return a copy or a view of the array. By default, the
            object is only copied if needed. Raises :exc:`ValueError`
            if a copy is needed.

        Returns
        -------
        numpy.typing.NDArray[numpy.number]
            The array representation of the tensor.

        See Also
        --------
        numpy.asarray : Convert the input to an array.
        """
        return np.asarray(self._array, dtype=dtype, copy=copy)

    @classmethod
    def from_voigt(cls, array_like: ArrayLike) -> Self:
        """Create a tensor from an array in voigt notation.

        Parameters
        ----------
        array_like : numpy.typing.ArrayLike
            Array in Voigt notation.

        Raises
        ------
        TypeError
            If the Voigt array does not have the correct data type.
        ValueError
            If the Voigt array does not have the appropriate shape.

        Returns
        -------
        typing.Self
            The tensor.
        """
        voigt_array = np.asarray(array_like)

        if not np.issubdtype(voigt_array.dtype, np.number):
            msg = "voigt array must only contain numeric data types."
            raise TypeError(msg)

        if voigt_array.shape != cls.voigt_shape:
            msg = f"voigt array must be of shape {cls.voigt_shape}."
            raise ValueError(msg)

        # scale the voigt array based on voigt scale
        scaled_voigt_array = voigt_array / cls.voigt_scale

        # initialize full tensor array
        array = np.zeros((cls.dimension,) * cls.rank, dtype=scaled_voigt_array.dtype)

        for voigt_index in np.ndindex(voigt_array.shape):
            # map voigt indices to tensor indices
            tensor_index_sets: list[tuple[tuple[int, ...], ...]] = []
            if cls.rank % 2 == 0:
                # even rank case, all indices map to pairs
                tensor_index_sets.extend(cls.voigt_to_tensor_index[i] for i in voigt_index)
            elif cls.rank % 2 == 1:
                # odd rank case, first index maps directly
                tensor_index_sets.append(((voigt_index[0],),))
                tensor_index_sets.extend(cls.voigt_to_tensor_index[i] for i in voigt_index[1:])

            # generate cartesian product of tensor index sets
            for grouped_tensor_index in itertools.product(*tensor_index_sets):
                tensor_index = tuple(itertools.chain(*grouped_tensor_index))
                array[tensor_index] = scaled_voigt_array[voigt_index]

        return cls(array)

    def is_symmetric(
        self, indices: tuple[int, ...] | None = None, tolerance: float = 1e-6
    ) -> bool:
        """Check for symmetry under permutation of the given indices.

        Parameters
        ----------
        indices : tuple[int, ...] | None
            List of indices to check for symmetry. Checks all indices if None.

        Other Parameters
        ----------------
        tolerance : float
            Absolute tolerance for floating point comparisons.

        Returns
        -------
        bool
            Whether the tensor is symmetric in the given indices.
        """
        indices = tuple(range(self.rank)) if indices is None else indices
        return all(
            np.allclose(self, np.swapaxes(self, i, j), atol=tolerance)
            for i, j in itertools.combinations(indices, 2)
        )

    def is_voigt_symmetric(self, tolerance: float = 1e-6) -> bool:
        """Check if the tensor is Voigt symmetric.

        Parameters
        ----------
        tolerance : float
            Absolute tolerance for floating point comparisons.

        Returns
        -------
        bool
            Whether the tensor is Voigt symmetric.
        """
        # scalars and vectors are always voigt symmetric
        if self.rank < 2:  # noqa: PLR2004
            return True

        index_pairs = [(i, i + 1) for i in range(self.rank % 2, self.rank, 2)]
        return all(self.is_symmetric((i, j), tolerance) for i, j in index_pairs)

    def to_voigt(self) -> NDArray[np.number]:
        """Convert the tensor to Voigt notation.

        Raises
        ------
        ValueError
            If the tensor is not Voigt symmetric.

        Returns
        -------
        numpy.typing.NDArray[numpy.number]
            The tensor in Voigt notation.
        """
        if not self.is_voigt_symmetric():
            msg = "tensor must be voigt symmetric."
            raise ValueError(msg)

        # initialize voigt array
        voigt_array = np.zeros(self.voigt_shape, dtype=self._array.dtype)

        for voigt_index in np.ndindex(voigt_array.shape):
            # map voigt indices to tensor indices
            # do not map the first index for odd rank tensors
            # only take the first mapping for remaining voigt indices
            grouped_tensor_index = (((voigt_index[0],),) if self.rank % 2 == 1 else ()) + tuple(
                self.voigt_to_tensor_index[i][0] for i in voigt_index[self.rank % 2 :]
            )
            tensor_index = tuple(itertools.chain(*grouped_tensor_index))
            voigt_array[voigt_index] = self._array[tensor_index]

        # scale the voigt array based on voigt scale
        return voigt_array * self.voigt_scale

    def transform(self, matrix: ArrayLike) -> Self:
        r"""Apply a transformation matrix to the tensor.

        Only transformations that preserve the Euclidean metric are allowed,
        that is, rotations and reflections. The transformation matrix must
        therefore be orthogonal.

        Parameters
        ----------
        matrix : numpy.typing.ArrayLike
            The real-valued transformation matrix.

        Returns
        -------
        typing.Self
            The transformed tensor.

        Raises
        ------
        ValueError
            If the transformation matrix is not of shape (3,3).
        ValueError
            If the transformation matrix is not orthogonal.

        Notes
        -----
        Using Einstein summation convention, the components of a rank :math:`N`
        tensor :math:`T` transform under an orthogonal change of basis :math:`Q`
        to the components :math:`T'` in the new basis according to the following
        rule.

        .. math::
            T'_{i_1 i_2 \dots i_N} = Q_{i_1 j_1} \, Q_{i_2 j_2} \dots
            Q_{i_N j_N} \, T_{j_1 j_2 \dots j_N}

        """
        array = np.asarray(matrix, dtype=np.double)

        if array.shape != (self.dimension, self.dimension):
            msg = f"transformation matrix must be of shape {(self.dimension, self.dimension)}."
            raise ValueError(msg)

        if not np.allclose(array @ array.T, np.eye(self.dimension)):
            msg = "transformation matrix must be orthogonal."
            raise ValueError(msg)

        # scalars are invariant under transformations
        if self.rank == 0:
            return type(self)(self._array.copy())

        # build the einstein summation string containing the appropriate indices
        t_out_indices = string.ascii_lowercase[: self.rank]
        t_in_indices = string.ascii_lowercase[self.rank : 2 * self.rank]
        q_indices = ",".join(
            f"{out_index}{in_index}"
            for out_index, in_index in zip(t_out_indices, t_in_indices, strict=True)
        )
        einsum_string: str = f"{q_indices},{t_in_indices}->{t_out_indices}"

        # group the arguments for einstein summation
        einsum_args: list[NDArray[np.double] | Self] = [array] * self.rank + [self]

        # return the transformed tensor
        return type(self)(np.einsum(einsum_string, *einsum_args))
