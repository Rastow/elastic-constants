"""Tensor module."""

import math
import string

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import Self

import numpy as np

from numpy.typing import ArrayLike
from numpy.typing import NDArray


__all__ = ["Tensor"]


class Tensor(np.ndarray):
    """Tensor class.

    Base class for tensors of arbitrary rank in three-dimensional Euclidean
    space.

    Parameters
    ----------
    input_array : numpy.typing.ArrayLike
        Input array.
    """

    dimension: Final[int] = 3
    rank: ClassVar[int]
    voigt_shape: ClassVar[tuple[int, ...]]
    voigt_scale: ClassVar[NDArray[np.double]]

    def __init_subclass__(cls, rank: int, voigt_scale: NDArray[np.double] | None = None) -> None:
        """Subclass initialization."""
        cls.rank = rank
        cls.voigt_shape = tuple(
            [cls.dimension] * (cls.rank % 2)
            + [math.comb(cls.dimension + 2 - 1, 2)] * (cls.rank // 2)
        )
        cls.voigt_scale = (
            voigt_scale if voigt_scale is not None else np.ones(cls.voigt_shape, dtype=np.double)
        )  # needs to be overwritten when the subclass defines it

    def __new__(cls, input_array: ArrayLike) -> Self:
        """Instance initialization through explicit constructor call.

        Raises
        ------
        TypeError
            If the input array does not have the correct data type.
        ValueError
            If the input array does not have the correct rank or dimension.
        """
        array = np.asarray(input_array)

        if not np.issubdtype(array.dtype, np.number):
            msg = "array must only contain number data types."
            raise TypeError(msg)

        if array.ndim != cls.rank:
            msg = f"input for {cls.__name__.lower()} must be rank {cls.rank}."
            raise ValueError(msg)

        if any(dimension != cls.dimension for dimension in array.shape):
            msg = f"input for {cls.__name__.lower()} must be {cls.dimension}-dimensional."
            raise ValueError(msg)

        return array.view(cls)

    def __array_finalize__(self, obj: NDArray[np.double] | None) -> None:
        """Handle instance creation.

        Views and slices of the tensor will be upcast to the base
        :class:`numpy.ndarray` class.

        Parameters
        ----------
        obj : numpy.typing.NDArray[numpy.double] or None
            The original array or :class:`None` if the instance is created
            through the constructor.
        """
        if obj is None:
            return
        self.__class__ = np.ndarray  # pyright: ignore[reportAttributeAccessIssue]

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: np.ndarray,
        **kwargs: Any,
    ) -> Any:
        """Override the default behaviour of universal functions

        All inputs of type :class:`Tensor` are converted to
        :class:`numpy.ndarray` before passing them to the universal function.

        Returns
        -------
        Any
            The result of the universal function.
        """
        inputs = tuple(i.view(np.ndarray) if isinstance(i, Tensor) else i for i in inputs)
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Override the default behaviour of :mod:`numpy` functions.

        All arguments of type :class:`Tensor` are converted to
        :class:`numpy.ndarray` before passing them to the :mod:`numpy` function.

        Returns
        -------
        Any
            The result of the :mod:`numpy` function.
        """
        args = tuple(a.view(np.ndarray) if isinstance(a, Tensor) else a for a in args)
        return func(*args, **kwargs)

    @classmethod
    def from_voigt(cls, voigt_array: ArrayLike) -> Self:
        """Create a tensor from an array in voigt notation.

        Parameters
        ----------
        voigt_array : numpy.typing.ArrayLike
            Voigt array.

        Raises
        ------
        ValueError
            If the voigt array does not have the appropriate shape.

        Returns
        -------
        typing.Self
            The tensor.
        """
        array = np.asarray(voigt_array)

        if array.shape != cls.voigt_shape:
            msg = f"input must be of shape {cls.voigt_shape}."
            raise ValueError(msg)

        # check if voigt symmetric

        return cls(array)

    def transform(self, matrix: ArrayLike) -> Self:
        r"""Apply a transformation matrix to the tensor.

        Only transformations that preserve the Euclidean metric are allowed,
        that is, rotations and reflections. The transformation matrix must
        therefore be orthogonal.

        Parameters
        ----------
        matrix : numpy.typing.ArrayLike
            The transformation matrix.

        Returns
        -------
        typing.Self
            The transformed tensor.

        Raises
        ------
        ValueError
            If the transformation matrix does not have the correct shape.
        ValueError
            If the transformation is not orthogonal.

        Notes
        -----
        Using Einstein summation convention, the components of a tensor
        rank :math:`N` :math:`T` transform under an orthogonal change of
        basis :math:`Q` to the components :math:`T'` in the new basis according
        to the following rule.

        .. math::
            T'_{i_1 i_2 \dots i_N} = Q_{i_1 j_1} \, Q_{i_2 j_2} \dots
            Q_{i_N j_N} \, T_{j_1 j_2 \dots j_N}

        """
        array = np.asarray(matrix)

        if array.shape != (self.dimension, self.dimension):
            msg = f"transformation matrix must be of shape {(self.dimension, self.dimension)}."
            raise ValueError(msg)
        if not np.allclose(array @ array.T, np.eye(self.dimension), atol=1e-8):
            msg = "transformation matrix must be orthogonal."
            raise ValueError(msg)

        # build the Einstein summation string
        out_indices = string.ascii_lowercase[: self.rank]
        in_indices = string.ascii_lowercase[self.rank : 2 * self.rank]
        q_indices = ",".join(
            f"{out_index}{in_index}"
            for out_index, in_index in zip(out_indices, in_indices, strict=True)
        )
        einsum_string = f"{q_indices},{in_indices}->{out_indices}"
        einsum_args = [array] * self.rank + [self]
        return np.einsum(einsum_string, *einsum_args)
