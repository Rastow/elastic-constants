import itertools
import math

from typing import Any
from typing import NotRequired
from typing import Required
from typing import TypedDict
from typing import Unpack

import numpy as np
import pytest

from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis.extra.numpy import array_shapes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import byte_string_dtypes
from hypothesis.extra.numpy import complex_number_dtypes
from hypothesis.extra.numpy import datetime64_dtypes
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.extra.numpy import integer_dtypes
from hypothesis.extra.numpy import unicode_string_dtypes
from hypothesis.strategies import DrawFn
from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import one_of
from numpy.typing import NDArray

from elastic_constants.tensor import Tensor


def numeric_arrays(
    shape: tuple[int, ...] | SearchStrategy[tuple[int, ...]],
    *,
    min_value: float = -1e8,
    max_value: float = 1e8,
    min_magnitude: float = 0,
    max_magnitude: float = 1e8,
) -> SearchStrategy[NDArray[np.number]]:
    """Generate numeric arrays with the given shape.

    The arrays contain finite numbers within a reasonable range to avoid
    overflow during tensor operations.
    """
    return arrays(
        dtype=one_of(
            integer_dtypes(sizes=64), floating_dtypes(sizes=64), complex_number_dtypes(sizes=128)
        ),
        shape=shape,
        elements={
            "min_value": min_value,
            "max_value": max_value,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
            "min_magnitude": min_magnitude,
            "max_magnitude": max_magnitude,
        },
    )


def symmetrize(array: NDArray[np.number], indices: set[int]) -> NDArray[np.number]:
    """Symmetrize an array over the given set of axes."""
    mappings = [
        dict(zip(permutation, indices, strict=True))
        for permutation in itertools.permutations(sorted(indices))
    ]
    transposed_arrays = [
        np.transpose(array, axes=[mapping.get(i, i) for i in range(array.ndim)])
        for mapping in mappings
    ]
    return np.sum(transposed_arrays, axis=0) / len(mappings)


def voigt_symmetrize(array: NDArray[np.number]) -> NDArray[np.number]:
    """Symmetrize an array according to Voigt notation rules."""
    # no symmetrization required for scalars and vectors
    if array.ndim < 2:
        return array

    dimension = 3
    rank = array.ndim * 2 - (0 if array.ndim == 0 or array.shape[0] == dimension else 1)
    index_pairs = [{i, i + 1} for i in range(rank % 2, rank, 2)]
    for index_pair in index_pairs:
        array = symmetrize(array, index_pair)
    return array


def valid_arrays(rank: int = 0) -> SearchStrategy[NDArray[np.number]]:
    """Generate valid arrays for Tensor instantiation."""
    dimension = 3
    shapes = array_shapes(min_dims=rank, max_dims=rank, min_side=dimension, max_side=dimension)
    return numeric_arrays(shapes)


class TensorAttributes(TypedDict, total=False):
    rank: Required[int]
    voigt_symmetric: NotRequired[bool]
    voigt_scale: NotRequired[NDArray[np.number]]


def tensor_subclass(**kwargs: Unpack[TensorAttributes]) -> type[Tensor]:
    """Create a Tensor subclass with the given attributes."""
    rank: int = kwargs["rank"]
    name: str = f"Rank{rank}Tensor"
    bases: tuple[type[Tensor]] = (Tensor,)
    attrs: dict[str, Any] = dict(**kwargs)
    return type(name, bases, attrs)


def voigt_shape(rank: int, dimension: int = 3) -> tuple[int, ...]:
    """Get the Voigt shape for a given tensor rank."""
    return (dimension,) * (rank % 2) + (math.comb(dimension + 2 - 1, 2),) * (rank // 2)


@composite
def valid_tensor(
    draw: DrawFn,
    *,
    min_rank: int = 0,
    max_rank: int = 6,
    voigt_symmetric: bool = False,
    generate_voigt_scale: bool = False,
) -> Tensor:
    """Generate valid Tensor instances."""
    # draw a tensor rank
    rank = draw(integers(min_value=min_rank, max_value=max_rank))

    # draw a valid array and optionally symmetrize it
    array = draw(valid_arrays(rank=rank))
    array = array if not voigt_symmetric else voigt_symmetrize(array)

    # instantiate a mock tensor subclass with matching rank
    kwargs: TensorAttributes = {
        "rank": rank,
        "voigt_symmetric": voigt_symmetric,
        "voigt_scale": np.ones(voigt_shape(rank), dtype=array.dtype)
        if not generate_voigt_scale
        else draw(numeric_arrays(voigt_shape(rank), min_value=1e-8)),
    }
    return tensor_subclass(**kwargs)(array)


@settings(max_examples=10)
@given(valid_arrays())
def test_tensor_base_class_instantiation(array: NDArray[np.number]) -> None:
    err_msg = "only tensor subclasses may be instantiated."
    with pytest.raises(TypeError, match=err_msg):
        _ = Tensor(array)


@settings(max_examples=10)
@given(valid_arrays())
def test_tensor_subclass_instantiation_missing_rank(array: NDArray[np.number]) -> None:
    err_msg = "tensor must define its rank as a class variable."
    with pytest.raises(AttributeError, match=err_msg):
        _ = type("NoRankTensor", (Tensor,), {})(array)


# invalid voigt scale


@settings(max_examples=30)
@given(
    arrays(
        one_of(byte_string_dtypes(), unicode_string_dtypes(), datetime64_dtypes()),
        array_shapes(min_dims=0, max_dims=6, min_side=3, max_side=3),
    )
)
def test_tensor_instantiation_non_numeric_data(
    array: NDArray[np.bytes_ | np.str_ | np.datetime64],
) -> None:
    err_msg = "array must only contain numeric data types."
    with pytest.raises(TypeError, match=err_msg):
        _ = tensor_subclass(rank=array.ndim)(array)


@settings(max_examples=10)
@given(valid_arrays(), integers(min_value=0, max_value=6))
def test_tensor_instantiation_incorrect_rank(array: NDArray[np.number], rank: int) -> None:
    assume(rank != array.ndim)
    err_msg = f"must be rank {rank}."
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor_subclass(rank=rank)(array)


@settings(max_examples=10)
@given(numeric_arrays(array_shapes(min_dims=1, max_dims=6)))
def test_tensor_instantiation_incorrect_dimension(array: NDArray[np.number]) -> None:
    # scalars are excluded from this test as they have no dimension
    dimension = 3
    assume(set(array.shape) != {dimension})
    subclass = tensor_subclass(rank=array.ndim)
    err_msg = f"must be {subclass.dimension}-dimensional."
    with pytest.raises(ValueError, match=err_msg):
        _ = subclass(array)


@given(numeric_arrays(integers(min_value=0, max_value=6).map(voigt_shape)))
def test_tensor_from_voigt(voigt_array: NDArray[np.number]) -> None:
    dimension = 3
    rank = voigt_array.ndim * 2 - (0 if voigt_array.ndim == 0 or voigt_array.shape[0] == dimension else 1)
    tensor = tensor_subclass(rank=rank).from_voigt(voigt_array)
    assert tensor._array.shape == (dimension,) * rank  # type: ignore [reportPrivateUsage]
    assert tensor.is_voigt_symmetric()


@given(valid_tensor(voigt_symmetric=True))
def test_tensor_to_voigt(tensor: Tensor) -> None:
    voigt_array = tensor.to_voigt()
    assert voigt_array.shape == voigt_shape(tensor.rank)


@given(valid_tensor(voigt_symmetric=True, generate_voigt_scale=True))
def test_tensor_from_voigt_inverts_to_voigt(tensor: Tensor) -> None:
    voigt_array = tensor.to_voigt()
    reconstructed_tensor = tensor.from_voigt(voigt_array)
    np.testing.assert_array_equal(tensor, reconstructed_tensor)


# test tensor slicing, numpy function, numpy ufunc returns ndarray
# test numpy action on tensor return same result as acting on ndarray
# test tensor transformation: reflection, rotation
# test non-orthogonal transformations throw error
# from_voigt inverts to_voigt (care for dtype promotion)
# test voigt scaling is applied correctly
# generate symmetric tensors by averaging with transpose
# test symmetric tensor stays symmetric under transformation
# test copy


def test_tensor_ufunc():
    # outputs are ndarray
    pass


def test_tensor_array_function():
    # outputs are ndarray
    pass
