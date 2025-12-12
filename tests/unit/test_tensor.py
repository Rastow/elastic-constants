import itertools
import math
import re

from collections.abc import Callable
from typing import Any
from typing import NotRequired
from typing import Required
from typing import TypedDict
from typing import Unpack

import numpy as np
import pytest

from hypothesis import assume, settings
from hypothesis import given
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
from hypothesis.strategies import just
from hypothesis.strategies import one_of
from numpy.typing import NDArray

from elastic_constants.tensor import Tensor


def numeric_arrays(
    shape: tuple[int, ...] | SearchStrategy[tuple[int, ...]],
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
            "min_value": -1e8,
            "max_value": 1e8,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
            "min_magnitude": 0,
            "max_magnitude": 1e8,
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
    rank = array.ndim

    # no symmetrization required for scalars and vectors
    if rank < 2:  # noqa: PLR2004
        return array

    # symmetrize over each pair of adjacent indices
    index_pairs = [{i, i + 1} for i in range(rank % 2, rank, 2)]
    for index_pair in index_pairs:
        array = symmetrize(array, index_pair)
    return array


@composite
def valid_arrays(draw: DrawFn, min_rank: int = 0, max_rank: int = 6) -> NDArray[np.number]:
    """Generate valid arrays for Tensor instantiation."""
    rank = draw(integers(min_value=min_rank, max_value=max_rank))
    dimension = 3
    shapes = array_shapes(min_dims=rank, max_dims=rank, min_side=dimension, max_side=dimension)
    return draw(numeric_arrays(shapes))


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
def valid_tensors(
    draw: DrawFn,
    *,
    min_rank: int = 0,
    max_rank: int = 6,
    voigt_symmetric: bool = False,
    generate_voigt_scale: bool = False,
) -> Tensor:
    """Generate valid Tensor instances."""
    # draw a valid array and optionally symmetrize it
    array = draw(valid_arrays(min_rank=min_rank, max_rank=max_rank))
    array = array if not voigt_symmetric else voigt_symmetrize(array)

    # create a mock tensor subclass with matching rank
    rank = array.ndim
    kwargs: TensorAttributes = {
        "rank": rank,
        "voigt_symmetric": voigt_symmetric,
        "voigt_scale": np.ones(voigt_shape(rank), dtype=array.dtype)
        if not generate_voigt_scale
        else draw(
            numeric_arrays(voigt_shape(rank)).filter(lambda x: not np.any(np.isclose(x, 0)))
        ),
    }

    # return the instantiated tensor
    return tensor_subclass(**kwargs)(array)


@settings(max_examples=10)
@given(valid_arrays())
def test_tensor_base_class_instantiation(array: NDArray[np.number]) -> None:
    err_msg = "only tensor subclasses may be instantiated."
    with pytest.raises(TypeError, match=err_msg):
        _ = Tensor(array)


def test_tensor_subclass_missing_rank() -> None:
    err_msg = "tensor must define its rank as a class variable."
    with pytest.raises(AttributeError, match=err_msg):
        _ = type("NoRankTensor", (Tensor,), {})


@settings(max_examples=10)
@given(integers(min_value=0, max_value=6), numeric_arrays(array_shapes(min_dims=0, max_dims=6)))
def test_tensor_subclass_invalid_voigt_scale_shape(
    rank: int, voigt_scale: NDArray[np.number]
) -> None:
    assume(voigt_scale.shape != voigt_shape(rank))
    err_msg = re.escape(f"must be of shape {voigt_shape(rank)}.")
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor_subclass(rank=rank, voigt_scale=voigt_scale)


@settings(max_examples=10)
@given(numeric_arrays(integers(min_value=0, max_value=6).map(voigt_shape)))
def test_tensor_subclass_invalid_voigt_scale_values(voigt_scale: NDArray[np.number]) -> None:
    indices = np.random.choice((True, False), voigt_scale.shape)
    voigt_scale[indices] = 0
    rank = voigt_scale.ndim * 2 - (
        0 if voigt_scale.ndim == 0 or voigt_scale.shape[0] != 3 else 1
    )
    err_msg = "must not contain zero."
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor_subclass(rank=rank, voigt_scale=voigt_scale)


@settings(max_examples=10)
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
    assume(any(side != dimension for side in array.shape))
    subclass = tensor_subclass(rank=array.ndim)
    err_msg = f"must be {subclass.dimension}-dimensional."
    with pytest.raises(ValueError, match=err_msg):
        _ = subclass(array)


@settings(max_examples=10)
@given(valid_arrays())
def test_tensor_instantiation_not_voigt_symmetric(array: NDArray[np.number]) -> None:
    assume(not np.allclose(array, voigt_symmetrize(array)))
    err_msg = "must be voigt symmetric."
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor_subclass(rank=array.ndim, voigt_symmetric=True)(array)


@settings(max_examples=10)
@given(
    arrays(
        one_of(byte_string_dtypes(), unicode_string_dtypes(), datetime64_dtypes()),
        integers(min_value=0, max_value=6).map(voigt_shape),
    )
)
def test_tensor_from_voigt_non_numeric_data(voigt_array: NDArray[np.number]) -> None:
    dimension = 3
    rank = voigt_array.ndim * 2 - (
        0 if voigt_array.ndim == 0 or voigt_array.shape[0] != dimension else 1
    )
    err_msg = "voigt array must only contain numeric data types."
    with pytest.raises(TypeError, match=err_msg):
        _ = tensor_subclass(rank=rank).from_voigt(voigt_array)


@settings(max_examples=10)
@given(integers(min_value=0, max_value=6), numeric_arrays(array_shapes(min_dims=0, max_dims=6)))
def test_tensor_from_voigt_invalid_shape(rank: int, voigt_array: NDArray[np.number]) -> None:
    assume(voigt_shape(rank) != voigt_array.shape)
    err_msg = re.escape(f"voigt array must be of shape {voigt_shape(rank)}.")
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor_subclass(rank=rank, voigt_symmetric=True).from_voigt(voigt_array)


@given(numeric_arrays(integers(min_value=0, max_value=6).map(voigt_shape)))
def test_tensor_from_voigt(voigt_array: NDArray[np.number]) -> None:
    dimension = 3
    rank = voigt_array.ndim * 2 - (
        0 if voigt_array.ndim == 0 or voigt_array.shape[0] != dimension else 1
    )
    tensor = tensor_subclass(rank=rank).from_voigt(voigt_array)
    assert tensor._array.shape == (dimension,) * rank  # type: ignore [reportPrivateUsage]
    assert tensor.is_voigt_symmetric()


@settings(max_examples=10)
@given(valid_tensors(voigt_symmetric=False))
def test_tensor_to_voigt_not_voigt_symmetric(tensor: Tensor) -> None:
    assume(not tensor.is_voigt_symmetric())
    err_msg = "tensor must be voigt symmetric."
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor.to_voigt()


@given(valid_tensors(voigt_symmetric=True, generate_voigt_scale=True))
def test_tensor_to_voigt(tensor: Tensor) -> None:
    voigt_array = tensor.to_voigt()
    assert voigt_array.shape == voigt_shape(tensor.rank)


@given(valid_tensors(voigt_symmetric=True, generate_voigt_scale=True))
def test_tensor_from_voigt_inverts_to_voigt(tensor: Tensor) -> None:
    voigt_array = tensor.to_voigt()
    reconstructed_tensor = tensor.from_voigt(voigt_array)
    np.testing.assert_allclose(reconstructed_tensor, tensor, atol=1e-6)


@settings(max_examples=10)
@given(
    valid_tensors(),
    arrays(
        dtype=floating_dtypes(sizes=64),
        shape=array_shapes(),
        elements={
            "min_value": -1,
            "max_value": 1,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
        },
    ),
)
def test_tensor_transform_incorrect_shape(
    tensor: Tensor, transformation_matrix: NDArray[np.floating]
) -> None:
    assume(transformation_matrix.shape != (3, 3))
    err_msg = re.escape("transformation matrix must be of shape (3, 3).")
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor.transform(transformation_matrix)


@settings(max_examples=10)
@given(
    valid_tensors(),
    arrays(
        dtype=floating_dtypes(sizes=64),
        shape=(3, 3),
        elements={
            "min_value": -1,
            "max_value": 1,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
        },
    ),
)
def test_tensor_transform_non_orthogonal(
    tensor: Tensor, transformation_matrix: NDArray[np.floating]
) -> None:
    assume(not np.allclose(transformation_matrix @ transformation_matrix.T, np.eye(3)))
    err_msg = "transformation matrix must be orthogonal."
    with pytest.raises(ValueError, match=err_msg):
        _ = tensor.transform(transformation_matrix)


@given(
    valid_tensors(max_rank=4),
    arrays(
        dtype=floating_dtypes(sizes=64),
        shape=(3, 3),
        elements={
            "min_value": -1,
            "max_value": 1,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
        },
    )
    .map(lambda m: np.linalg.qr(m)[0])
    .filter(lambda m: np.linalg.cond(m) < 100),  # noqa: PLR2004
)
def test_tensor_transform_inverts_inverse_transform(
    tensor: Tensor, transformation_matrix: NDArray[np.floating]
) -> None:
    # generate orthogonal matrices using qr decomposition
    # filter ill-conditioned matrices to avoid numerical instability
    transformed_tensor = tensor.transform(transformation_matrix)
    reconstructed_tensor = transformed_tensor.transform(np.transpose(transformation_matrix))
    np.testing.assert_allclose(tensor, reconstructed_tensor, atol=1e-6)


@given(
    valid_tensors(max_rank=4, voigt_symmetric=True),
    arrays(
        dtype=floating_dtypes(sizes=64),
        shape=(3, 3),
        elements={
            "min_value": -1e8,
            "max_value": 1e8,
            "allow_nan": False,
            "allow_infinity": False,
            "allow_subnormal": True,
        },
    )
    .map(lambda m: np.linalg.qr(m)[0])
    .filter(lambda m: np.linalg.cond(m) < 100),  # noqa: PLR2004
)
def test_tensor_transform_preserves_voigt_symmetry(
    tensor: Tensor, transformation_matrix: NDArray[np.floating]
) -> None:
    transformed_tensor = tensor.transform(transformation_matrix)
    assert transformed_tensor.is_voigt_symmetric()


@given(
    one_of(valid_tensors(min_rank=2, max_rank=2), valid_arrays(min_rank=2, max_rank=2)),
    one_of(valid_tensors(min_rank=2, max_rank=2), valid_arrays(min_rank=2, max_rank=2)),
    one_of(just(np.add), just(np.matmul), just(np.greater)),
)
def test_tensor_ufunc(
    operand1: Tensor | NDArray[np.number],
    operand2: Tensor | NDArray[np.number],
    ufunc: Callable[[Any, Any], Any],
) -> None:
    result = ufunc(operand1, operand2)
    assert not isinstance(result, Tensor)
