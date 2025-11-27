import numpy as np
import pytest

from hypothesis.strategies import SearchStrategy
from hypothesis.strategies import one_of
from hypothesis.extra.numpy import complex_number_dtypes
from hypothesis.extra.numpy import floating_dtypes
from hypothesis.extra.numpy import integer_dtypes
from hypothesis.extra.numpy import unsigned_integer_dtypes
from hypothesis.extra.numpy import arrays
from hypothesis.extra.numpy import array_shapes

from numpy.typing import NDArray

def numeric_dtypes() -> SearchStrategy[np.dtype[np.number]]:
    return one_of(
        integer_dtypes(), unsigned_integer_dtypes(), floating_dtypes(), complex_number_dtypes()
    )


@pytest.fixture
def tensor_input() -> SearchStrategy[NDArray[np.number]]:
    # extra.numpy.array_dtypes
    # extra.numpy.array_shapes
    # ranks 0?-4 -> min_dims = 1, max_dims = 4
    # dimension 3 -> min_side = max_side = 3
    # extra.numpy.arrays
    # inputs
    return arrays(
        dtype = numeric_dtypes(),
        shape = array_shapes(min_dims=1, max_dims=6, min_side=3, max_side=3),

    )


def test_tensor_explicit_constructor(tensor: Tensor):
    # check for Tensor
    # explicit constructor call
    pass


def test_tensor_view_casting():
    # cast to ndarray
    pass


def test_tensor_slicing():
    # use basic_indices
    # creating new from template
    # cast to ndarray
    pass


# test all ufunc methods
def test_tensor_ufunc():
    # outputs are ndarray
    pass


def test_tensor_array_function():
    # outputs are ndarray
    pass
