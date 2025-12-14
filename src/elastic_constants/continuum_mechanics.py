"""Continuum mechanics module."""

import numpy as np

from numpy.typing import NDArray

from elastic_constants.tensor import Tensor


__all__ = ["Compliance", "DeformationGradient", "Elasticity", "Strain", "Stress"]


class DeformationGradient(Tensor):
    """Deformation gradient tensor class."""

    rank = 2


class Strain(Tensor):
    """Strain tensor class."""

    rank = 2
    voigt_symmetric = True
    voigt_scale = np.asarray((1, 1, 1, 2, 2, 2), dtype=np.double)


class Stress(Tensor):
    """Stress tensor class."""

    rank = 2
    voigt_symmetric = True

    def to_second_piola_kirchhoff(
        self, deformation_gradient: DeformationGradient
    ) -> NDArray[np.double]:
        r"""Convert Cauchy stress to second Piola-Kirchhoff stress tensor.

        Parameters
        ----------
        deformation_gradient : DeformationGradient
            Deformation gradient tensor.

        Returns
        -------
        numpy.typing.NDArray[numpy.double]
            Second Piola-Kirchhoff stress tensor.

        Notes
        -----
        The second Piola-Kirchhoff stress tensor :math:`S` is related to the
        Cauchy stress tensor :math:`\sigma` by the relation

        .. math::
            S = J \, F^{-1} \cdot \sigma \cdot F^{-T}

        where :math:`J` is the determinant of the deformation gradient
        :math:`F`.
        """
        j = np.linalg.det(deformation_gradient)
        f_inv = np.linalg.inv(deformation_gradient)
        return j * f_inv @ self @ f_inv.T


class Elasticity(Tensor):
    """Elasticity tensor class."""

    rank = 4
    voigt_symmetric = True


class Compliance(Tensor):
    """Compliance tensor class."""

    rank = 4
    voigt_symmetric = True
    voigt_scale = np.asarray(
        (
            (1, 1, 1, 2, 2, 2),
            (1, 1, 1, 2, 2, 2),
            (1, 1, 1, 2, 2, 2),
            (2, 2, 2, 4, 4, 4),
            (2, 2, 2, 4, 4, 4),
            (2, 2, 2, 4, 4, 4),
        ),
        dtype=np.double,
    )
