# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""Power deposition tracking buffer interface for WarpX power deposition tracking functionality."""

import math

import numpy as np

from ._libwarpx import libwarpx
from .LoadThirdParty import load_cupy


class PowerDepositionTrackerWrapper(object):
    """Wrapper around per-species power deposition (J.E) tracking buffers.
    This provides a convenient way to query and manipulate power deposition
    tracking data, similar to how CollisionBufferWrapper works for collision
    tracking data.

    Each species that has ``enable_power_deposition_tracking`` set gets a
    per-cell, 3-component buffer (Px, Py, Pz) accumulating the J.E power
    deposited into that species by the fields during its particle push. The
    values are sampled at the exact velocity and field values used in the
    momentum push, so they are self-consistent with the numerical push itself
    rather than an independently-evaluated physical quantity.

    Example usage:
        power_tracker = power_deposition_trackers.PowerDepositionTrackerWrapper()
        power_tracker.clear_buffers(["electrons"])  # Reset tracking data to zero

        # Get raw per-cell power for one species
        px, py, pz = power_tracker.get("electrons", level=0)

        # Get per-cell power density (divided by cell length/area/volume)
        px, py, pz = power_tracker.get("electrons", level=0, normalize=True)
    """

    def __init__(self):
        pass

    def get_species(self, species_name):
        """
        Get the species particle container by name.

        Parameters
        ----------
        species_name : str
            Name of the species (e.g., 'electrons', 'ions')

        Returns
        -------
        WarpXParticleContainer or None
            The species particle container if found, None otherwise
        """
        try:
            return libwarpx.warpx.multi_particle_container().get_particle_container_from_name(
                species_name
            )
        except AttributeError as e:
            msg = "You must initialize WarpX before accessing power deposition tracking buffers."
            raise AttributeError(msg) from e

    def _cell_volume(self, level):
        """Auto-compute the cell length (1D) / area (2D) / volume (3D) from the
        simulation geometry. Not valid for RZ or other non-Cartesian geometries,
        since the true cell volume there depends on radius."""
        if libwarpx.geometry_dim in ("rz", "rcylinder", "rsphere"):
            raise NotImplementedError(
                f"Automatic cell-volume normalization is not supported for "
                f"geometry_dim='{libwarpx.geometry_dim}' (cell volume depends on "
                f"radius there). Pass normalize=False or an explicit cell_volume."
            )
        dx = libwarpx.warpx.Geom(level).data().CellSize()
        return math.prod(dx)

    def get(
        self,
        species_name,
        level=0,
        copy_to_host=True,
        gather=True,
        normalize=False,
        cell_volume=None,
    ):
        """
        Get power deposition tracking data for a specific species.

        Parameters
        ----------
        species_name : str
            Name of the species (e.g., 'electrons')
        level : int, optional
            AMR refinement level (default=0)
        copy_to_host : bool, optional
            Whether to copy GPU data to host (default=True)
        gather : bool, optional
            Whether to gather data from all boxes/ranks into a single array (default=True).
            If False, returns data from local boxes only.
        normalize : bool, optional
            If True, divide the raw per-cell values by a cell-volume-like constant
            (cell length in 1D, area in 2D, volume in 3D) to give a density-like
            quantity (default=False, i.e. raw weight-summed power).
        cell_volume : float or array-like, optional
            Manual override for the normalization constant used when
            ``normalize=True``. If not given, it is auto-computed from the
            simulation geometry (only supported for Cartesian geometries -- pass
            this explicitly for RZ/cylindrical/spherical geometries).

        Returns
        -------
        tuple of arrays (Px, Py, Pz)
            Per-cell power deposited into this species, one array per direction.
        """
        species = self.get_species(species_name)
        if species is None:
            raise ValueError(f"Species '{species_name}' not found")

        if gather:
            data_flat, box_lo, box_hi = species.gather_power_deposition_tracking(
                level, ngrow=0
            )

            if len(data_flat) == 0:
                return None, None, None

            data_flat = np.array(data_flat, dtype=np.float64)

            dims = [box_hi[i] - box_lo[i] + 1 for i in range(len(box_lo))]
            ncomp = 3

            ncells = len(data_flat) // ncomp
            tracking_data = data_flat.reshape((ncells, ncomp))

            if len(dims) == 3:
                tracking_data = tracking_data.reshape(dims[0], dims[1], dims[2], ncomp)
            elif len(dims) == 2:
                tracking_data = tracking_data.reshape(dims[0], dims[1], ncomp)
            elif len(dims) == 1:
                tracking_data = tracking_data.reshape(dims[0], ncomp)
        else:
            tracking_data = species.get_power_deposition_tracking(level)

            if tracking_data is None:
                return None, None, None

            tracking_data = tracking_data.to_numpy(copy=True)

            if isinstance(tracking_data, list):
                if len(tracking_data) == 0:
                    return None, None, None
                elif len(tracking_data) == 1:
                    tracking_data = tracking_data[0]
                else:
                    tracking_data = np.concatenate(tracking_data, axis=0)

        if not copy_to_host:
            xp, cupy_status = load_cupy()
            if cupy_status is not None:
                libwarpx.amr.Print(cupy_status)
            if xp is not None:
                tracking_data = xp.array(tracking_data, copy=True)

        if normalize:
            dv = cell_volume if cell_volume is not None else self._cell_volume(level)
            tracking_data = tracking_data / dv

        px = tracking_data[..., 0]
        py = tracking_data[..., 1]
        pz = tracking_data[..., 2]

        return px, py, pz

    def get_all(
        self,
        species_names,
        level=0,
        copy_to_host=True,
        gather=True,
        normalize=False,
        cell_volume=None,
    ):
        """
        Get power deposition tracking data for multiple species at once.

        Parameters
        ----------
        species_names : list of str
            Names of the species to query
        level : int, optional
            AMR refinement level (default=0)
        copy_to_host : bool, optional
            Whether to copy GPU data to host (default=True)
        gather : bool, optional
            Whether to gather data from all boxes/ranks into a single array (default=True)
        normalize : bool, optional
            If True, divide by a cell-volume-like constant (default=False)
        cell_volume : float or array-like, optional
            Manual override for the normalization constant

        Returns
        -------
        dict
            Dictionary with species names as keys and (Px, Py, Pz) tuples as values.
        """
        result = {}
        for species_name in species_names:
            result[species_name] = self.get(
                species_name,
                level=level,
                copy_to_host=copy_to_host,
                gather=gather,
                normalize=normalize,
                cell_volume=cell_volume,
            )
        return result

    def clear_buffers(self, species_names, level=0):
        """
        Clear (zero out) power deposition tracking buffers for a list of species.

        Parameters
        ----------
        species_names : list of str
            List of species names to clear.
        level : int, optional
            AMR refinement level (default=0)
        """
        for species_name in species_names:
            species = self.get_species(species_name)
            if species is None:
                raise ValueError(f"Species '{species_name}' not found")

            species.reset_power_deposition_tracking(level)

    def save_to_file(
        self, species_name, filename, rank, level=0, normalize=False, cell_volume=None
    ):
        """
        Save power deposition tracking data to a numpy .npz file.

        Parameters
        ----------
        species_name : str
            Name of the species
        filename : str
            Output filename (will add .npz if not present)
        rank : int
            MPI rank number
        level : int, optional
            AMR refinement level (default=0)
        normalize : bool, optional
            If True, divide by a cell-volume-like constant (default=False)
        cell_volume : float or array-like, optional
            Manual override for the normalization constant
        """
        px, py, pz = self.get(
            species_name,
            level=level,
            copy_to_host=True,
            normalize=normalize,
            cell_volume=cell_volume,
        )

        if rank == 0:
            if px is None:
                print(
                    f"Warning: No power deposition tracking data available for species '{species_name}'"
                )
                return

            if not filename.endswith(".npz"):
                filename += ".npz"

            np.savez(
                filename,
                species_name=species_name,
                info="Px, Py, Pz are per-cell J.E power deposited into this species.",
                Px=px,
                Py=py,
                Pz=pz,
            )
