# Copyright 2025 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL
"""Collision tracking buffer interface for WarpX collision tracking functionality."""

import numpy as np

from ._libwarpx import libwarpx
from .LoadThirdParty import load_cupy


class CollisionBufferWrapper(object):
    """Wrapper around collision tracking buffers.
    This provides a convenient way to query and manipulate collision tracking data
    similar to how ParticleBoundaryBufferWrapper works for particle boundary data.

    Example usage:
        collision_tracker = collision_trackers.CollisionBufferWrapper()
        collision_tracker.clear_buffer()  # Reset all tracking data to zero

        # Get tracking data for a specific collision and process
        counts, energy = collision_tracker.get('coll_elec', 'excitation1', level=0)

        # Get all tracking data for a collision
        all_data = collision_tracker.get_all('coll_elec', level=0)
    """

    def __init__(self):
        pass

    def get_collision(self, collision_name):
        """
        Get the collision object by name.

        Parameters
        ----------
        collision_name : str
            Name of the collision (e.g., 'coll_elec', 'coll_ion')

        Returns
        -------
        BackgroundMCCCollision or None
            The collision object if found and it's a BackgroundMCCCollision, None otherwise
        """
        try:
            return libwarpx.warpx.get_collision(collision_name)
        except AttributeError as e:
            msg = (
                "You must initialize WarpX before accessing collision tracking buffers."
            )
            raise AttributeError(msg) from e

    def get_process_names(self, collision_name):
        """
        Get the list of process names for a given collision.

        Parameters
        ----------
        collision_name : str
            Name of the collision

        Returns
        -------
        list of str
            Process names (e.g., ['elastic', 'excitation1', 'ionization'])
        """
        collision = self.get_collision(collision_name)
        if collision is None:
            raise ValueError(
                f"Collision '{collision_name}' not found or doesn't support tracking"
            )

        return collision.get_process_names()

    def get(
        self,
        collision_name,
        process_name,
        level=0,
        copy_to_host=True,
        gather=True,
        energy_units="eV",
    ):
        """
        Get collision tracking data for a specific collision and process.

        Parameters
        ----------
        collision_name : str
            Name of the collision (e.g., 'coll_elec')
        process_name : str
            Name of the process (e.g., 'excitation1', 'ionization')
        level : int, optional
            AMR refinement level (default=0)
        copy_to_host : bool, optional
            Whether to copy GPU data to host (default=True)
        gather : bool, optional
            Whether to gather data from all boxes/ranks into a single array (default=True).
            If False, returns data from local boxes only.
        energy_units : str, optional
            Units for energy transfer: 'eV' or 'J' (default='eV')

        Returns
        -------
        tuple of arrays (counts, energy_transfer)
            counts : numpy/cupy array
                Number of collisions per cell for this process
            energy_transfer : numpy/cupy array
                Total energy transferred per cell for this process (in specified units)
        """
        collision = self.get_collision(collision_name)
        if collision is None:
            raise ValueError(
                f"Collision '{collision_name}' not found or doesn't support tracking"
            )

        # Get the process names and find the index
        process_names = collision.get_process_names()
        if process_name not in process_names:
            raise ValueError(
                f"Process '{process_name}' not found in collision '{collision_name}'. "
                f"Available processes: {process_names}"
            )

        process_idx = process_names.index(process_name)
        num_processes = len(process_names)

        if gather:
            # Use C++ gather method for proper multi-box handling
            data_flat, box_lo, box_hi = collision.gather_collision_tracking(
                level, ngrow=0
            )

            if len(data_flat) == 0:
                return None, None

            # Convert flat data to numpy array
            data_flat = np.array(data_flat, dtype=np.float64)

            # Calculate box dimensions
            dims = [box_hi[i] - box_lo[i] + 1 for i in range(len(box_lo))]
            ncomp = 2 * num_processes

            # Reshape: data is stored as (ncells * ncomp) in row-major order
            ncells = len(data_flat) // ncomp
            tracking_data = data_flat.reshape((ncells, ncomp))

            # Further reshape to spatial dimensions
            if len(dims) == 3:
                tracking_data = tracking_data.reshape(dims[0], dims[1], dims[2], ncomp)
            elif len(dims) == 2:
                tracking_data = tracking_data.reshape(dims[0], dims[1], ncomp)
            elif len(dims) == 1:
                tracking_data = tracking_data.reshape(dims[0], ncomp)
        else:
            # Original method: get data from local boxes
            tracking_data = collision.get_collision_tracking(level)

            if tracking_data is None:
                return None, None

            # Convert MultiFab to array
            tracking_data = tracking_data.to_numpy(copy=True)

            # Handle list of arrays from multiple boxes
            if isinstance(tracking_data, list):
                if len(tracking_data) == 0:
                    return None, None
                elif len(tracking_data) == 1:
                    tracking_data = tracking_data[0]
                else:
                    # Concatenate multiple boxes along first axis
                    tracking_data = np.concatenate(tracking_data, axis=0)

        if not copy_to_host:
            # If user wants GPU array, convert numpy to cupy
            xp, cupy_status = load_cupy()
            if cupy_status is not None:
                libwarpx.amr.Print(cupy_status)
            if xp is not None:
                tracking_data = xp.array(tracking_data, copy=True)

        # Extract the specific process data from interleaved layout
        # Even components (0, 2, 4, ...): collision counts
        # Odd components (1, 3, 5, ...): energy transfer
        counts = tracking_data[..., 2 * process_idx]
        energy_transfer = tracking_data[..., 2 * process_idx + 1]

        # Convert energy units if requested
        if energy_units == "eV":
            # Convert from Joules to eV
            energy_transfer = energy_transfer / 1.602e-19
        elif energy_units != "J":
            raise ValueError(
                f"Invalid energy_units '{energy_units}'. Must be 'eV' or 'J'."
            )

        return counts, energy_transfer

    def get_all(
        self, collision_name, level=0, copy_to_host=True, gather=True, energy_units="eV"
    ):
        """
        Get all collision tracking data for a collision.

        Parameters
        ----------
        collision_name : str
            Name of the collision
        level : int, optional
            AMR refinement level (default=0)
        copy_to_host : bool, optional
            Whether to copy GPU data to host (default=True)
        gather : bool, optional
            Whether to gather data from all boxes/ranks into a single array (default=True).
            If False, returns data from local boxes only.
        energy_units : str, optional
            Units for energy transfer: 'eV' or 'J' (default='eV')

        Returns
        -------
        dict
            Dictionary with process names as keys and (counts, energy) tuples as values.
            Each tuple contains:
            - counts: array of collision counts per cell
            - energy: array of energy transfer per cell (in specified units)
        """
        collision = self.get_collision(collision_name)
        if collision is None:
            raise ValueError(
                f"Collision '{collision_name}' not found or doesn't support tracking"
            )

        process_names = collision.get_process_names()
        num_processes = len(process_names)

        if gather:
            # Use C++ gather method for proper multi-box handling
            data_flat, box_lo, box_hi = collision.gather_collision_tracking(
                level, ngrow=0
            )

            if len(data_flat) == 0:
                return None

            # Convert flat data to numpy array
            data_flat = np.array(data_flat, dtype=np.float64)

            # Calculate box dimensions
            dims = [box_hi[i] - box_lo[i] + 1 for i in range(len(box_lo))]
            ncomp = 2 * num_processes

            # Reshape: data is stored as (ncells * ncomp) in row-major order
            ncells = len(data_flat) // ncomp
            tracking_data = data_flat.reshape((ncells, ncomp))

            # Further reshape to spatial dimensions
            if len(dims) == 3:
                tracking_data = tracking_data.reshape(dims[0], dims[1], dims[2], ncomp)
            elif len(dims) == 2:
                tracking_data = tracking_data.reshape(dims[0], dims[1], ncomp)
            elif len(dims) == 1:
                tracking_data = tracking_data.reshape(dims[0], ncomp)
        else:
            # Original method: get data from local boxes
            tracking_data = collision.get_collision_tracking(level)

            if tracking_data is None:
                return None

            # Convert MultiFab to array
            tracking_data = tracking_data.to_numpy(copy=True)

            # Handle list of arrays from multiple boxes
            if isinstance(tracking_data, list):
                if len(tracking_data) == 0:
                    return None
                elif len(tracking_data) == 1:
                    tracking_data = tracking_data[0]
                else:
                    # Concatenate multiple boxes along first axis
                    tracking_data = np.concatenate(tracking_data, axis=0)

        if not copy_to_host:
            # If user wants GPU array, convert numpy to cupy
            xp, cupy_status = load_cupy()
            if cupy_status is not None:
                libwarpx.amr.Print(cupy_status)
            if xp is not None:
                tracking_data = xp.array(tracking_data, copy=True)

        # Extract counts and energy from interleaved layout
        # Even components: counts, Odd components: energy
        counts_all = tracking_data[..., ::2]  # [0, 2, 4, 6, ...]
        energy_all = tracking_data[..., 1::2]  # [1, 3, 5, 7, ...]

        # Convert energy units if requested
        if energy_units == "eV":
            # Convert from Joules to eV
            energy_all = energy_all / 1.602e-19
        elif energy_units != "J":
            raise ValueError(
                f"Invalid energy_units '{energy_units}'. Must be 'eV' or 'J'."
            )

        # Build dictionary with process names as keys and (counts, energy) tuples as values
        result = {}
        for i, process_name in enumerate(process_names):
            result[process_name] = (counts_all[..., i], energy_all[..., i])

        return result

    def clear_buffers(self, collision_names=None, level=0):
        """
        Clear (zero out) collision tracking buffers for a list of collisions.

        Parameters
        ----------
        collision_names : list of str or None
            List of collision names to clear. TODO: If None, clear all collisions (not implemented yet).
        level : int, optional
            AMR refinement level (default=0)
        """
        if collision_names is None:
            raise NotImplementedError(
                "Clearing all collision trackers is not implemented yet."
            )

        for cname in collision_names:
            collision = self.get_collision(cname)
            if collision is None:
                raise ValueError(
                    f"Collision '{cname}' not found or doesn't support tracking"
                )

            collision.reset_collision_tracking(level)

    def save_to_file(self, collision_name, filename, rank, level=0, energy_units="eV"):
        """
        Save collision tracking data to a numpy .npz file.

        Parameters
        ----------
        collision_name : str
            Name of the collision
        filename : str
            Output filename (will add .npz if not present)
        rank : int
            MPI rank number
        level : int, optional
            AMR refinement level (default=0)
        energy_units : str, optional
            Units for energy transfer: 'eV' or 'J' (default='eV')
        """
        collision = self.get_collision(collision_name)
        if collision is None:
            raise ValueError(
                f"Collision '{collision_name}' not found or doesn't support tracking"
            )

        data = self.get_all(
            collision_name, level=level, copy_to_host=True, energy_units=energy_units
        )

        if rank == 0:
            if data is None:
                print(
                    f"Warning: No tracking data available for collision '{collision_name}'"
                )
                return

            # Ensure .npz extension
            if not filename.endswith(".npz"):
                filename += ".npz"

            # Build save dictionary with process names as keys and (counts, energy) tuples as values
            save_dict = {
                "collision_name": collision_name,
                "info": f"Each process name key contains a tuple of (counts, energy) arrays. Energy units: {energy_units}",
            }
            # Add each process as a separate item
            for process_name, (counts, energy) in data.items():
                save_dict[process_name] = (counts, energy)

            np.savez(filename, **save_dict)
