# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Adam Collin
#    Copyright (C) 2017-2026 Mathew Topper
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
This collection of functions perform simple processes on the input data.

.. module:: utils
   :synopsis: Input data processing.

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
.. moduleauthor:: Mathew Topper <damm_horse@yahoo.co.uk>
"""

import operator
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import spatial
from shapely.geometry import Point, Polygon

PointTuple = tuple[float, float, float]


def hydro_process(
    power_factor: Sequence[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    """Structural placeholder for the hydrodynamic processing to be shifted to
    the electrical design module.

    Args:
        unknown

    Attributes:
        unknown

    Returns:
        list: histogram of array power output; val1 = Bin edge [pc of array
            installed power], val2 = frequency of occurrence [pc]

    Note:
        This function needs to be completed. It now responds to the number of
        values requested in the power factor.

    """

    # get power outputs for assessment from power_factor
    p_out = []

    for power in power_factor:
        p_out.append(power[0])

    probability = 1.0 / len(p_out)

    return p_out, [probability] * len(p_out)


def seabed_range(bathy_data: pd.DataFrame) -> tuple[float, float]:
    """Get the min and max water depth from the given bathymetry data.

    Returns:
        min_ (float) [m]
        max_ (float) [m]

    """

    min_ = abs(bathy_data["layer 1 start"]).min()
    max_ = abs(bathy_data["layer 1 start"]).max()

    return min_, max_


def device_footprints_from_coords(
    layout: dict[str, tuple[float, float]],
    footprint: list[PointTuple],
) -> list[Polygon]:
    """Get the device footprint using coordinate system."""

    all_exclusions: list[Polygon] = []

    for value in layout.values():
        exclusion = []

        for point in footprint:
            new_point = tuple(map(operator.add, value[:2], point[:2]))
            exclusion.append(new_point)

        all_exclusions.append(Polygon(exclusion))

    return all_exclusions


def device_footprints_from_rad(
    layout: dict[str, tuple[float, float]],
    radius: float,
) -> list[Polygon]:
    """Get the device footprint from radius."""

    all_exclusions: list[Polygon] = []

    for value in layout.values():
        all_exclusions.append(Point(value).buffer(radius))

    return all_exclusions


def ideal_power_quantities(
    seastate_occurrence: Sequence[float],
    n_devices: int,
    device_power: float,
) -> tuple[float, list[float]]:
    """Calculate array power output assuming no losses. Used for efficiency
    calculations later in module.

    Args:
        seastate_occurrence (list):

    Attributes:
        bins
        bin_edges (list):

    Returns:
        ideal_annual_yield (float)
        ideal_histogram (list)

    """

    bins = 1.0 / len(seastate_occurrence)
    bin_edges = np.linspace(bins, 1, len(seastate_occurrence))

    year_hours = 365 * 24
    ideal_annual_yield = 0
    ideal_histogram: list[float] = []
    array_power = n_devices * device_power

    for time, power in zip(seastate_occurrence, bin_edges):
        ideal_annual_yield += time * year_hours * array_power
        ideal_histogram.append(power * array_power)

    return ideal_annual_yield, ideal_histogram


def get_bin_edges(power_factor: Sequence[tuple[float, float]]) -> list[float]:
    """Get bin edges of power factor var for analysis.

    Args:
        power_factor (list): structured input of user input power factor data.

    Returns:
         (list): bin edges

    """

    return [edge[0] for edge in power_factor]


def snap_to_grid(
    grid_points: pd.DataFrame,
    point: tuple[float, ...],
) -> tuple[tuple[float, ...], int]:
    """Snap a point to the grid.

    Args:
        grid_points (pd.DataFrame)
        point (tuple) [m]: Coordinates of point under consideration, x and
                y coordinates.

    Attributes:
        new_coords (list) [m]: Coordinates of nearest point, x, y and z.

    Returns:
        tuple

    """

    grid = np.array(grid_points[["x", "y"]])

    new_coords = grid[spatial.KDTree(grid).query(np.array(point))[1]].tolist()

    # and add z coord
    z = grid_points[
        (grid_points.x == new_coords[0]) & (grid_points.y == new_coords[1])
    ]["layer 1 start"].values[0]

    grid_id = int(
        grid_points[
            (grid_points.x == new_coords[0]) & (grid_points.y == new_coords[1])
        ]["id"].values[0]
    )

    new_coords.append(z)
    new_coords = [float(i) for i in new_coords]

    return ((tuple(new_coords)), grid_id)


def convert_df_column_type(
    df: pd.DataFrame,
    ids: Sequence[str],
    data_type: Any,
) -> pd.DataFrame:
    """Convert pandas DataFrame column data types.

    Args:
        df (pd.DataFrame) [-]: DataFrame to be updated.
        ids (list) [-]: List of columns to be updated.
        data_type () []: Data type to be applied to columns.

    Returns:
        df (pd.DataFrame) [-]: Updated DataFrame.

    """

    df[ids] = df[ids].astype(data_type)

    return df


def set_burial_from_bpi(row: dict[str, str]) -> float:
    """Function to code the bpi for burial depths.

    Args:
        row (pd) [-]: Row of pandas dataframe.

    Attributes:
        bpi (float) [m]: Burial depth from burial protection index.

    Returns:
        bpi

    """

    if row["layer 1 type"] in [
        "hard glacial till",
        "cemented",
        "soft rock coral",
        "hard rock",
        "gravel cobble",
    ]:
        bpi = 0.0

    elif "sand" in row["layer 1 type"]:
        bpi = 0.5

    elif "clay" in row["layer 1 type"]:
        bpi = 1.0

    else:
        errStr = "Sediment type '{}' is not recognised".format(
            row["layer 1 type"]
        )
        raise ValueError(errStr)

    return bpi
