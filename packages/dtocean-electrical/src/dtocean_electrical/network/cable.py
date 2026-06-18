# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Adam Collin
#    Copyright (C) 2017-2018 Mathew Topper
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
Created on Thu Apr 07 13:38:45 2016

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
.. moduleauthor:: Mathew Topper <mathew.topper@dataonlygreater.com>
"""

from typing import Optional, Sequence

import pandas as pd


class Cable:
    """Class to collect all attributes of a cable object.

    Args:
        index (int) [-]: Cable identification number. Each cable is numbered
            sequentially from 0 to n, where n is the number of cables in the
            network.
        length (float) [m]: Cable length.
        db_key (Unknown) [-]: Reference to database object.
        marker (int) [-]: Unique network identification number.

    Attributes:
        voltage (float) [V]: The rated voltage.
        current (float) [A]: The rated current.
        r (float) [Ohm/km]: ac resistance at 90 degree.
        c (float) [uF/km]: capacitance per unit length.
        x (float) [Ohm/km]: inductive reactance per unit length.
        type_ (str) [-]: Cable type: array, export or umbilical.
        upstream_id (int) [-]: Marker of cable termination in the sea
            direction.
        downstream_id (int) [-]: Marker of cable termination in the shore
            direction.
        upstream_type (str) [-]: Type of cable termination in the sea
            direction.
        downstream_type (str) [-]: Type of cable termination in the shore
            direction.

    Note:
        The marker is associated with the network format data structure, where
        each component is given a unique marker.

    """

    def __init__(self, index: int, length: float, db_key: int, marker: int):
        # what attributes do cables have? These are mostly db but could be used
        # for identification
        self.voltage: Optional[float] = None
        self.current: Optional[float] = None
        self.r: Optional[float] = None
        self.c: Optional[float] = None
        self.x: Optional[float] = None

        # what attributes do we place on cables?
        self.id_ = index
        self.db_key = db_key
        self.type_: Optional[str] = None
        self.length = length
        self.upstream_id: Optional[int] = None
        self.downstream_id: Optional[int] = None
        self.upstream_type: Optional[str] = None
        self.downstream_type: Optional[str] = None
        self.marker = marker

    def __str__(self):
        """Override print command to display some info."""
        msg = "This is " + self.__class__.__name__ + " " + str(self.id_) + "."

        if self.upstream_type is not None and self.upstream_id is not None:
            msg += (
                " This cable connects from up: "
                + str(self.upstream_type)
                + " "
                + str(self.upstream_id)
            )

            if (
                self.downstream_type is not None
                and self.downstream_id is not None
            ):
                msg += (
                    " to down: "
                    + str(self.downstream_type)
                    + " "
                    + str(self.downstream_id)
                )

            msg += "."

        msg += (
            "\nThe cable length is: "
            + str(self.length)
            + ". The cable marker is: "
            + str(self.marker)
            + "."
        )

        return msg


class StaticCable(Cable):
    """Static cable definition.

    Args:
        route () [-]: Cable route.

    """

    def __init__(
        self,
        index: int,
        length: float,
        db_key: int,
        marker: int,
        route: Sequence[int],
        burial: Sequence[float],
        split_pipe: Sequence[bool],
    ):
        super(StaticCable, self).__init__(index, length, db_key, marker)
        self.route = route
        self.split_pipe = split_pipe
        self.target_burial_depth = burial


class ArrayCable(StaticCable):
    def __init__(
        self,
        index: int,
        length: float,
        db_key: int,
        marker: int,
        route: Sequence[int],
        burial: Sequence[float],
        split_pipe: Sequence[bool],
        upstream_type: str,
        downstream_type: str,
        upstream_id: int,
        downstream_id: int,
    ):
        super(ArrayCable, self).__init__(
            index,
            length,
            db_key,
            marker,
            route,
            burial,
            split_pipe,
        )
        self.type_ = "array"
        self.upstream_type = upstream_type
        self.upstream_id = upstream_id
        self.downstream_type = downstream_type
        self.downstream_id = downstream_id


class ExportCable(StaticCable):
    def __init__(
        self,
        index: int,
        length: float,
        db_key: int,
        marker: int,
        route: Sequence[int],
        burial: Sequence[float],
        split_pipe: Sequence[bool],
        upstream_type: str,
        upstream_id: int,
    ):
        super(ExportCable, self).__init__(
            index,
            length,
            db_key,
            marker,
            route,
            burial,
            split_pipe,
        )
        self.type_ = "export"
        self.downstream_type = "Landing point"
        self.upstream_type = upstream_type
        self.upstream_id = upstream_id


class UmbilicalCable(Cable):
    def __init__(
        self,
        index: int,
        length: float,
        db_key: int,
        marker: int,
        seabed_connection_point: tuple[float, float, float],
        device: str,
        x_coordinates: list[float],
        z_coordinates: list[float],
    ):
        super(UmbilicalCable, self).__init__(index, length, db_key, marker)

        self.type_ = "umbilical"
        self.seabed_termination_x = seabed_connection_point[0]
        self.seabed_termination_y = seabed_connection_point[1]
        self.seabed_termination_z = seabed_connection_point[2]
        self.device = device
        self.x_coordinates = x_coordinates
        self.z_coordinates = z_coordinates


def get_burial_depths(
    route: Sequence[int],
    grid: pd.Series,
    target_depth: Optional[float] = None,
):
    """Get the target burial depths.

    Args:
        route (list) [-]: Cable route defined by grid point id.
        grid (pd) [-]: Pandas series containing only the Target burial depth
            column of the grid_pd data.

    Attributes:
        burial_depth (list) [m]: List of burial depths.

    Returns
        burial_depth

    """

    if target_depth is not None:
        burial_depth = [target_depth] * len(route)

    else:
        indexed_grid = grid.set_index("id")
        depth_cols = ["Target burial depth"] * len(route)

        burial_depth = list(indexed_grid.lookup(route, depth_cols))

    return burial_depth


def get_split_pipes(burial_depth: list[float]):
    """Set split pipes based on burial depth.

    Args:
        burial_depth (list) [m]: List of burial depths.

    Attributes:
        split_pipe (list) [-]: List of bools indicating need or not of split
            pipe.

    Returns:
        split_pipe

    """

    split_pipe = [False if x > 0 else True for x in burial_depth]

    return split_pipe
