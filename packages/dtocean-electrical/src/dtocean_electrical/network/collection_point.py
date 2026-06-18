# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Adam Collin
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
Created on Thu Apr 07 13:41:07 2016

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
"""

from typing import Optional

import pandas as pd


class CollectionPoint:
    """Class to define all properties of the offshore collection point."""

    def __init__(
        self,
        index: int,
        loc: tuple[float, ...],
        db_key: int,
        data: pd.DataFrame,
    ):
        # set attributes of a collection point have
        self.electrical_type_ = None
        self.operating_environment = data.operating_environment.values[0]
        self.n_inputs: int = data.input.values[0]
        self.n_output: int = data.output.values[0]
        self.input_type: str = data.input_connector.values[0]
        self.output_type: str = data.output_connector.values[0]
        self.foundation_type: str = data.foundation.values[0]
        self.mass: float = data.dry_mass.values[0]
        self.centre_of_gravity: tuple[float, float, float] = (
            data.gravity_centre.values[0]
        )
        self.wet_frontal_area: float = data.wet_frontal_area.values[0]
        self.wet_beam_area: float = data.wet_beam_area.values[0]
        self.dry_frontal_area: float = data.dry_frontal_area.values[0]
        self.dry_beam_area: float = data.dry_beam_area.values[0]
        self.length: float = data.depth.values[0]
        self.width: float = data.width.values[0]
        self.height: float = data.height.values[0]
        self.volume: float = self.length * self.width * self.height
        self.profile: str = "rectangular"
        self.surface_roughness: float = 1e-6
        self.orientation_angle: float = data.orientation_angle.values[0]
        self.foundation_locations: tuple[float, float, float] = (
            data.foundation_loc.values[0]
        )
        #        self.subsea = self._check_operating_environment(
        #            data.operating_environment.values[0])

        # se attributes we place on the collection point
        self.id_ = index
        self.db_key = db_key
        self.location = loc
        self.utm_x = loc[0]
        self.utm_y = loc[1]
        self.input_connectors: str = data.input_connector.item()
        self.output_connectors: str = data.output_connector.item()
        self.marker: Optional[int] = None  # the network marker is added later

        self.type_: str
        self.subsea: bool
        self.configuration: str | list[str]

    def __str__(self):
        """Override print command to display some info"""

        return (
            "This is the "
            + self.__class__.__name__
            + " class. This "
            + self.__class__.__name__
            + " has "
            + str(self.n_inputs)
            + " inputs."
        )


class PassiveHub(CollectionPoint):
    """Special instance of CollectionPoint class."""

    def __init__(
        self,
        index: int,
        loc: tuple[float, ...],
        db_key: int,
        data: pd.DataFrame,
    ):
        super(PassiveHub, self).__init__(index, loc, db_key, data)
        self.type_ = "passive hub"
        self.subsea = True
        self.configuration = "busbar"


class Substation(CollectionPoint):
    """Special instance of CollectionPoint class."""

    def __init__(
        self,
        index: int,
        loc: tuple[float, ...],
        db_key: int,
        data: pd.DataFrame,
    ):
        super(Substation, self).__init__(index, loc, db_key, data)
        self.type_ = "substation"
        self.subsea = True
        self.configuration = ["busbar", "transformer", "disconnector"]
