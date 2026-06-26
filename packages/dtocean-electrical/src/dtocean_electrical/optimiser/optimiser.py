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
This module defines the DTOcean electrical subsystems network design process.

.. module:: optimiser
   :synopsis: Control of network design process.

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
.. moduleauthor:: Mathew Topper <damm_horse@yahoo.co.uk>
"""

import bisect
import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.cluster.vq import kmeans2, vq
from shapely.geometry import LinearRing, LineString, Point

from ..inputs import ElectricalComponentDatabase
from ..network.network import Network
from . import array_layout as connect
from .power_flow import ComponentLoading, PyPower
from .umbilical import UmbilicalDesign

if TYPE_CHECKING:
    from ..main import Electrical

module_logger = logging.getLogger(__name__)


class Optimiser(ABC):
    """Optimiser class to find the solution. This takes configuration options
    as constraints and searches within a predefined space for a best solution.

    Args:
        meta_data (object) [-]: Instance of the Electrical class.

    Attributes:
        attributes (type): Description.

    Returns:
        returns (type): Description.

    """

    def __init__(self, meta_data: "Electrical"):
        self.meta_data = meta_data
        self.lcoe: list[float] = []
        self.floating = meta_data.array_data.machine_data.floating
        self.limits: Optional[dict[float, dict[str, float]]] = None
        self.allowable_voltages: Optional[list[float]] = None
        self.transmission_voltages: Optional[list[float]] = None
        self.array_voltages: Optional[list[float]] = None
        self.levels: Optional[int] = None
        self.voltage_combinations: Optional[list[tuple[float, float]]] = None
        self.networks: list[Network] = []

    @property
    @abstractmethod
    def network_type(self) -> str: ...

    @abstractmethod
    def set_network_design_limits(
        self,
        device_loc: np.ndarray,
        device_power: float,
        device_voltage: float,
        *args,
        **kwargs,
    ): ...

    @abstractmethod
    def run_it(self, tool: str) -> Network: ...

    def _transmission_limits(self) -> dict[float, dict[str, float]]:
        """Prescribed transmission of widely used voltage levels, acting as a
        look-up table. These can be adjusted to modify the solution. For each
        voltage level a min and max power and distance are defined. These pairs
        act to define a feasible tranmission range.

        Attributes:
            transmission_limits_069 (dict) [-]: Transmission limits of 0.69 kV.
            transmission_limits_11 (dict) [-]: Transmission limits of 11 kV.
            transmission_limits_22 (dict) [-]: Transmission limits of 22 kV.
            transmission_limits_33 (dict) [-]: Transmission limits of 33 kV.
            transmission_limits_35 (dict) [-]: Transmission limits of 35 kV.
            transmission_limits_66 (dict) [-]: Transmission limits of 66 kV.
            transmission_limits_110 (dict) [-]: Transmission limits of 110 kV.
            min_power (float) [MW]
            max_power (float) [MW]
            min_distance (float) [km]
            max_distance (float) [km]

        Returns:
            dictionary

        Note:
            Values for 22 and 35 kV have been included, as they may be
            applicable in certain European areas, but are not activated.

        """

        transmission_limits_00069 = {
            "min_power": 0.4,
            "max_power": 1.0,
            "min_distance": 0.6,
            "max_distance": 1.3,
        }

        transmission_limits_0066 = {
            "min_power": 3.5,
            "max_power": 9.0,
            "min_distance": 6.0,
            "max_distance": 12.0,
        }

        transmission_limits_011 = {
            "min_power": 5.0,
            "max_power": 14.0,
            "min_distance": 9.0,
            "max_distance": 19.0,
        }

        # transmission_limits_22 = {'min_power': 10., 'max_power': 28.,
        #                          'min_distance': 18., 'max_distance': 38.}

        transmission_limits_033 = {
            "min_power": 15.0,
            "max_power": 43.0,
            "min_distance": 27.0,
            "max_distance": 58.0,
        }

        # transmission_limits_35 = {'min_power': 18., 'max_power': 50.,
        #                          'min_distance': 31., 'max_distance': 68.}

        transmission_limits_066 = {
            "min_power": 34.0,
            "max_power": 94.0,
            "min_distance": 59.0,
            "max_distance": 128.0,
        }

        transmission_limits_110 = {
            "min_power": 57.0,
            "max_power": 157.0,
            "min_distance": 99.0,
            "max_distance": 213.0,
        }

        keys = [690.0, 6600, 11000.0, 33000.0, 66000.0, 110000.0]

        values = [
            transmission_limits_00069,
            transmission_limits_0066,
            transmission_limits_011,
            transmission_limits_033,
            transmission_limits_066,
            transmission_limits_110,
        ]

        dictionary = dict(zip(keys, values))

        return dictionary

    def get_user_voltage(self, system: str) -> float | None:
        if system == "transmission":
            voltage = self.meta_data.options.export_voltage
        else:
            voltage = None

        return voltage

    def check_user_voltage_level(self, system: str) -> float | None:
        """Run a quick test on user defined input levels."""

        if self.allowable_voltages is None:
            return None

        user_voltage = self.get_user_voltage(system)

        if user_voltage is None:
            return None

        if user_voltage not in self.allowable_voltages:
            i = bisect.bisect_right(self.allowable_voltages, user_voltage)

            if i == 0:
                user_voltage = self.allowable_voltages[0]

            else:
                closest_down = self.allowable_voltages[i - 1 : i]
                max_voltage = closest_down[0]

                if self.get_next_voltage([max_voltage]) is None:
                    user_voltage = max_voltage

                else:
                    user_voltage = self.get_next_voltage([max_voltage])

            msg = (
                "User defined {} voltage not in valid values. Setting to "
                "next greatest value: {} V"
            ).format(system, user_voltage)
            module_logger.warning(msg)

        return user_voltage

    def compare_user_voltage_levels(
        self,
        system: str,
        suggested_voltage: float,
        override: bool = False,
    ) -> tuple[float, ...]:
        """Compare user voltage against the suggested. Add option to return
        both if different or just output warning.

        """

        user_voltage = self.get_user_voltage(system)
        if user_voltage is None:
            return ()

        if user_voltage != suggested_voltage:
            if override:
                system_voltages = (user_voltage, suggested_voltage)
            else:
                # add info message to the user that other voltage levels are
                # possible.
                # Also look at suitability of user voltage as possible warning
                # message
                system_voltages = (user_voltage,)

        else:
            system_voltages = (user_voltage,)

        return system_voltages

    def get_next_voltage(
        self,
        suggested_voltages: Sequence[float],
    ) -> float | None:
        """Get next greater voltage for solution."""

        if self.allowable_voltages is None:
            return None

        max_voltage = max(suggested_voltages)

        try:
            next_voltage = self.allowable_voltages[
                self.allowable_voltages.index(max_voltage) + 1
            ]
        except IndexError:
            next_voltage = None

        return next_voltage

    def make_voltage_combinations(self):
        """Get voltage combinations for analysis. Two levels are allowed for
        radial; three levels for star.

        Note: all_combo from stack overflow.

        """

        if self.array_voltages is None or self.transmission_voltages is None:
            self.voltage_combinations = None
            return

        list1 = self.array_voltages
        list2 = self.transmission_voltages
        all_combo = [(export, array) for export in list2 for array in list1]

        # Flatten nested list, loops used as all_combo to sanitise in place
        voltage_combinations = []

        for item in all_combo:
            if item[0] >= item[1]:
                voltage_combinations.append(item)

        self.voltage_combinations = voltage_combinations

    def set_design_limits(self):
        """Compare distance, power and voltage to obtain approximation of
        technically feasible transmission and array voltages. This incorporates
        a look-up table of limits.

        Slightly different approaches are considered for radial and star
        layouts. One voltage transformation for radial is allowed, while two
        are voltage transformations are achievable in star layouts.

        """

        self.limits = self._transmission_limits()
        self.allowable_voltages = list(self.limits.keys())
        self.allowable_voltages.sort()

        # get approximate distances from lease area to shore
        edge_to_shore = self._approximate_lease_edge_distance_to_shore()

        device_power = self.meta_data.array_data.machine_data.power
        array_power = self.meta_data.array_data.total_power
        device_voltage = self.meta_data.array_data.machine_data.voltage

        module_logger.debug("Calculating transmission voltages...")

        if self.meta_data.options.export_voltage:
            system = "transmission"

            self.meta_data.options.export_voltage = (
                self.check_user_voltage_level(system)
            )

            voltage = self.allowable_voltages[0]
            suggested_voltage = self.cross_check_limits(
                voltage,
                edge_to_shore,
                device_power,
            )
            assert suggested_voltage is not None

            transmission_voltages = self.compare_user_voltage_levels(
                system,
                suggested_voltage,
            )
            if transmission_voltages is not None:
                self.transmission_voltages = list(transmission_voltages)

        else:
            transmission_voltage = self.cross_check_limits(
                device_voltage,
                edge_to_shore,
                array_power,
            )
            assert transmission_voltage is not None
            transmission_voltages = [transmission_voltage]
            next_voltage = self.get_next_voltage(transmission_voltages)

            if next_voltage is not None:
                transmission_voltages.append(next_voltage)
                self.transmission_voltages = transmission_voltages
            else:
                self.transmission_voltages = transmission_voltages

        module_logger.debug("Calculating array voltages...")

        device_loc = self.convert_layout_to_numpy()
        self.set_network_design_limits(
            device_loc,
            device_power,
            device_voltage,
        )

    def cross_check_limits(
        self,
        voltage: float,
        distance: float,
        power: float,
    ) -> float | None:
        """Compare a given voltage, distance and power against prescribed
        transmission limits.

        Args:
            voltage (float) [V]:
            distance (float) [m]:
            power (float) [W]:

        Attributes:
            limits (dict) [-]:
            applicable_limits (dict) [-]:
            safe_voltage (float) [-]:
            next_voltage (float) [-]:
            all_voltages (list) [-]:

        Returns:
            safe_voltage

        """

        if self.limits is None or self.allowable_voltages is None:
            return None

        applicable_limits = self.limits[voltage]

        if (distance / 1000 < applicable_limits["max_distance"]) and (
            power / 1000000 < applicable_limits["max_power"]
        ):
            safe_voltage = voltage

        else:
            # get next voltage
            all_voltages = deepcopy(self.allowable_voltages)
            all_voltages.sort()

            try:
                next_voltage = all_voltages[all_voltages.index(voltage) + 1]
                safe_voltage = self.cross_check_limits(
                    next_voltage, distance, power
                )

            except IndexError:
                # End of list
                # add log message here - beyond scope and set voltage to max
                safe_voltage = all_voltages[-1]

        return safe_voltage

    def device_spacing_summary(
        self,
        distance_array: np.ndarray,
    ) -> tuple[float, float, float, float]:
        min_val = np.min(distance_array[np.nonzero(distance_array)])
        max_val = np.max(distance_array[np.nonzero(distance_array)])
        ave_val = np.average(distance_array[np.nonzero(distance_array)])
        chain = np.max(np.sum(distance_array, axis=1))

        return min_val, max_val, float(ave_val), chain

    def _approximate_lease_edge_distance_to_shore(self) -> float:
        """Find the approximate distance to shore from the centre of the lease
        area and from the edge of the lease area.
        """

        area = self.meta_data.grid.lease_boundary
        point = Point(
            self.meta_data.array_data.landing_point[0],
            self.meta_data.array_data.landing_point[1],
        )

        area_exterior = LineString(area.exterior.coords)
        d = area_exterior.project(point)
        p = area_exterior.interpolate(d)
        closest_point_coords = list(p.coords)[0]
        distance = point.distance(Point(closest_point_coords))

        return distance

    def _approximate_lease_centre_distance_to_shore(self) -> float:
        """Find the approximate distance to shore from the centre of the lease
        area.
        """

        area = self.meta_data.grid.lease_boundary
        point = Point(
            self.meta_data.array_data.landing_point[0],
            self.meta_data.array_data.landing_point[1],
        )

        area_centre_x = area.centroid.x
        area_centre_y = area.centroid.y
        area_centre = Point([area_centre_x, area_centre_y])
        distance = point.distance(area_centre)

        return distance

    def set_substation_location(
        self,
        device_loc: np.ndarray,
        n_cp: int,
        edge_buffer: Optional[float] = None,
    ) -> tuple[tuple[float, ...], int]:
        module_logger.debug("Calculating substation location...")

        # Need to check shape of array - stacked or not

        lease = self.meta_data.grid.lease_boundary

        if edge_buffer is not None:
            lease = lease.buffer(-edge_buffer)

        lease_area_ring = LinearRing(list(lease.exterior.coords))

        min_x = min(device_loc[:, 0])
        max_x = max(device_loc[:, 0])
        diff_x = max_x - min_x

        min_y = min(device_loc[:, 1])
        max_y = max(device_loc[:, 1])
        diff_y = max_y - min_y

        initial_estimate, _ = kmeans2(device_loc[:, :2], n_cp, minit="points")
        idx, _ = vq(device_loc[:, :2], initial_estimate)

        grid = np.array(self.meta_data.grid.grid_pd[["x", "y"]])

        initial_estimate_on_grid = self.snap_to_grid(grid, initial_estimate[0])

        export_line = LineString(
            [initial_estimate_on_grid, self.meta_data.array_data.landing_point]
        )

        device_points = []

        for item in device_loc:
            device_points.append(Point(item[0], item[1]))

        threshold = 100  # this can be updated based on device spacing
        shift = 100  # this can be updated based on device spacing
        shift_increment = 0
        shift_flag = False

        # check spread of devices
        if len(device_loc) < 2:
            interim_estimate = (initial_estimate_on_grid[0] + shift, min_y)

            shift_flag = True
            shift_increment += shift

        elif diff_x / diff_y >= 1:
            # these devices are considered stacked in the y dimension
            mid_point_x = min_x + diff_x / 2

            err = (
                self.meta_data.array_data.landing_point[1]
                - initial_estimate_on_grid[1]
            )

            if err >= 0:
                interim_estimate = (
                    mid_point_x,
                    initial_estimate_on_grid[1] + shift,
                )

            else:
                interim_estimate = (
                    mid_point_x,
                    initial_estimate_on_grid[1] - shift,
                )

            shift_flag = True
            shift_increment += shift

        elif diff_y / diff_x >= 1:
            # these devices are considered stacked in the x dimension
            mid_point_y = min_y + diff_y / 2

            err = (
                self.meta_data.array_data.landing_point[0]
                - initial_estimate_on_grid[0]
            )

            if err >= 0:
                interim_estimate = (
                    initial_estimate_on_grid[0] + shift,
                    mid_point_y,
                )

            else:
                interim_estimate = (
                    initial_estimate_on_grid[0] - shift,
                    mid_point_y,
                )

            shift_flag = True
            shift_increment += shift

        interim_estimate_shapely = Point(interim_estimate[:2])

        if not interim_estimate_shapely.within(lease):
            interim_estimate = connect.set_substation_to_edge(
                export_line,
                lease_area_ring,
                self.meta_data.site_data.bathymetry,
                lease,
            )

            if interim_estimate is None:
                err_msg = "Failed to find suitable substation location"
                raise RuntimeError(err_msg)

            close = False

        else:
            close = connect.closeness_test(
                device_points, interim_estimate, threshold
            )

            if close is True:
                if shift_flag is False:
                    # shift to the edge
                    device_locs = [tuple(x) for x in device_loc]
                    interim_estimate = connect.offset_cp(
                        device_locs,
                        export_line,
                        interim_estimate,
                        "edge",
                        shift,
                    )

                    close = connect.closeness_test(
                        device_points, interim_estimate, threshold
                    )

                while close is True:
                    interim_estimate = export_line.interpolate(shift_increment)
                    interim_estimate = (interim_estimate.x, interim_estimate.y)
                    close = connect.closeness_test(
                        device_points, interim_estimate, threshold
                    )
                    shift_increment += shift

                    interim_estimate_shapely = Point(
                        interim_estimate[0], interim_estimate[1]
                    )

                    if not interim_estimate_shapely.within(lease):
                        interim_estimate = connect.set_substation_to_edge(
                            export_line,
                            lease_area_ring,
                            self.meta_data.site_data.bathymetry,
                            lease,
                        )

                        close = False

        cp_loc = self.snap_to_grid(grid, interim_estimate)

        return cp_loc, idx

    def convert_layout_to_numpy(self) -> np.ndarray:
        """Convert the layout into a list for further use. This is sorted by
        device id.

        Args:
            devices (list) [-]: Iterator for accessing devices keys in order.
            device_loc (list) [m]: Device x and y coordinates, ordered by key.

        Returns:
            device_loc (list)

        """

        devices = [
            "Device" + str(oec + 1).zfill(3)
            for oec in range(self.meta_data.array_data.n_devices)
        ]

        device_loc = np.asarray(
            [self.meta_data.array_data.layout[oec] for oec in devices]
        )

        return device_loc

    def db_compatibility(
        self,
        db: ElectricalComponentDatabase,
        oec_voltage: float,
        export_voltage: float,
        array_voltage: float,
    ) -> dict[str, Any]:
        """Select a single valid component set.

        Args:
            db () [-]:
            oec_voltage () [-]:
            array_power () [-]:

        Attributes:
            array_cable (int) [-]: Database id of compatible static array
                cable.
            cp (int) [-]: Database id of compatible collection point.
            connector (int) [-]: Database id of compatible connector.
            db_keys (dict) [-]: Key value pairs of compatible components.

        Returns:
            db_keys

        Note:
            Method to be updated to check for extent of data in the database
            as part of constraining solutions. This can be tidied up by passing
            db data extraction to a different function.

        """

        array = array_voltage
        export = export_voltage
        oec = oec_voltage  # not scaled

        array_cable = self._get_component_id(
            db.array_cable,
            "v_rate",
            array,
            "array cable",
            "array voltage",
        )

        export_cable = self._get_component_id(
            db.export_cable,
            "v_rate",
            export,
            "export cable",
            "export cable voltage",
        )

        all_cp = self._get_component_id(
            db.collection_points,
            "v1",
            array,
            "collection point",
            "array voltage",
        )

        # Select cp with correct v1 and v2
        if len(all_cp) > 1:
            for item in all_cp:
                cp_v = db.collection_points[
                    db.collection_points.id == item
                ].v2.item()

                if float(cp_v) == export:
                    cp = item
                    break

        else:
            cp = all_cp[0]

        wet_connector = self._get_component_id(
            db.wet_mate_connectors,
            "v_rate",
            array,
            "wet mate connector",
            "array voltage",
        )

        dry_connector = self._get_component_id(
            db.dry_mate_connectors,
            "v_rate",
            array,
            "dry mate connector",
            "array voltage",
        )

        # TODO: What if there are multiple matching records (or none?)
        wet_connector = wet_connector[0]
        dry_connector = dry_connector[0]

        db_keys = {
            "array": array_cable,
            "cp": cp,
            "wet_connector": wet_connector,
            "dry_connector": dry_connector,
            "export": export_cable,
        }

        if self.meta_data.array_data.machine_data.technology == "floating":
            # check for user umbilical else select one
            if self.meta_data.options.user_umbilical:
                umbilical_cable = self.meta_data.options.user_umbilical

            else:
                if db.dynamic_cable is None:
                    raise ValueError("Dynamic cable data not set")

                umbilical_cable = self._get_component_id(
                    db.dynamic_cable,
                    "v_rate",
                    oec,
                    "umbilical cable",
                    "OEC voltage",
                )

                umbilical_cable = umbilical_cable[0]  # sanitize this

            db_keys["umbilical"] = umbilical_cable

        if self.network_type == "Star":
            device_cable_all = self._get_component_id(
                db.array_cable, "v_rate", oec, "static cable", "OEC voltage"
            )

            # get max current rating for device
            current_rating = self.meta_data.array_data.machine_data.max_current

            db_temp = db.array_cable[db.array_cable.id.isin(device_cable_all)]

            test_cable = False

            for _, val in db_temp.sort_values("a_air").iterrows():
                if val.a_air > current_rating:
                    device_cable = val.id
                    test_cable = True

                    break

            if test_cable:
                db_keys["device"] = device_cable

            else:
                errStr = (
                    "No cables in db able to connect device. Check "
                    + "voltage and current limits. Based on input data "
                    + "voltage {} V and current {} A are required".format(
                        self.meta_data.array_data.machine_data.voltage,
                        current_rating,
                    )
                )

                raise ValueError(errStr)

        return db_keys

    def _get_component_id(
        self,
        comp_table: pd.DataFrame,
        comp_column: str,
        match_value: Any,
        comp_type_str: str,
        match_type_str: str,
        allow_greater: bool = True,
    ) -> list[int]:
        found_component = []

        # Try and pick up an exact component
        match_components: pd.DataFrame = comp_table[
            comp_table[comp_column] == match_value
        ]

        # If we have the exact component then return
        if not match_components.empty:
            return match_components.id.to_list()

        # If desired, try for one with greater than the matching value
        if allow_greater:
            match_components = comp_table[comp_table[comp_column] > match_value]

            if len(match_components) >= 1:
                match_components = match_components.sort_values([comp_column])
                found_component = match_components.id.to_list()

        # If not component is available, raise
        if match_components.empty:
            errStr = "No suitable {} found for {}: {}".format(
                comp_type_str, match_type_str, match_value
            )
            raise ValueError(errStr)

        return found_component

    def symmetrize(self, matrix: np.ndarray):
        """Make the matrix symmetrical. Code from:
        http://stackoverflow.com/questions/2572916/numpy-smart-symmetric-matrix

        Args:
            matrix (numpy array): The unsymmetrical network connection matrix.

        Returns:
            numpy array

        """

        return matrix + matrix.T - np.diag(matrix.diagonal())

    def iterate_cable_solutions(
        self,
        n_cp: int,
        cp_loc: list[tuple[float, ...]] | tuple[float, ...],
        network_connections: dict[str, np.ndarray],
        network_count: int,
        components: dict[str, Any],
        distances: np.ndarray,
        export_length: float,
        export_route: list[int],
        export_voltage: float,
        array_voltage: float,
        paths: np.ndarray,
        burial_targets: pd.DataFrame,
        umbilical_design: Optional[dict[str, dict[str, Any]]] = None,
        umbilical_impedance: Optional[Sequence[tuple[float, ...]]] = None,
        cp_cp_distances: Optional[np.ndarray] = None,
        cp_cp_paths: Optional[np.ndarray] = None,
    ):
        # if multiple cables, compare solutions - treat array and export as
        # discrete systems
        component_combinations = self.make_cable_solutions(components)

        logMsg = ("{} component combinations " "found").format(
            len(component_combinations)
        )
        module_logger.debug(logMsg)

        for i, cable_set in enumerate(component_combinations):
            logMsg = "Evaluating component combination {}".format(i)
            module_logger.debug(logMsg)
            module_logger.debug("Creating pypower object...")

            py_power_network = self.create_pypower_object(
                n_cp,
                network_connections,
                cable_set,
                distances,
                export_length,
                export_voltage,
                array_voltage,
                umbilical_impedance,
                cp_cp_distances,
            )

            module_logger.debug("Checking cable loadings...")

            export_constraints, array_constraints = self.check_cable_loading(
                py_power_network,
                cable_set,
                export_voltage,
                array_voltage,
            )

            module_logger.debug("Building network object...")

            network = self.create_network_object(
                network_count,
                py_power_network,
                cp_loc,
                cable_set,
                distances,
                paths,
                export_route,
                export_length,
                burial_targets,
                export_constraints,
                array_constraints,
                umbilical_design,
                cp_cp_paths,
                cp_cp_distances,
            )

            # Record the export cable voltage
            network.export_voltage = export_voltage

            assert network.lcoe is not None
            self.lcoe.append(network.lcoe)
            self.networks.append(network)

        network_count += 1

        return network_count

    def make_cable_solutions(
        self,
        components: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Create unique copies of the components dictionary for each cable
        solution.

        Args:
            componens (dict) [-]: Selected components.

        """

        local_use = deepcopy(components)
        unique_component_dicts: list[dict[str, Any]] = []

        for export in local_use["export"]:
            for array in local_use["array"]:
                temp_dictionary = deepcopy(local_use)
                temp_dictionary["export"] = export
                temp_dictionary["array"] = array

                unique_component_dicts.append(temp_dictionary)

        return unique_component_dicts

    def check_cable_loading(
        self,
        py_power_network: PyPower,
        components: dict[str, Any],
        export_voltage: float,
        array_voltage: float,
    ) -> tuple[ComponentLoading, ComponentLoading]:
        if py_power_network.all_results is None:
            raise RuntimeError("py_power_network object must have been solved")

        # export
        db_key = components["export"]
        export_cable_rating = self.cable_rating(db_key, "export")
        export_constraint = ComponentLoading("export", export_voltage)
        export_constraint.check_component_loading(
            py_power_network.all_results,
            export_cable_rating,
            py_power_network.export_branches,
        )

        # array
        db_key = components["array"]
        array_cable_rating = self.cable_rating(db_key, "array")
        array_constraint = ComponentLoading("array", array_voltage)
        array_constraint.check_component_loading(
            py_power_network.all_results,
            array_cable_rating,
            py_power_network.array_branches,
        )

        return (export_constraint, array_constraint)

    def create_pypower_object(
        self,
        n_cp: int,
        network_connections: dict[str, np.ndarray],
        components: dict[str, Any],
        cp_device_distances: np.ndarray,
        export_length: float,
        export_voltage: float,
        array_voltage: float,
        umbilical_impedance: Optional[Sequence[tuple[float, ...]]] = None,
        cp_cp_distances: Optional[np.ndarray] = None,
    ) -> PyPower:
        """Unified code to create pypower object for both network types."""

        network_type = self.meta_data.options.network_configuration[0]

        offshore_substation = True

        # initialise PyPower network
        pypower_network = PyPower(
            n_cp,
            self.meta_data.array_data.n_devices,
            network_connections,
            export_voltage,
            array_voltage,
            self.meta_data.array_data.machine_data.voltage,
            offshore_substation,
            network_type,
            self.floating,
        )

        ### Build it
        # array / device cable
        db_key = components["array"]
        impedance = self.cable_impedance(db_key, "array")

        if cp_cp_distances is not None:
            array_impedance_matrix = pypower_network.calculate_impedances(
                cp_cp_distances,
                impedance,
                "array",
            )

        else:
            array_impedance_matrix = None

        device_impedance_matrix = pypower_network.calculate_impedances(
            cp_device_distances,
            impedance,
            "device",
        )

        # export cable
        db_key = components["export"]
        impedance = self.cable_impedance(db_key, "export")

        z_export = pypower_network.calculate_export_impedance(
            export_length,
            impedance,
        )

        # Transformer impedance
        # export to array
        if export_voltage != array_voltage:
            T_export_array = pypower_network.transformer_impedance(
                export_voltage,
                array_voltage,
                self.meta_data.database.transformers,
            )

        else:
            T_export_array = 0.0

        oec_voltage = self.meta_data.array_data.machine_data.voltage

        if array_voltage != oec_voltage:
            T_array_device = pypower_network.transformer_impedance(
                array_voltage,
                oec_voltage,
                self.meta_data.database.transformers,
            )

        else:
            T_array_device = 0.0

        if self.floating:
            if umbilical_impedance is None:
                msg = (
                    "umbilical_impedance must not be None if self.floating is "
                    "True"
                )
                raise ValueError(msg)

            z_umbilical = pypower_network.calculate_umbilical_impedance(
                umbilical_impedance
            )

        else:
            z_umbilical = None

        pypower_network.build_network(
            z_export,
            device_impedance_matrix,
            T_export_array,
            array_impedance_matrix,
            T_array_device,
            z_umbilical,
        )

        power_factor = self.meta_data.array_data.machine_data.power_factor
        assert isinstance(power_factor, list)

        pypower_network.run_pf(
            power_factor,
            self.meta_data.array_data.machine_data.power,
        )

        return pypower_network

    def create_network_object(
        self,
        network_count: int,
        py_power_network: PyPower,
        cp_loc: list[tuple[float, ...]] | tuple[float, ...],
        components: dict[str, Any],
        cp_device_distances: np.ndarray,
        cp_device_paths: np.ndarray,
        export_route: list[int],
        export_length: float,
        burial_targets: pd.DataFrame,
        export_constraints: ComponentLoading,
        array_constraints: ComponentLoading,
        umbilical_design: Optional[dict[str, dict[str, Any]]] = None,
        cp_cp_paths: Optional[np.ndarray] = None,
        cp_cp_distances: Optional[np.ndarray] = None,
    ):
        if py_power_network.onshore_active_power is None:
            raise RuntimeError("py_power_network object must have been solved")

        network_type = self.meta_data.options.network_configuration[0]
        cp_db = self.meta_data.database.collection_points

        if network_type == "Star":
            if not isinstance(cp_loc, list):
                raise ValueError("cp_loc must be list for star network")
            cp_locs = cp_loc
        else:
            if not isinstance(cp_loc, tuple):
                raise ValueError("cp_loc must be tuple for radial network")
            cp_locs = [cp_loc]

        # create network object to carry this information
        network = Network(
            network_count,
            self.floating,
            py_power_network.onshore_active_power,
            self.meta_data.array_data,
            self.meta_data.database,
            export_constraints,
            array_constraints,
            py_power_network,
            cp_locs,
            components["cp"],
            cp_db,
            cp_device_distances,
            cp_cp_distances,
            cp_device_paths,
            cp_cp_paths,
            export_route,
            export_length,
            components,
            burial_targets,
            self.meta_data.options.target_burial_depth_array,
            self.meta_data.options.target_burial_depth_export,
            self.meta_data.grid,
            umbilical_design,
        )

        return network

    def cable_impedance(
        self,
        db_key: int,
        cable_type: str,
    ) -> tuple[float, float, float]:
        """Extract cable impedance from database."""

        db = self._get_cable_db(cable_type)
        cable = db.loc[db["id"] == db_key]

        # r, x, c
        impedance = (
            float(cable["r_ac"].item()),
            float(cable["xl"].item()),
            float(cable["c"].item()),
        )

        return impedance

    def cable_rating(
        self,
        db_key: int,
        cable_type: str,
    ) -> int:
        """Extract cable rating from database."""

        db = self._get_cable_db(cable_type)
        cable = db.loc[db["id"] == db_key]

        rating = cable["a_air"].item()

        return int(rating)

    def _get_cable_db(self, cable_type: str) -> pd.DataFrame:
        if cable_type == "array":
            db = self.meta_data.database.array_cable
        elif cable_type == "export":
            db = self.meta_data.database.export_cable
        else:
            raise KeyError("Unsupported cable type: {}".format(cable_type))

        return db

    def check_lcoe(self) -> int:
        """Get ranked solutions and iterate over to find first valid. This is
        taken as best.

        """

        order = self.sort_lcoe()
        result = None

        for network in order:
            network_lcoe = self.networks[network].lcoe
            if (
                self.networks[network].array_constraints.flag is False
                and self.networks[network].export_constraints.flag is False
                and network_lcoe is not None
                and np.isfinite(network_lcoe)
            ):
                result = network
                break

            else:
                msgStr = (
                    "Invalid solution detected: Export constraint flag: "
                    "{}; Array constraint flag: {}; LCOE {}"
                ).format(
                    self.networks[network].export_constraints.flag,
                    self.networks[network].array_constraints.flag,
                    self.networks[network].lcoe,
                )
                module_logger.warning(msgStr)

        if result is None:
            errStr = (
                "Could not find valid solution for given database and "
                "options."
            )
            raise ValueError(errStr)

        return result

    def sort_lcoe(self) -> list[int]:
        """Sort nework solutions by target.

        Return:
            list: The sorted index

        """

        return sorted(range(len(self.lcoe)), key=self.lcoe.__getitem__)

    def snap_to_grid(self, grid, point) -> tuple[float, ...]:
        """Snap a point to the grid.

        Args:
            grid (np.array) [m]: Array of x and y coordinates.
            point (tuple) [m]: Coordinates of point under consideration, x and
                y coordinates.

        Attributes:
            new_coords (list) [m]: Coordinates of nearest point, x, y and z.

        Returns:
            tuple

        """

        new_coords = grid[
            spatial.KDTree(grid).query(np.array(point))[1]
        ].tolist()

        # and add z coord
        z = self.meta_data.grid.grid_pd[
            (self.meta_data.grid.grid_pd.x == new_coords[0])
            & (self.meta_data.grid.grid_pd.y == new_coords[1])
        ]["layer 1 start"].values[0]

        new_coords.append(z)
        new_coords = [float(i) for i in new_coords]

        return tuple(new_coords)

    def update_static_cables(
        self,
        umbilical_design: dict[str, dict[str, Any]],
        paths: np.ndarray,
        sol: list[list[int]],
        distances: np.ndarray,
    ):
        """Update the static cable distance and path in the presence of an
        umbilical cable.

        """

        network_type = self.meta_data.options.network_configuration[0]

        grid = self.meta_data.grid

        # chop array cable at point of connection
        for cable in umbilical_design.values():
            # get device id for indexing path array
            device = int(cable["device"].split("Device")[1])
            d_id = [item for item in sol if device in item][0]
            downstream = d_id.index(device) - 1
            umbilical_path = paths[d_id[downstream]][device]

            # get termination point coords for grid id
            grid_point = grid.grid_pd[
                (grid.grid_pd["x"].isin([cable["termination"][0]]))
                & (grid.grid_pd["y"].isin([cable["termination"][1]]))
            ]["id"].values[0]

            # chop the array cable at this point
            new_array_path = umbilical_path[
                : umbilical_path.index(grid_point) + 1
            ]

            paths[d_id[downstream]][device] = tuple(new_array_path)

            # update distance
            new_distance = 0

            for i, point in enumerate(new_array_path[:-1]):
                point2 = new_array_path[i + 1]

                new_distance += grid.graph.get_edge_data(point, point2)[
                    "weight"
                ]

            if network_type == "Star":
                distances[device] = new_distance

            elif network_type == "Radial":
                distances[d_id[downstream]][device] = new_distance
                distances[device][d_id[downstream]] = new_distance

                # add to path of next device if not last in chain
                if len(d_id) > 2:
                    local_id = d_id.index(device)

                    if len(d_id) > (local_id + 1):
                        next_path = list(paths[device][d_id[local_id + 1]])
                        start_idx = umbilical_path.index(grid_point) + 1
                        path_linked = False

                        for item in reversed(umbilical_path[start_idx:]):
                            if item in next_path:
                                break

                            if path_linked:
                                next_path.insert(0, item)

                            else:
                                _, link_path = connect.dijkstra(
                                    grid.graph, item, next_path[0]
                                )

                                next_path = link_path[:-1] + next_path
                                path_linked = True

                        paths[device][d_id[local_id + 1]] = tuple(next_path)

    def make_outputs(self, min_lcoe: int) -> Network:
        return self.networks[min_lcoe]

    def select_seabed(self, tool: Optional[str] = None) -> nx.Graph | None:
        if tool == "Jetting":
            graph = self.meta_data.grid.jetting_graph

        elif tool == "Ploughing":
            graph = self.meta_data.grid.ploughing_graph

        elif tool == "Cutting":
            graph = self.meta_data.grid.cutting_graph

        elif tool == "Dredging":
            graph = self.meta_data.grid.dredging_graph

        elif tool is None:
            graph = self.meta_data.grid.graph

        else:
            errStr = "Unrecognised installation tool: {}".format(tool)
            raise ValueError(errStr)

        return graph


class RadialNetwork(Optimiser):
    """Radial network topology."""

    @property
    def network_type(self) -> str:
        return "Radial"

    def set_network_design_limits(
        self,
        device_loc: np.ndarray,
        device_power: float,
        device_voltage: float,
        *args,
        **kwargs,
    ):
        self.array_voltages = [device_voltage]
        self.levels = 2
        self.make_voltage_combinations()

    def run_it(self, installation_tool: Optional[str] = None) -> Network:
        """Control logic for designing a radial network."""

        seabed_graph = self.select_seabed(installation_tool)

        if seabed_graph is None or seabed_graph.size() == 0:
            raise nx.NetworkXNoPath

        combo_export, combo_array, combo_devices = self.control_simulations()
        n_combos = len(combo_export)

        module_logger.info("Defined {} network combinations".format(n_combos))

        burial_targets = self.meta_data.grid.grid_pd[
            ["id", "Target burial depth"]
        ]
        network_count = 0
        solutions = []

        n_cp = 1  # fixed for ram compatibility

        device_loc = self.convert_layout_to_numpy()

        module_logger.info("Setting substation location...")
        cp_loc, _ = self.set_substation_location(
            device_loc,
            n_cp,
            self.meta_data.options.edge_buffer,
        )

        module_logger.info("Defining export cable route...")

        # get export cable here
        (export_length, export_route) = connect.get_export(
            cp_loc,
            self.meta_data.array_data.landing_point,
            self.meta_data.grid,
            seabed_graph,
        )

        module_logger.info("Calculating array distances...")
        (distance_matrix, path_matrix) = connect.calculate_distance_dijkstra(
            self.meta_data.array_data.layout_grid,
            cp_loc,
            self.meta_data.grid,
            seabed_graph,
        )

        local_umbilical = UmbilicalDesign(self.meta_data)

        for i, simulation in enumerate(
            zip(combo_export, combo_array, combo_devices)
        ):
            module_logger.info(
                "Evaluating network combination {} of " "{}".format(
                    i + 1, n_combos
                )
            )

            # Copy the distance and path matrices
            sim_distance_matrix = deepcopy(distance_matrix)
            sim_path_matrix = deepcopy(path_matrix)

            export_voltage = simulation[0]
            array_voltage = simulation[1]
            device_per_string = simulation[2]

            module_logger.debug("Selecting components")

            components = self.db_compatibility(
                self.meta_data.database,
                self.meta_data.array_data.machine_data.voltage,
                export_voltage,
                array_voltage,
            )

            sol = self.brute_force_method(
                device_per_string + 1,
                sim_distance_matrix,
                sim_path_matrix,
            )

            module_logger.debug("Storing solution...")
            solutions.append(sol)

            # call umbilical design model
            if self.floating:
                module_logger.debug("Designing umbilical...")

                umbilical_db_key = components["umbilical"]

                # call umbilical design model
                local_umbilical.umbilical_design(
                    sim_path_matrix, umbilical_db_key, sol
                )
                umbilical_design = local_umbilical.designs
                assert umbilical_design is not None

                umbilical_impedance = (
                    local_umbilical.umbilical_impedance_table()
                )

                self.update_static_cables(
                    umbilical_design,
                    sim_path_matrix,
                    sol,
                    sim_distance_matrix,
                )

            else:
                umbilical_design = None
                umbilical_impedance = None

            module_logger.debug("Calculating network cost and efficiency...")

            network_connections = self.convert_to_pypower(sol)
            network_count = self.iterate_cable_solutions(
                n_cp,
                cp_loc,
                network_connections,
                network_count,
                components,
                sim_distance_matrix,
                export_length,
                export_route,
                export_voltage,
                array_voltage,
                sim_path_matrix,
                burial_targets,
                umbilical_design,
                umbilical_impedance,
            )

        module_logger.debug("Creating outputs...")

        min_lcoe = self.check_lcoe()
        network_outputs = self.make_outputs(min_lcoe)

        return network_outputs

    def control_simulations(self) -> tuple[list[float], list[float], list[int]]:
        assert self.voltage_combinations is not None

        v_export: list[float] = []
        v_array: list[float] = []
        n_devices: list[int] = []
        max_devices_per_line = None

        if self.meta_data.options.devices_per_string is not None:
            if (
                self.meta_data.options.devices_per_string
                > self.meta_data.array_data.n_devices
            ):
                msg = (
                    "Max number of devices per string {} is greater than "
                    "array size {}. Capped at array size."
                ).format(
                    self.meta_data.options.devices_per_string,
                    self.meta_data.array_data.n_devices,
                )
                module_logger.warning(msg)

                max_devices_per_line = self.meta_data.array_data.n_devices

            else:
                max_devices_per_line = self.meta_data.options.devices_per_string

        for combo in self.voltage_combinations:
            if max_devices_per_line is not None:
                v_export.append(combo[0])
                v_array.append(combo[1])
                n_devices.append(max_devices_per_line)

                continue

            for device_per_line in range(self.meta_data.array_data.n_devices):
                v_export.append(combo[0])
                v_array.append(combo[1])
                n_devices.append(device_per_line)

        return v_export, v_array, n_devices

    def brute_force_method(
        self,
        max_: int,
        distance_matrix: np.ndarray,
        path_matrix: np.ndarray,
    ) -> list[list[int]]:
        """Brute force optimisation of radial network. Iterate through all
        possible combinations of radial networks.

        Args:
            arguments (type): Description.

        Attributes:
            attributes (type): Description.

        Returns:
            connect_matrix (list): vector of array connections.

        """

        module_logger.debug("Starting brute force method")
        module_logger.debug("Maximum devices per line is {}".format(max_))

        n_devices = self.meta_data.array_data.n_devices
        layout = deepcopy(self.meta_data.array_data.layout)

        # initialise route vector
        route_vector: list[tuple[int, int]] = []
        for n in range(0, n_devices):
            interim = (n + 1, 0)
            route_vector.append(interim)

        # initialise path vector, P
        path_vector = [list(reversed(edge)) for edge in route_vector]

        device_positions = connect.add_device_positions(
            layout,
            self.meta_data.array_data.layout_grid,
        )

        module_logger.debug("Creating savings vector...")
        savings_vector = connect.calculate_saving_vector(
            distance_matrix,
            n_devices,
        )

        module_logger.debug("Building path...")
        connect_matrix = connect.make_optimal_paths(
            savings_vector,
            path_vector,
            route_vector,
            max_,
            path_matrix,
            device_positions,
            self.meta_data.grid,
        )

        return connect_matrix

    def convert_to_pypower(
        self,
        network_structure: list[list[int]],
    ) -> dict[str, np.ndarray]:
        """From route paths to pypower bin string format for networks with a
        single collection point.

        Args:
            arguments (type): Description.

        Attributes:
            attributes (type): Description.

        Returns:
            returns (type): Description.

        """

        n_devices = self.meta_data.array_data.n_devices

        # create shore to device matrix
        shore_to_device = np.array([[0] * n_devices])
        shore_to_cp = np.array([1])
        cp_to_cp = np.array([])
        cp_to_device = np.array([[0] * n_devices])
        device_to_device = np.array([[0] * n_devices] * n_devices)

        # Work is here to connect in bin string style
        for item in network_structure:
            cp_to_device[0][item[1] - 1] = 1
            chain = item[1:]

            for idx in range(len(chain) - 1):
                d_to_d_int = np.array([0] * n_devices)
                d_to_d_int[chain[idx + 1] - 1] = 1
                device_to_device[chain[idx] - 1] = (
                    device_to_device[chain[idx] - 1] + d_to_d_int
                )

        device_to_device = self.symmetrize(device_to_device)

        network_connections = {
            "shore_to_device": shore_to_device,
            "shore_to_cp": shore_to_cp,
            "cp_to_cp": cp_to_cp,
            "cp_to_device": cp_to_device,
            "device_to_device": device_to_device,
        }

        return network_connections


class StarNetwork(Optimiser):
    """Star network topology."""

    @property
    def network_type(self) -> str:
        return "Star"

    def set_network_design_limits(
        self,
        device_loc: np.ndarray,
        device_power: float,
        device_voltage: float,
        *args,
        **kwargs,
    ):
        groups = self.star_groups()
        substation = True  # Force substation for ram compatibility
        seabed_graph = self.select_seabed()

        if seabed_graph is None or seabed_graph.size() == 0:
            raise nx.NetworkXNoPath

        found_layout = False

        for n_cp in groups:
            skip_flag, star_network = self.star_layout(
                device_loc,
                n_cp,
                substation,
                seabed_graph,
            )

            if skip_flag:
                continue
            else:
                found_layout = True

            #                ## If any paths are of length 1, too many cps in area = skip
            #                skip_flag = self.check_path_lengths(cp_cp_paths)
            #
            #                skip_flag = self.check_path_lengths(np.array(cp_device_paths, dtype=object))

            # unpackage array
            cp_device = star_network["cp_device"]
            cp_cp_distances = star_network["cp_cp_distances"]

            # if skip_flag, break out

            min_val, max_val, ave_val, chain = self.device_spacing_summary(
                cp_cp_distances
            )

            # get power of the clusters
            cluster_size = [sum(cluster) for cluster in cp_device]
            cluster_power = [cluster * device_power for cluster in cluster_size]

            array_test_one = self.cross_check_limits(
                device_voltage, min_val, max(cluster_power)
            )

            array_test_two = self.cross_check_limits(
                device_voltage, max_val, max(cluster_power)
            )

        # Check whether a layout was found.
        errStr = (
            "A star network layout could not be designed for the given device "
            "positions"
        )
        if not found_layout or array_test_one is None or array_test_two is None:
            raise RuntimeError(errStr)

        # then compare v voltage levels - take largest power
        if len(set([array_test_one, array_test_two])) == 1:
            array_voltages = self.get_next_voltage([array_test_one])
            if array_voltages is None:
                raise ValueError(errStr)
            self.array_voltages = [array_test_one, array_voltages]
        else:
            self.array_voltages = [array_test_one, array_test_two]

        self.levels = 3
        self.make_voltage_combinations()

    def run_it(self, installation_tool: Optional[str] = None) -> Network:
        """Control logic for designing a radial network."""

        seabed_graph = self.select_seabed(installation_tool)

        if seabed_graph is None or seabed_graph.size() == 0:
            raise nx.NetworkXNoPath

        network_count = 0
        combo_export, combo_array = self.control_simulations()
        device_loc = self.convert_layout_to_numpy()
        burial_targets = self.meta_data.grid.grid_pd[
            ["id", "Target burial depth"]
        ]
        groups = self.star_groups()
        offshore_substation = True  # Force substation for ram compatibility

        for simulation in zip(combo_export, combo_array):
            export_voltage = simulation[0]
            array_voltage = simulation[1]

            # check component database
            components = self.db_compatibility(
                self.meta_data.database,
                self.meta_data.array_data.machine_data.voltage,
                export_voltage,
                array_voltage,
            )

            for n_cp in groups:
                skip_flag, star_network = self.star_layout(
                    device_loc,
                    n_cp,
                    offshore_substation,
                    seabed_graph,
                )

                if skip_flag:
                    msg = (
                        "Star network solution with {} collection points"
                        "skipped - lack of space for cable routes".format(n_cp)
                    )
                    module_logger.info(msg)

                    continue

                # unpackage array
                cp_cp = star_network["cp_to_cp"]
                cp_cp_paths = star_network["cp_cp_paths"]
                cp_device_paths = star_network["cp_device_paths"]
                cp_device = star_network["cp_device"]
                cp_cp_distances = star_network["cp_cp_distances"]
                cp_device_distances = star_network["cp_device_distances"]
                cp_loc = star_network["cp_loc"]

                network_connections = self.convert_to_pypower(
                    n_cp,
                    cp_cp,
                    cp_device,
                    offshore_substation,
                )

                modified_cp_device_paths = []
                modified_cp_device_distances = []

                if offshore_substation:
                    del cp_device_paths[0]
                    del cp_device[0]
                    del cp_device_distances[0]

                if self.floating:
                    umbilical_design = {}
                    umbilical_impedance = []

                    for cable_path, cable_distance, local_sol in zip(
                        cp_device_paths, cp_device_distances, cp_device
                    ):
                        # make sol to pass to design
                        sol = [
                            [0, idx + 1]
                            for idx, val in enumerate(local_sol)
                            if val == 1
                        ]

                        paths = np.array([[0] + cable_path])
                        distances = [0] + cable_distance

                        # call umbilical design model
                        local_umbilical = UmbilicalDesign(self.meta_data)

                        local_umbilical.umbilical_design(
                            paths, components["umbilical"], sol
                        )
                        assert local_umbilical.designs is not None

                        umbilical_design.update(local_umbilical.designs)

                        self.update_static_cables(
                            local_umbilical.designs, paths, sol, distances
                        )

                        modified_cp_device_distances.append(distances[1:])
                        modified_cp_device_paths.append(paths[0][1:])

                    umbilical_impedance = (
                        local_umbilical.umbilical_impedance_table(
                            umbilical_design
                        )
                    )

                else:
                    umbilical_design = None
                    umbilical_impedance = None
                    modified_cp_device_paths = cp_device_paths
                    modified_cp_device_distances = cp_device_distances

                if offshore_substation:
                    n_cp += 1

                    export_length, export_route = connect.get_export(
                        cp_loc[0],
                        self.meta_data.array_data.landing_point,
                        self.meta_data.grid,
                        seabed_graph,
                    )

                    modified_cp_device_distances = [
                        [0] * self.meta_data.array_data.n_devices
                    ] + modified_cp_device_distances

                    cp_device = [
                        [0] * self.meta_data.array_data.n_devices
                    ] + cp_device

                    modified_cp_device_paths = [
                        [0] * self.meta_data.array_data.n_devices
                    ] + modified_cp_device_paths

                else:
                    export_route = cp_cp_paths[0][cp_cp[0][1]]
                    export_length = cp_cp_distances[0][cp_cp[0][1]]

                    # now trim cp cp array
                    cp_cp_distances = np.delete(cp_cp_distances, (0), axis=0)
                    cp_cp_distances = np.delete(cp_cp_distances, (0), axis=1)
                    cp_cp_paths = np.delete(cp_cp_paths, (0), axis=0)
                    cp_cp_paths = np.delete(cp_cp_paths, (0), axis=1)

                network_count = self.iterate_cable_solutions(
                    n_cp,
                    cp_loc,
                    network_connections,
                    network_count,
                    components,
                    np.asarray(modified_cp_device_distances),
                    export_length,
                    export_route,
                    export_voltage,
                    array_voltage,
                    np.asarray(modified_cp_device_paths),
                    burial_targets,
                    umbilical_design,
                    umbilical_impedance,
                    cp_cp_distances,
                    cp_cp_paths,
                )

        min_lcoe = self.check_lcoe()
        network_outputs = self.make_outputs(min_lcoe)

        return network_outputs

    def check_path_lengths(self, paths: np.ndarray) -> bool:
        """Check the off diagonal elements of the path. If any element is
        length = 1 then this is not a valid path.

        Args:
            paths () [-]:

        Attributes:
            flag (bool) [-]: True == invalid path in array; False == all valid
                paths.

        Returns:
            flag

        """

        flag = False
        col, _ = paths.shape

        flat_paths = paths.flatten()

        # convert zero elements to empty tuples
        flat_paths_padded = [
            tuple() if path == 0 else path for path in flat_paths
        ]

        path_lengths = [len(path) for path in flat_paths_padded]

        del path_lengths[0 :: col + 1]  # remove diagonal elements

        if 1 in path_lengths:
            flag = True

        return flag

    def control_simulations(self) -> tuple[list[float], list[float]]:
        if self.voltage_combinations is None:
            return [], []

        v_export: list[float] = []
        v_array: list[float] = []

        for combo in self.voltage_combinations:
            v_export.append(combo[0])
            v_array.append(combo[1])

        return v_export, v_array

    def star_layout(
        self,
        device_loc: np.ndarray,
        n_cp: int,
        substation: bool,
        seabed_graph: nx.Graph,
    ) -> tuple[bool, dict[str, Any]]:
        """Master-slave star layout to achieve single export to shore. This is
        limited to only single connections between a device and collection
        point.

        """

        skip_flag = False
        network_build: dict[str, Any] = {}
        cp_loc, strings = self.set_cp_location(device_loc, n_cp)

        if substation:
            # site substations
            substation_loc, _ = self.set_substation_location(
                device_loc,
                1,
                self.meta_data.options.edge_buffer,
            )
            (cp_cp_sol, cp_cp_distances, cp_cp_paths) = self.connect_cps(
                n_cp,
                1,
                cp_loc,
                substation_loc,
                seabed_graph,
            )
        else:
            target = self.meta_data.array_data.landing_point
            cp_cp_sol, cp_cp_distances, cp_cp_paths = self.connect_cps(
                n_cp,
                n_cp,
                cp_loc,
                target,
                seabed_graph,
            )

        # get device paths to cps
        cp_device = [[0] * self.meta_data.array_data.n_devices] * n_cp
        cp_device_paths = []
        cp_device_distances = []

        for idx, cp in enumerate(cp_loc):
            # get connected devices
            local = self.local_devices(idx, strings, device_loc)
            # make cp to device connection
            local_temp = [0] * self.meta_data.array_data.n_devices

            local_devices = {}
            local_devices_grid_id = []

            for oec in local:
                local_temp[oec] = 1
                key = "Device" + str(oec + 1).zfill(3)
                local_devices[key] = self.meta_data.array_data.layout[key]

                for item in self.meta_data.array_data.layout_grid:
                    if item[0] == int(key[6:]):
                        local_devices_grid_id.append((int(key[6:]), item[1]))

            sorted_local_locs = sorted(
                local_devices_grid_id,
                key=lambda x: x[0],
            )

            cp_device[idx] = local_temp
            distances, paths = connect.calculate_distance_dijkstra(
                sorted_local_locs,
                cp,
                self.meta_data.grid,
                seabed_graph,
            )

            # hard break of loop, invalid solution
            if not distances[0, :].all():
                skip_flag = True
                return (skip_flag, network_build)

            cp_device_distances.append(distances)
            cp_device_paths.append(paths)

        if substation:
            cp_loc = [list(substation_loc)] + cp_loc
            cp_device = [[0] * self.meta_data.array_data.n_devices] + cp_device
            cp_device_distances = [
                [0] * (self.meta_data.array_data.n_devices + 1)
            ] + cp_device_distances
            cp_device_paths = [
                [0] * (self.meta_data.array_data.n_devices + 1)
            ] + cp_device_paths

        # build dictionary
        network_build["cp_to_cp"] = cp_cp_sol
        network_build["cp_cp_paths"] = cp_cp_paths
        network_build["cp_device_paths"] = cp_device_paths
        network_build["cp_device"] = cp_device
        network_build["cp_cp_distances"] = cp_cp_distances
        network_build["cp_device_distances"] = cp_device_distances
        network_build["cp_loc"] = cp_loc

        return (skip_flag, network_build)

    def local_devices(
        self,
        cp_n: int,
        strings: np.ndarray,
        device_loc: np.ndarray,
    ) -> list[int]:
        """Get the local devices of a given collection point."""

        local_devices: list[int] = []

        for device, ownership in enumerate(zip(device_loc, strings)):
            if ownership[1] == cp_n:
                local_devices.append(device)

        return local_devices

    def set_cp_location(
        self,
        device_loc: np.ndarray,
        n_cp: int,
    ) -> tuple[list[tuple[float, ...]], np.ndarray]:
        cp_loc_estimate, _ = kmeans2(device_loc[:, :2], n_cp)
        idx, _ = vq(device_loc[:, :2], cp_loc_estimate)

        grid = np.array(self.meta_data.grid.grid_pd[["x", "y"]])

        cp_loc: list[tuple[float, ...]] = []

        for cp in cp_loc_estimate:
            cp_loc.append(self.snap_to_grid(grid, cp))

        return cp_loc, idx

    def connect_cps(
        self,
        n_cp: int,
        max_: int,
        cp_loc: list[tuple[float, ...]],
        target: tuple[float, ...],
        seabed_graph: nx.Graph,
    ) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
        """Brute force optimisation of cp-to-cp network. Iterate through all
        possible combinations of networks.

        Args:
            arguments (type): Description.

        Attributes:
            attributes (type): Description.

        Returns:
            connect_matrix (list): vector of array connections.

        """

        # initialise route vector
        route_vector: list[tuple[int, int]] = []
        for n in range(0, n_cp):
            interim = (n + 1, 0)
            route_vector.append(interim)

        # initialise path vector, P
        path_vector = [list(reversed(edge)) for edge in route_vector]

        # make cps in dict format for compatibility
        cp_layout = {}
        for idx, cp in enumerate(cp_loc):
            key = "Device" + str(idx + 1).zfill(3)
            cp_layout[key] = cp

        cp_grid_ids = []
        for cp, loc in cp_layout.items():
            grid_id = self.meta_data.grid.grid_pd[
                (self.meta_data.grid.grid_pd.x == loc[0])
                & (self.meta_data.grid.grid_pd.y == loc[1])
            ]["id"].values[0]

            cp_grid_ids.append((int(cp[6:]), grid_id))

        sorted_cp_locs = sorted(cp_grid_ids, key=lambda x: x[0])

        (distance_matrix, path_matrix) = connect.calculate_distance_dijkstra(
            sorted_cp_locs, target, self.meta_data.grid, seabed_graph
        )

        cps = connect.add_device_positions(cp_layout, sorted_cp_locs)

        savings_vector = connect.calculate_saving_vector(distance_matrix, n_cp)

        connect_matrix = connect.make_optimal_paths(
            savings_vector,
            path_vector,
            route_vector,
            max_,
            path_matrix,
            cps,
            self.meta_data.grid,
        )

        return connect_matrix, distance_matrix, path_matrix

    def convert_to_pypower(
        self,
        n_cp: int,
        cp_cp_sol: np.ndarray,
        cp_device: list[list[int]],
        substation: bool,
    ) -> dict[str, np.ndarray]:
        if substation:
            n_cp += 1

        n_devices = self.meta_data.array_data.n_devices

        shore_to_device = np.array([[0] * n_devices])
        shore_to_cp = np.array([0] * n_cp)
        cp_to_cp = np.array([[0] * n_cp] * n_cp)
        device_to_device = np.array([[0] * n_devices] * n_devices)

        # split collection points
        # shore_to_cp[cp_cp_sol[0][1]] = 1 # this location is always to shore

        shore_to_cp[cp_cp_sol[0][1] - 1] = 1  # this location is always to shore

        # Work is here to connect in bin string style
        for item in cp_cp_sol:
            chain = item[1:]

            if len(chain) > 1:
                for idx in range(len(chain) - 1):
                    c_to_c_int = np.array([0] * n_cp)
                    c_to_c_int[chain[idx + 1] - 1] = 1
                    cp_to_cp[chain[idx] - 1] = (
                        cp_to_cp[chain[idx] - 1] + c_to_c_int
                    )

            else:
                c_to_c_int = np.array([0] * n_cp)
                cp_to_cp[item[0]][chain[0]] = 1

        cp_to_cp = self.symmetrize(cp_to_cp)

        network_connections = {
            "shore_to_device": shore_to_device,
            "shore_to_cp": shore_to_cp,
            "cp_to_cp": cp_to_cp,
            "cp_to_device": np.asarray(cp_device),
            "device_to_device": device_to_device,
        }

        return network_connections

    def star_groups(self) -> list[int]:
        """Get cluster sizes for star layout.

        Attributes:
            cp_size (list, float) [-]: Consider collection point sizes.
            groups (list, float) [-]: Cluster size for each cp_size.
            clean_group (list, int) [-]: Cluster size for each cp_size as int.

        Note:
            Possible update to consider only integer solutions.

        """

        #        cp_size = [2., 4., 8.]

        cp_size = [2.0, 4.0, 6.0, 8.0]

        groups = [
            int(np.round(self.meta_data.array_data.n_devices / cp))
            for cp in cp_size
        ]

        clean_groups: list[int] = []

        for group_size in groups:
            if not (group_size == 0 or group_size == 1):
                clean_groups.append(group_size)

        return clean_groups


def select_installation_tool(
    soils: dict[str, float],
    tools: Sequence[str],
    rates: pd.DataFrame,
) -> list[str]:
    if len(soils) == 1:
        # rank tools on single soil
        soil = list(soils.keys())[0]
        tool_order = rates.sort_values(soil, ascending=False).index.tolist()

    else:
        all_solutions = {}

        # do something better
        for tool in tools:
            tool_total = 0

            for soil in soils:
                rate = rates.loc[tool, soil]
                assert isinstance(rate, float)
                tool_total += rate * soils[soil]

            all_solutions[tool] = tool_total

        clean_solutions = {k: v for k, v in all_solutions.items() if v != 0}

        tool_order = sorted(
            clean_solutions,
            key=lambda k: clean_solutions[k],
            reverse=True,
        )

    return tool_order
