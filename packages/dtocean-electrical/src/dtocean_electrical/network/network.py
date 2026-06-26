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
Created on Thu Apr 07 13:32:59 2016

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
.. moduleauthor:: Mathew Topper <damm_horse@yahoo.co.uk>
"""

import logging
from collections import Counter
from copy import deepcopy
from pprint import pformat
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from ..inputs import ElectricalComponentDatabase
from ..optimiser.power_flow import ComponentLoading
from .cable import (
    ArrayCable,
    ExportCable,
    UmbilicalCable,
    get_burial_depths,
    get_split_pipes,
)
from .collection_point import CollectionPoint, PassiveHub, Substation
from .connector import DryMateConnector, WetMateConnector

# Start logging
module_logger = logging.getLogger(__name__)

PointTuple = tuple[float, float, float]


class Network:
    """Data structure for the network description. This is composed of a number
    of network component objects and contains all data required to describe the
    network structure and performance.

    Args:
        index (int) [-]: Unique index of the instance of the Network object.
        configuration (str) [-]: The network configuration, defined as either:
            radial or star.
        power_histogram (list) [pc]: The probability of occurrence of each
            power bin.
        array_power_output (list) [MW]: The array power output for each for
            each bin edge in the power histogram.

    Attributes:
        index (int)
        configuration (str)
        n_cp (int) [-]: The number of collection points.
        n_export (int) [-]: The number of export cables.
        export_voltage (float) [kV]: The export cable voltage.
        array_voltage (float) [kV]: The array cable voltage.
        shore_to_device (nparray) [-]: Shore to device connection matrix.
        device_to_device (nparray) [-]: Device to device connection matrix.
        shore_to_cp (nparray) [-]: Shore to collection point connection matrix.
        cp_to_cp (nparray) [-]: Collection point to collection point connection
            matrix.
        cp_to_device (nparray) [-]: Collection point to device connection
            matrix.
        export_cables (list) [-]: List of ExportCable objects.
        array_cables (list) [-]: List of ArrayCable objects.
        collection_points (list) [-]: List of CollectionPoint objects.
        wet_mate (list) [-]: List of WetMateConnector objects.
        dry_mate (list) [-]: List of DryMateConnector objects.
        seastate_occurrence () [-]
        array_power_output () [-]
        b_o_m (pd.DataFrame) [-]: Network bill of materials;
            db ref (int) [-]: Component database key.
            install_type (str) [-]: Component type.
            marker (int) [-]: Component unique marker.
            quantity (float): Unit quantity of the component. Cable lengths
                given in [m], all others are dimensionless.
            utm_x (float) [m]: UTM coordinate in the x direction.
            utm_y (float) [m]: UTM coordinate in the y direction.
        economics_data (pd.DataFrame) [-]: Total component cost of the network;
            db ref (int) [-]: Component database key.
            quantity (float) [-]: Total unit quantity of all uses of component
                in the network. Cable lengths given in [m], all others are
                dimensionless.
            cost (float) [E]: Total cost of all uses of component in the
                network.
            year (int) [-]: Installation year. Assumed zero for all.
        total_cost (float) [E]: Total cost, i.e. sum, of all components in the
            network.
        all_connections (dict) [-]: Structure which carries all data required
            for hierarchy and network_design. Each component is represented by
            the database id and the unique component marker.
        hierarchy (dict): Structure to carry the component-to-component
            connection relationship. See notes for further information.
        network_design (dict) [-]: Network design in the dictionary format
            required for downstream analysis. See notes for further
            information.
        cable_routes (pd.DataFrame): Cable route information as points;
            db ref (int) [-]: Component database key.
            marker (int) [-]: Component unique marker.
            grid_id (int) [-]: Point object id traversed by cable.
            burial_depth (float) [m]: Burial depth at the specified point.
            split pipe (bool) [-]: Presence of split pipe protection at the
                specified point.
        collection_points_design (pd.DataFrame) [-]: Collection point
            specification for foundation design;
            centre_of_gravity (np.array) [m]: Centre of gravity with respect to
                local coordinate system, as [x,y,z].
            dry beam area (float) [m2]: Dry beam area.
            dry frontal area (float) [m2]: Dry frontal area.
            foundation type (str) [-]: Predefined foundation type, either
                'gravity' or 'pile'.
            foundation locations (np.array) [m]: Foundation location with
                respect to the origin point, as [x,y,z].
            height (float) [m]: Unit height.
            length (float) [m]: Unit length.
            marker (int) [-]: Collection point unique marker.
            mass (float) [kg]: Total unit weight.
            orientation angle (float) [deg]: Device orientation angle.
            origin (np.array) [m]: Collection point origin point, UTM
                coordinates as [x,y].
            profile (str) [-]: Shape, either 'cylindrical' or 'rectangular'.
            surface roughness (float) [m]: Surface roughness.
            type_ (str) [-]: Collection point type, either 'subsea passive',
                'subsea substation' or 'surface substation'.
            volume (float) [m3]: Submerged volumne.
            wet beam area (float) [m2]: Wet beam area.
            wet frontal area (float) [m2]: Wet frontal area.
            width (float) [m]: Unit width.
        annual_yield (float) [Wh]: Array power output for a year period
            less electrical losses.
        annual_losses (float) [Wh]: Electrical losses for a year period.
        annual_efficiency (float) [pc]: Network efficiency for a year period.
        histogram_losses (list) [Wh]: Electrical losses at each power
            generation level.
        histogram_efficiency (list) [pc]: Electrical efficiency at each power
            generation level.

    Returns:
        none

    Note
        The structure of hierarchy is divided by two types of key. 'array'
        denotes the system level connection: 'Export cable' is the connection
        from the onshore landing point to the 'Substation'. 'layout' denotes
        the series/parallel connections of the oec: within the list, comma
        separated values represent series connections, brackets separated by a
        comma indicate a new branch. The 'device' keys indicate the connection
        between the last system and the device. All values are database ids.

        network_design follows the description above but specifies the number
        of unique componens in a given system. Each component is assigned a
        unique marker.

    """

    def __init__(
        self,
        index: int,
        power_histogram: Sequence[float],
        array_power_output: Sequence[float],
        floating: bool,
        export_constraints: ComponentLoading,
        array_constraints: ComponentLoading,
    ):
        # network characteristics
        self.index = index
        self.floating = floating
        self.export_voltage: float = 0.0
        self.array_voltage: float = 0.0
        self.shore_to_device: Optional[np.ndarray] = None
        self.device_to_device: Optional[np.ndarray] = None
        self.shore_to_cp: Optional[np.ndarray] = None
        self.cp_to_cp: Optional[np.ndarray] = None
        self.cp_to_device: Optional[np.ndarray] = None

        # network components
        self.export_cables: list[ExportCable] = []
        self.array_cables: list[ArrayCable] = []
        self.umbilical_cables: list[UmbilicalCable] = []
        self.collection_points: list[CollectionPoint] = []
        self.wet_mate: list[WetMateConnector] = []
        self.dry_mate: list[DryMateConnector] = []

        # assessment states
        self.power_histogram = power_histogram
        self.array_power_output = array_power_output

        self.array_constraints = array_constraints
        self.export_constraints = export_constraints

        # high level description
        self.b_o_m: Optional[pd.DataFrame] = None
        self.economics_data: Optional[pd.DataFrame] = None
        self.total_cost: Optional[float] = None
        self.lcoe: Optional[float] = None
        self.all_connections: Optional[dict[str, Any]] = None
        self.hierarchy: Optional[dict[str, Any]] = None
        self.network_design: Optional[dict[str, Any]] = None
        self.cable_routes: Optional[pd.DataFrame] = None
        self.collection_points_design: Optional[pd.DataFrame] = None
        self.umbilical_cable_design: Optional[pd.DataFrame] = None
        self.annual_yield: Optional[float] = None
        self.annual_losses: Optional[float] = None
        self.annual_efficiency: Optional[float] = None
        self.histogram_losses: Optional[list[float]] = None
        self.histogram_efficiency: Optional[list[float]] = None

    @property
    def n_cp(self) -> int:
        return len(self.collection_points)

    def add_collection_points(
        self,
        cp_locs: list[tuple[float, ...]],
        db_key: int,
        db: pd.DataFrame,
    ):
        """Set collection point object(s) for the network object.

        Args:
            cp_locs (list): list of collection point locations.

        Returns:
            none.

        """

        data = db[db.id == db_key]

        if data.empty:
            raise ValueError("db_key not found in db")

        for cpi, cp_loc in enumerate(cp_locs):
            if data.v1.values[0] == data.v2.values[0]:
                self.collection_points.append(
                    PassiveHub(cpi, cp_loc, db_key, data)
                )
            else:
                self.collection_points.append(
                    Substation(cpi, cp_loc, db_key, data)
                )

    def add_cables(
        self,
        cp_device_distance: np.ndarray,
        cp_cp_distance: Optional[np.ndarray],
        device_connection: str,
        device_layout: dict[str, tuple[float, ...]],
        cp_device_paths: np.ndarray,
        cp_cp_paths: Optional[np.ndarray],
        export_route: Sequence[int],
        export_length: float,
        umbilical_data: Optional[dict[str, dict[str, Any]]],
        components: dict[str, int],
        burial_depths: pd.DataFrame,
        burial_array: Optional[float],
        burial_export: Optional[float],
        shore_to_cp: np.ndarray,
        cp_to_device: np.ndarray,
        device_to_device: np.ndarray,
        cp_to_cp: Optional[np.ndarray] = None,
    ):
        """Add cables to the network.  This also adds connectors at cable ends.
        This also produces the hierarchy and network design dictionaries.

        Args:
            distance_array (nparray): Distance between devices and central
                location.
            device_connection (str): Type of connector.
            device_layout (dict): Device locations.
            path_array (nparray): Seabed paths between devices and central
                location.
            export_route (tuple): Export cable route, defined by grid point
                ids.
            export_length (float): Export cable length.
            umbilical_data
            components
            burial_depths
            burial_array (float) [-]: User defined array cable burial depth.
                Can be None.
            burial_export (float) [-]: User defined export cable burial depth.
                Can be None.

        Attributes:
            marker (int): Unique component marker.
            device_to_device (nparray): Copy of the device to device connection
                matrix.
            visited_nodes (list): List of visisted devices.
            export_idx (int): Export cable index.
            array_idx (int): Array cable index.
            wet_mate_idx (int): Wet mate connector index.
            dry_mate_idx (int): Dry mate connector index.
            hierarchy (dict): Network connection hierarchy for downstream
                analysis.
            array (list): Container for array level information, to be stored
                in heirarchy.

        Returns:
            none.

        """

        if len(shore_to_cp) != self.n_cp:
            msg = (
                "Length of shore_to_cp must equal self.n_cp. Have you called "
                "add_collection_points?"
            )
            raise ValueError(msg)

        if cp_to_device.shape[0] != self.n_cp:
            msg = (
                "First dimension of cp_to_device must equal self.n_cp. Have "
                "you called add_collection_points?"
            )
            raise ValueError(msg)

        if (
            device_to_device.shape[0] != device_to_device.shape[1]
            or device_to_device.shape[0] != cp_to_device.shape[1]
        ):
            msg = (
                "device_to_device must have equal dimensions with length "
                "matching the second dimension of cp_to_device"
            )
            raise ValueError(msg)

        if cp_to_cp is not None and (
            cp_to_cp.shape[0] != cp_to_cp.shape[1]
            or cp_to_cp.shape[0] != self.n_cp
        ):
            msg = (
                "If given, cp_to_cp must have equal dimensions with length "
                "equal to self.n_cp. Have you called add_collection_points?"
            )
            raise ValueError(msg)

        self.shore_to_cp = shore_to_cp
        self.cp_to_device = cp_to_device
        self.device_to_device = device_to_device
        self.cp_to_cp = cp_to_cp

        marker = 0
        export_idx = 0
        array_idx = 0
        wet_mate_idx = 0
        dry_mate_idx = 0
        umbilical_idx = 0

        # vars for dictionary structures
        hierarchy: dict[str, Any] = {}
        array: list[dict[str, Any]] = []
        cp_to_device_copy = deepcopy(self.cp_to_device)
        device_to_device_copy = deepcopy(self.device_to_device)
        cp_to_cp_copy = deepcopy(self.cp_to_cp)

        # iterate through cps connected to shore
        for cp_idx in np.where(self.shore_to_cp > 0)[0]:
            cluster: dict[str, Any] = {"layout": []}

            marker, export_idx = self._add_export_cable(
                cluster,
                marker,
                cp_idx,
                export_idx,
                export_route,
                export_length,
                components["export"],
                burial_depths,
                burial_export,
            )

            marker, wet_mate_idx, dry_mate_idx = self._add_substation(
                cluster,
                hierarchy,
                marker,
                cp_idx,
                wet_mate_idx,
                dry_mate_idx,
                components,
            )

            # Star layout
            if cp_to_cp_copy is not None and cp_to_cp_copy[cp_idx].any():
                marker, array_idx, wet_mate_idx, dry_mate_idx = self._add_star(
                    cluster,
                    hierarchy,
                    marker,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    cp_idx,
                    cp_to_cp_copy,
                    cp_cp_distance,
                    cp_cp_paths,
                    components,
                    burial_depths,
                    burial_array,
                )

            marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx = (
                self._cp_to_devices(
                    cluster,
                    hierarchy,
                    marker,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    umbilical_idx,
                    device_connection,
                    device_layout,
                    cp_to_device_copy,
                    device_to_device_copy,
                    cp_device_distance,
                    cp_device_paths,
                    umbilical_data,
                    components,
                    burial_depths,
                    burial_array,
                )
            )

            array.append(cluster)

        hierarchy["array"] = array
        self.all_connections = hierarchy

    def _add_export_cable(
        self,
        cluster: dict[str, Any],
        marker: int,
        cp_idx: int,
        export_idx: int,
        export_route: Sequence[int],
        export_length: float,
        db_key: int,
        burial_depths: pd.DataFrame,
        burial_export: Optional[float],
    ) -> tuple[int, int]:
        burial = get_burial_depths(
            export_route,
            burial_depths,
            burial_export,
        )

        split_pipe = get_split_pipes(burial)

        self.export_cables.append(
            ExportCable(
                export_idx,
                export_length,
                db_key,
                marker,
                export_route,
                burial,
                split_pipe,
                "collection point",
                cp_idx,
            )
        )

        cluster["Export cable"] = [(db_key, marker)]
        export_idx += 1
        marker += 1

        return marker, export_idx

    def _add_substation(
        self,
        cluster: dict[str, Any],
        hierarchy: dict[str, Any],
        marker: int,
        cp_idx: int,
        wet_mate_idx: int,
        dry_mate_idx: int,
        components: dict[str, int],
    ):
        """Create substation

        Need to make sure that each is given a marker, then update the marker
        of the collection point. Need to handle differently if there is no
        substation."""
        cp = self.collection_points[cp_idx]

        if isinstance(cp, PassiveHub):
            subhub_key = "subhub" + str(cp_idx).zfill(3)
            cluster["Substation"] = ["Ideal"]
            cluster["layout"].append(subhub_key)

            cp.marker = marker
            hierarchy[subhub_key] = {
                "Elec sub-system": [],
                "Substation": [(cp.db_key, cp.marker)],
            }

        else:
            # need to add reference to export side connector for
            # installation

            export_connector = self.collection_points[cp_idx].input_connectors
            db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
                export_connector,
                wet_mate_idx,
                dry_mate_idx,
                marker,
                (cp.utm_x, cp.utm_y),
                components,
            )

            if "Export cable" not in cluster:
                raise RuntimeError("Export cable must exist in cluster")

            cluster["Export cable"].append((db_key, marker))
            marker += 1

            cp.marker = marker
            cluster["Substation"] = [(cp.db_key, cp.marker)]

        marker += 1

        return marker, wet_mate_idx, dry_mate_idx

    def _add_star(
        self,
        cluster: dict[str, Any],
        hierarchy: dict[str, Any],
        marker: int,
        array_idx: int,
        wet_mate_idx: int,
        dry_mate_idx: int,
        connection: int,
        cp_to_cp: np.ndarray | None,
        cp_cp_distance: np.ndarray | None,
        cp_cp_paths: np.ndarray | None,
        components: dict[str, int],
        burial_depths: pd.DataFrame,
        burial_array: Optional[float],
    ):
        if cp_to_cp is None:
            raise ValueError("cp_to_cp must be set if configuration is 'Star'")

        if cp_cp_distance is None:
            raise ValueError(
                "cp_cp_distance must be set if configuration is 'Star'"
            )

        if cp_cp_paths is None:
            raise ValueError(
                "cp_cp_paths must be set if configuration is 'Star'"
            )

        cp_to_cp[:, connection] = 0
        cp_layout = []

        for next_cp in np.where(cp_to_cp[connection] > 0)[0]:
            cp_to_cp[:, next_cp] = 0
            layout = []

            # Link to previous cp
            # need to add reference to previous cp side connector for
            # installation - keep as ideal
            array_connector = self.collection_points[
                connection
            ].output_connectors

            db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
                array_connector,
                wet_mate_idx,
                dry_mate_idx,
                marker,
                (
                    self.collection_points[connection].utm_x,
                    self.collection_points[connection].utm_y,
                ),
                components,
            )

            link_to_cp = []
            link_to_cp.append((db_key, marker))
            marker += 1

            cable_length = float(cp_cp_distance[connection][next_cp])
            route = [int(cp) for cp in cp_cp_paths[connection][next_cp]]
            db_key = components["array"]

            burial = get_burial_depths(route, burial_depths, burial_array)
            split_pipe = get_split_pipes(burial)

            self.array_cables.append(
                ArrayCable(
                    array_idx,
                    cable_length,
                    db_key,
                    marker,
                    route,
                    burial,
                    split_pipe,
                    "collection point",
                    "collection point",
                    connection,
                    next_cp,
                )
            )

            link_to_cp.append((db_key, marker))
            marker += 1
            array_idx += 1

            # Link at current cp
            # need to add reference to previous cp side connector for
            # installation - keep as ideal
            array_connector = self.collection_points[next_cp].input_connectors
            db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
                array_connector,
                wet_mate_idx,
                dry_mate_idx,
                marker,
                (
                    self.collection_points[connection].utm_x,
                    self.collection_points[connection].utm_y,
                ),
                components,
            )

            link_to_cp.append((db_key, marker))
            marker += 1

            layout.append("subhub" + str(next_cp).zfill(3))
            hierarchy["subhub" + str(next_cp).zfill(3)] = {}

            self.collection_points[next_cp].marker = marker
            marker += 1

            hierarchy["subhub" + str(next_cp).zfill(3)].update(
                {
                    "Elec sub-system": link_to_cp,
                    "Substation": [
                        (
                            self.collection_points[next_cp].db_key,
                            self.collection_points[next_cp].marker,
                        )
                    ],
                }
            )
            start_node = next_cp

            while np.any(cp_to_cp[next_cp] > 0):
                next_cp = int(np.where(cp_to_cp[next_cp] > 0)[0])
                cp_to_cp[:, next_cp] = 0

                cable_length = cp_cp_distance[start_node][next_cp]
                route = cp_cp_paths[start_node][next_cp]
                db_key = components["array"]

                burial = get_burial_depths(route, burial_depths, burial_array)
                split_pipe = get_split_pipes(burial)

                self.array_cables.append(
                    ArrayCable(
                        array_idx,
                        cable_length,
                        db_key,
                        marker,
                        route,
                        burial,
                        split_pipe,
                        "collection point",
                        "collection point",
                        start_node,
                        next_cp,
                    )
                )
                marker += 1
                array_idx += 1

                layout.append("subhub" + str(next_cp).zfill(3))
                hierarchy["subhub" + str(next_cp).zfill(3)] = {}

                self.collection_points[next_cp].marker = marker
                marker += 1

                hierarchy["subhub" + str(next_cp).zfill(3)].update(
                    {
                        "Elec sub-system": [
                            (db_key, self.collection_points[next_cp].marker)
                        ],
                        "Substation": [
                            (
                                self.collection_points[next_cp].db_key,
                                self.collection_points[next_cp].marker,
                            )
                        ],
                    }
                )
                start_node = next_cp

            cp_layout.append(layout)

        cluster["layout"] = cp_layout

        return marker, array_idx, wet_mate_idx, dry_mate_idx

    def _cp_to_devices(
        self,
        cluster: dict[str, Any],
        hierarchy: dict[str, Any],
        marker: int,
        array_idx: int,
        wet_mate_idx: int,
        dry_mate_idx: int,
        umbilical_idx: int,
        device_connection: str,
        device_layout: dict[str, tuple[float, ...]],
        cp_to_device: np.ndarray,
        device_to_device: np.ndarray,
        cp_device_distance: np.ndarray,
        cp_device_paths: np.ndarray,
        umbilical_data: dict[str, dict[str, Any]] | None,
        components: dict[str, int],
        burial_depths: pd.DataFrame,
        burial_array: Optional[float],
    ):
        if self.cp_to_device is None:
            return

        visited_nodes = []

        for cp_idx, devices in enumerate(self.cp_to_device):
            subhub_key = "subhub" + str(cp_idx).zfill(3)
            sub_hub_layout: list[list[str]] = []

            for dev_idx in np.where(devices > 0)[0]:
                layout: list[str] = []
                link_to_cp: list[tuple[int, int]] = []
                dev_key_lower = "device" + str(dev_idx + 1).zfill(3)

                layout.append(dev_key_lower)
                visited_nodes.append(dev_idx)

                # Link to cp
                # need to add reference to export side connector for
                # installation - keep as ideal
                array_connector = self.collection_points[
                    cp_idx
                ].output_connectors
                db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
                    array_connector,
                    wet_mate_idx,
                    dry_mate_idx,
                    marker,
                    (
                        self.collection_points[cp_idx].utm_x,
                        self.collection_points[cp_idx].utm_y,
                    ),
                    components,
                )

                link_to_cp.append((db_key, marker))
                marker += 1

                cable_length = cp_device_distance[cp_idx][dev_idx + 1]
                db_key = components["array"]
                route = cp_device_paths[cp_idx][dev_idx + 1]
                burial = get_burial_depths(route, burial_depths, burial_array)

                marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx = (
                    self._add_device(
                        link_to_cp,
                        marker,
                        dev_idx,
                        array_idx,
                        wet_mate_idx,
                        dry_mate_idx,
                        umbilical_idx,
                        cable_length,
                        db_key,
                        route,
                        burial,
                        "collection point",
                        cp_idx,
                        device_connection,
                        device_layout,
                        components,
                        umbilical_data,
                    )
                )

                hierarchy[dev_key_lower] = {"Elec sub-system": link_to_cp}
                cp_to_device[cp_idx][dev_idx] = 0

                (
                    marker,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    umbilical_idx,
                ) = self._device_to_device(
                    layout,
                    hierarchy,
                    visited_nodes,
                    dev_idx,
                    marker,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    umbilical_idx,
                    device_connection,
                    device_layout,
                    device_to_device,
                    cp_device_distance,
                    cp_device_paths,
                    components,
                    burial_depths,
                    burial_array,
                    umbilical_data,
                )

                if subhub_key in hierarchy:
                    sub_hub_layout.append(layout)
                else:
                    cluster["layout"].append(layout)

            if subhub_key in hierarchy:
                hierarchy[subhub_key]["layout"] = sub_hub_layout

        return marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx

    def _device_to_device(
        self,
        layout: list[str],
        hierarchy: dict[str, Any],
        visited_nodes: list[int],
        dev_idx: int,
        marker: int,
        array_idx: int,
        wet_mate_idx: int,
        dry_mate_idx: int,
        umbilical_idx: int,
        device_connection: str,
        device_layout: dict[str, tuple[float, ...]],
        device_to_device: np.ndarray,
        cp_device_distance: np.ndarray,
        cp_device_paths: np.ndarray,
        components: dict[str, int],
        burial_depths: pd.DataFrame,
        burial_array: Optional[float],
        umbilical_data: dict[str, dict[str, Any]] | None,
    ):
        start = dev_idx

        while np.any(device_to_device[dev_idx] > 0):
            next_devices = np.where(device_to_device[dev_idx] > 0)[0]

            # filter against visited nodes
            next_device: int | None = None

            for node in next_devices:
                if node not in visited_nodes:
                    next_device = node

            # Every node has been visited
            if next_device is None:
                return (
                    marker,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    umbilical_idx,
                )

            dev_idx = next_device
            next_dev_key_lower = "device" + str(dev_idx + 1).zfill(3)

            elec_sub_system: list[tuple[int, int]] = []
            layout.append(next_dev_key_lower)
            visited_nodes.append(dev_idx)

            # add static cable between connectors
            cable_length = cp_device_distance[start + 1][dev_idx + 1]
            db_key = components["array"]
            route = cp_device_paths[start + 1][dev_idx + 1]
            burial = get_burial_depths(route, burial_depths, burial_array)

            marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx = (
                self._add_device(
                    elec_sub_system,
                    marker,
                    dev_idx,
                    array_idx,
                    wet_mate_idx,
                    dry_mate_idx,
                    umbilical_idx,
                    cable_length,
                    db_key,
                    route,
                    burial,
                    "connector" if self.floating else "device",
                    marker - 3 if self.floating else start,
                    device_connection,
                    device_layout,
                    components,
                    umbilical_data,
                )
            )

            hierarchy[next_dev_key_lower] = {"Elec sub-system": elec_sub_system}
            device_to_device[start][dev_idx] = 0
            device_to_device[dev_idx][start] = 0
            start = dev_idx

        return marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx

    def _add_device(
        self,
        elec_sub_system: list[tuple[int, int]],
        marker: int,
        dev_idx: int,
        array_idx: int,
        wet_mate_idx: int,
        dry_mate_idx: int,
        umbilical_idx: int,
        array_cable_length: float,
        array_db_key: int,
        array_route: list[int],
        array_burial: list[float],
        array_downstream_type: str,
        array_downstream_id: int,
        device_connection: str,
        device_layout: dict[str, tuple[float, ...]],
        components: dict[str, int],
        umbilical_data: dict[str, dict[str, Any]] | None,
    ) -> tuple[int, int, int, int, int]:
        if self.floating and umbilical_data is None:
            raise ValueError("umbilical_data must be set if 'floating' is True")

        dev_key_upper = "Device" + str(dev_idx + 1).zfill(3)
        split_pipe = get_split_pipes(array_burial)

        self.array_cables.append(
            ArrayCable(
                array_idx,
                array_cable_length,
                array_db_key,
                marker,
                array_route,
                array_burial,
                split_pipe,
                "connector" if self.floating else "device",
                array_downstream_type,
                marker + 1 if self.floating else dev_idx,
                array_downstream_id,
            )
        )

        # add static cable to layout
        elec_sub_system.append((array_db_key, marker))

        marker += 1
        array_idx += 1

        # add connector to layout (either to device or umbilical)
        if self.floating:
            assert umbilical_data is not None
            location = umbilical_data[dev_key_upper]["termination"]
        else:
            location = device_layout[dev_key_upper]

        db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
            device_connection,
            wet_mate_idx,
            dry_mate_idx,
            marker,
            location,
            components,
        )

        elec_sub_system.append((db_key, marker))
        marker += 1

        if self.floating:
            assert umbilical_data is not None

            # add dynamic cable to layout
            cable = umbilical_data[dev_key_upper]
            self.umbilical_cables.append(
                UmbilicalCable(
                    umbilical_idx,
                    cable["length"],
                    cable["db_key"],
                    marker,
                    cable["termination"],
                    cable["device"],
                    cable["x coords"],
                    cable["z coords"],
                )
            )

            elec_sub_system.append((cable["db_key"], marker))

            umbilical_idx += 1
            marker += 1

            # add device connector to layout
            location = device_layout[dev_key_upper]
            db_key, wet_mate_idx, dry_mate_idx = self._add_connector(
                device_connection,
                wet_mate_idx,
                dry_mate_idx,
                marker,
                location,
                components,
            )
            elec_sub_system.append((db_key, marker))
            marker += 1

        return marker, array_idx, wet_mate_idx, dry_mate_idx, umbilical_idx

    def make_hierarchy(self, n_device: int):
        """Make the network hierarchy for downstream analysis.

        Args:
            n_device (int): The number of OECs in the array.

        Attributes:
            hier (dict): Network connection hierarchy for downstream analysis.
            array (list): Container for array level information, to be stored
                in hierarchy.
            sub_array (dict): Container for sub array level information, to be
                stored in array.

        Returns:
            none.

        """

        if self.all_connections is None:
            self.hierarchy = None
            return

        hier = {}
        index = "db"

        # devices
        for oec in range(n_device):
            hier["device" + str(oec + 1).zfill(3)] = {}
            hier["device" + str(oec + 1).zfill(3)]["Elec sub-system"] = []

            local_system = []

            for item in self.all_connections["device" + str(oec + 1).zfill(3)][
                "Elec sub-system"
            ]:
                local_system.append(item[0])

            hier["device" + str(oec + 1).zfill(3)]["Elec sub-system"].append(
                local_system
            )

        # array
        for item in self.all_connections["array"]:
            sub_array = dict.fromkeys(["Substation", "Export cable", "layout"])

            sub_array["layout"] = item["layout"]

            for system in ["Substation", "Export cable"]:
                sub_array[system] = [
                    self.get_network_components(system, item, index)
                ]

        hier["array"] = sub_array

        # sub hubs
        for key, val in self.all_connections.items():
            if "subhub" in key:
                sub_array = dict.fromkeys(
                    ["Substation", "Elec sub-system", "layout"]
                )

                sub_array["layout"] = val["layout"]

                for system in ["Substation", "Elec sub-system"]:
                    sub_array[system] = [
                        self.get_network_components(system, val, index)
                    ]

                hier[key] = sub_array

        self.hierarchy = hier

    def make_network_design(self, n_device: int):
        """Make the network design table for downstream analysis."""

        if self.all_connections is None:
            self.network_design = None
            return

        design = {}

        for oec in range(n_device):
            design["device" + str(oec + 1).zfill(3)] = {}
            design["device" + str(oec + 1).zfill(3)] = {"marker": []}

            local_system = []

            for item in self.all_connections["device" + str(oec + 1).zfill(3)][
                "Elec sub-system"
            ]:
                local_system.append(item[1])

            design["device" + str(oec + 1).zfill(3)]["marker"].append(
                local_system
            )

            design["device" + str(oec + 1).zfill(3)].update(
                {
                    "quantity": Counter(
                        component[0]
                        for component in self.all_connections[
                            "device" + str(oec + 1).zfill(3)
                        ]["Elec sub-system"]
                    )
                }
            )

        # sub hubs
        for key, val in self.all_connections.items():
            if "subhub" in key:
                design[key] = {}
                design[key] = {"marker": []}

                local_markers = []
                local_keys = []

                for item in self.all_connections[key]["Elec sub-system"]:
                    local_markers.append(item[1])
                    local_keys.append(item[0])

                for item in self.all_connections[key]["Substation"]:
                    local_markers.append(item[1])
                    local_keys.append(item[0])

                design[key]["marker"].append(local_markers)

                counter = Counter()

                for item in local_keys:
                    counter[item] += 1

                design[key].update({"quantity": counter})

        # array
        index = "marker"

        for item in self.all_connections["array"]:
            sub_array = dict.fromkeys(["Substation", "Export cable"], {})

            for system in ["Substation", "Export cable"]:
                components = self.get_network_components(system, item, index)

                sub_array[system] = {"marker": [components]}

                count = Counter(component[0] for component in item[system])

                sub_array[system].update({"quantity": count})

        design["array"] = sub_array

        self.network_design = design

    def get_network_components(self, system_name, system, index):
        if index == "db":
            i = 0

        else:
            i = 1

        local_system = [component[i] for component in system[system_name]]

        return local_system

    def make_bom(self):
        """Make the network bill of materials for downstream analysis. For each
        component get the marker, db ref, type, utm x, utm y and quantity. Then
        collate in pandas table.

        Attributes:
            markers (list): List of all component markers.
            db_key (list): List of all component database keys.
            install_type (list): List of all component types.
            utm_x (list): List of all component locations, x coordinate.
            utm_y (list): List of all component locations, y coordinate.
            quantity (list): Quantity of all components. This is '1 unit' for
                all components except cables which gives the cable length.
            b_o_m_dict (dict): Structured dictionary for converting to pandas
                dataframe.

        Note:
            This could be improved by making the list creation a function?

        """

        markers: list[int | Literal["None"]] = []
        db_key: list[int] = []
        install_type: list[str] = []
        utm_x: list[float | Literal["None"]] = []
        utm_y: list[float | Literal["None"]] = []
        quantity: list[int | float] = []

        # for all possible components
        for item in self.wet_mate:
            markers.append(item.marker)
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append(item.utm_x)
            utm_y.append(item.utm_y)
            quantity.append(1)

        for item in self.dry_mate:
            markers.append(item.marker)
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append(item.utm_x)
            utm_y.append(item.utm_y)
            quantity.append(1)

        for item in self.export_cables:
            assert item.type_ is not None
            markers.append(item.marker)
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append("None")
            utm_y.append("None")
            quantity.append(item.length)

        for item in self.array_cables:
            assert item.type_ is not None
            markers.append(item.marker)
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append("None")
            utm_y.append("None")
            quantity.append(item.length)

        for item in self.collection_points:
            markers.append(item.marker if item.marker is not None else "None")
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append(item.utm_x)
            utm_y.append(item.utm_y)
            quantity.append(1)

        for item in self.umbilical_cables:
            assert item.type_ is not None
            markers.append(item.marker)
            db_key.append(item.db_key)
            install_type.append(item.type_)
            utm_x.append("None")
            utm_y.append("None")
            quantity.append(item.length)

        # make the pandas table
        b_o_m_dict = {
            "marker": markers,
            "db ref": db_key,
            "install_type": install_type,
            "utm_x": utm_x,
            "utm_y": utm_y,
            "quantity": quantity,
        }

        self.b_o_m = pd.DataFrame(b_o_m_dict)

    def _get_db_keys_from_pd(self) -> list[int]:
        """Get all database keys of all components used in the array.

        Attributes:
            keys (set) [-]: DB keys.

        Returns:
            list [-]: List keys.

        """

        if self.b_o_m is None:
            return []

        keys = set(self.b_o_m["db ref"])
        return list(keys)

    def set_economics_data(
        self,
        db: ElectricalComponentDatabase,
        onshore_cost: Optional[float] = None,
    ):
        """Compile network design data into economics bill of materials.

        Args:
            db (pd) [-]: The component database.

        Attributes:
            network_keys (list) [-]: Database keys of all components used in
                the array.
            quantity (list) [-]: Total quantity of each unique component
                database key used in the array.
            economics_dict (dict) [-]: Structured dictionary for converting to
                pandas dataframe.

        Returns:
            none.

        """

        if self.b_o_m is None:
            self.economics_data = None
            return

        network_keys: list[Any] = self._get_db_keys_from_pd()
        quantity = []
        type_ = []

        for key in network_keys:
            type_.append(
                self.b_o_m[
                    self.b_o_m["db ref"] == key
                ].install_type.values.tolist()
            )

            quantity.append(
                self.b_o_m[self.b_o_m["db ref"] == key].sum()["quantity"]
            )

        type_ = [item[0] for item in type_]  # get item type without using set
        type_ = self._map_component_types(type_)

        cost = self._get_costs_from_db(db, network_keys, type_)

        if onshore_cost:
            network_keys.append(None)
            quantity.append(1)
            cost.append(onshore_cost)

        economics_dict = {
            "db ref": network_keys,
            "quantity": quantity,
            "cost": cost,
            "year": [0] * len(network_keys),
        }

        self.economics_data = pd.DataFrame(economics_dict)

    @classmethod
    def _map_component_types(cls, type_: Sequence[str]) -> list[str]:
        """Map component types. Required to ensure compatibility between names
        used in the install modules and the electrical module.

        Args:
            type_ (list) [-]: Component type list.

        Attributes:
            name_map (dict) [-]: Key = installation module label,
                                 value = electrical database label.

        Returns:
            list [-]: Component types to match electrical database.

        """

        name_map = {
            "export": "export_cable",
            "array": "array_cable",
            "wet-mate": "wet_mate_connectors",
            "dry-mate": "dry_mate_connectors",
            "substation": "collection_points",
            "passive hub": "collection_points",
            "umbilical": "dynamic_cable",
        }

        return [name_map[item] for item in type_]

    def _get_costs_from_db(
        self,
        db: ElectricalComponentDatabase,
        keys: list[int],
        type_: list[str],
    ) -> list[float]:
        """For each component, extract unit cost from the database.

        Args:
            db () [-]:
            keys () [-]:
            type_ () [-]:

        Attributes:
            all_cost (list) [E]: Total component cost.

        Returns:
            all_cost


        """

        all_cost = []

        for component in zip(keys, type_):
            db_dict = getattr(db, component[1])
            all_cost.append(db_dict[db_dict.id == component[0]].cost.values[0])

        return all_cost

    def total_network_cost(self):
        """Calculate total network cost and set total cost attribute.

        Returns:
            none.

        """
        if self.economics_data is None:
            self.total_cost = None
            return

        self.total_cost = sum(
            self.economics_data.cost * self.economics_data.quantity
        )

    def make_cable_routes(
        self,
        grid: pd.DataFrame,
        all_x: Sequence[float],
        all_y: Sequence[float],
    ):
        """Collect the cable routes in pd.DataFrame for downstream analysis.

        Args:
            grid
            all_x
            all_y

        """

        marker = []
        db_ref = []
        grid_id = []
        burial_depth = []
        split_pipe = []

        for cable in self.array_cables + self.export_cables:
            marker += [cable.marker] * len(cable.route)
            db_ref += [cable.db_key] * len(cable.route)
            burial_depth += cable.target_burial_depth
            split_pipe += cable.split_pipe
            grid_id += cable.route

        cable_x, cable_y = self._convert_path_to_coordinates(
            grid_id, all_x, all_y
        )

        indexed_grid = grid.set_index("id")
        cable_depth = indexed_grid["layer 1 start"].loc[grid_id]
        cable_type = indexed_grid["layer 1 type"].loc[grid_id]

        cable_dict = {
            "marker": marker,
            "db ref": db_ref,
            "burial_depth": burial_depth,
            "split pipe": split_pipe,
            "x": cable_x,
            "y": cable_y,
            "layer 1 start": cable_depth,
            "layer 1 type": cable_type,
        }

        self.cable_routes = pd.DataFrame(cable_dict)

    def make_collection_point_design(self):
        """Collection point output data structure.

        Args:
            none

        Attributes:
            centre_of_gravity (np.array) [m]: Centre of gravity with respect to
                local coordinate system, as [x,y,z].
            dry beam area (float) [m2]: Dry beam area.
            dry frontal area (float) [m2]: Dry frontal area.
            foundation type (str) [-]: Predefined foundation type, either
                'gravity' or 'pile'.
            foundation locations (np.array) [m]: Foundation location with
                respect to the origin point, as [x,y,z].
            height (float) [m]: Unit height.
            length (float) [m]: Unit length.
            marker (int) [-]: Collection point unique marker.
            mass (float) [kg]: Total unit weight.
            orientation angle (float) [deg]: Unit orientation angle.
            origin (np.array) [m]: Collection point origin point, UTM
                coordinates as [x,y].
            profile (str) [-]: Shape, either 'cylindrical' or 'rectangular'.
            surface roughness (float) [m]: Surface roughness.
            type_ (str) [-]: Collection point type, either 'subsea passive',
                'subsea substation' or 'surface substation'.
            volume (float) [m3]: Submerged volumne.
            wet beam area (float) [m2]: Wet beam area.
            wet frontal area (float) [m2]: Wet frontal area.
            width (float) [m]: Unit width.

        Returns:
            none

        Note:
            All attributes within container list for conversion to pandas
            DataFrame.

        """

        marker = []
        origin = []
        operating_environment = []
        foundation_type = []
        mass = []
        volume = []
        centre_of_gravity = []
        wet_frontal_area = []
        wet_beam_area = []
        dry_frontal_area = []
        dry_beam_area = []
        length = []
        width = []
        height = []
        profile = []
        surface_roughness = []
        orientation_angle = []
        foundation_locations = []

        for cp in self.collection_points:
            marker.append(cp.marker)
            origin.append(np.array((cp.utm_x, cp.utm_y)))
            foundation_type.append(cp.foundation_type)
            operating_environment.append(cp.operating_environment)
            mass.append(cp.mass)
            length.append(cp.length)
            width.append(cp.width)
            height.append(cp.height)
            volume.append(cp.volume)
            centre_of_gravity.append(np.array((0.0, 0.0, 0.0)))
            profile.append(cp.profile)
            wet_frontal_area.append(cp.wet_frontal_area)
            wet_beam_area.append(cp.wet_beam_area)
            dry_frontal_area.append(cp.dry_frontal_area)
            dry_beam_area.append(cp.dry_beam_area)
            surface_roughness.append(cp.surface_roughness)
            orientation_angle.append(cp.orientation_angle)
            foundation_locations.append(np.array((0.0, 0.0, 0.0)))

        collection_point_dict = {
            "marker": marker,
            "origin": origin,
            "type": operating_environment,
            "mass": mass,
            "volume": volume,
            "length": length,
            "width": width,
            "height": height,
            "centre_of_gravity": centre_of_gravity,
            "profile": profile,
            "wet frontal area": wet_frontal_area,
            "wet beam area": wet_beam_area,
            "dry frontal area": dry_frontal_area,
            "dry beam area": dry_beam_area,
            "surface roughness": surface_roughness,
            "orientation angle": orientation_angle,
            "foundation locations": foundation_locations,
        }

        self.collection_points_design = pd.DataFrame(collection_point_dict)

    def calculate_power_quantities(self, ideal_yield, ideal_histogram):
        """Processing of power quantities for network assessment.

        Args:
            ideal_yield (float): Array power output for a year period assuming
                no electrical losses.

            ideal_histogram (list): Array power output at each power generation
                level assuming no electrical losses.

        Returns:
            none.

        """

        self.annual_yield = self.calculate_annual_yield()
        self.annual_losses = self.calculate_annual_losses(ideal_yield)
        self.annual_efficiency = self.annual_yield / ideal_yield
        self.histogram_losses = self.calculate_histogram_losses(ideal_histogram)
        self.histogram_efficiency = self.calculate_histogram_efficiency(
            ideal_histogram
        )

    def calculate_annual_yield(self) -> float:
        """Calculate annual energy yield.

        Returns:
            annual_yield (float):  Array power output for a year period with
                electrical losses.

        """

        year_hours = 365 * 24
        annual_yield = 0.0

        for time, power in zip(self.power_histogram, self.array_power_output):
            # Convert from MW to W and ignore directionality
            power_w = abs(power * 1e6)

            annual_yield += time * year_hours * power_w

        # Correct for bad calculations
        if np.isnan(annual_yield):
            annual_yield = 0.0

        return annual_yield

    def calculate_annual_losses(self, ideal_yield: float) -> float:
        """Calculate annual energy losses by comparing against ideal.

        Args:
            ideal_yield (float): Array power output for a year period assuming
                no electrical losses.

        Returns:
            annual_losses (float): Electrical losses for a year period.

        """

        if self.annual_yield is None:
            return ideal_yield

        annual_losses = ideal_yield - self.annual_yield

        return annual_losses

    def calculate_histogram_losses(
        self,
        ideal_histogram: list[float],
    ) -> list[float]:
        """Calculate histogram losses by comparing against ideal.

        Args:
            ideal_histogram (list): Array power output at each power generation
                level assuming no electrical losses.

        Returns:
            histogram_losses (list): Electrical losses at each power
                generation level.

        """

        histogram_losses: list[float] = [
            ideal - (abs(actual) * 1000000.0)
            for ideal, actual in zip(ideal_histogram, self.array_power_output)
        ]

        return histogram_losses

    def calculate_histogram_efficiency(
        self,
        ideal_histogram: list[float],
    ) -> list[float]:
        """Calculate array efficiency at each bin in the power histogram.

        Args:
            ideal_histogram (list): Array power output at each power generation
                level assuming no electrical losses.

        Returns:
            histogram_efficiency (list): Electrical efficiency at each power
                generation level.

        """

        histogram_efficiency = [
            (abs(actual) * 1000000) / ideal
            for actual, ideal in zip(self.array_power_output, ideal_histogram)
        ]

        return histogram_efficiency

    def calculate_lcoe(self):
        """Simplified LCOE for comparison of electrical networks."""

        if self.annual_yield is None or self.total_cost is None:
            self.lcoe = None
        elif self.annual_yield == 0.0:
            self.lcoe = np.inf
        else:
            self.lcoe = self.total_cost / self.annual_yield * 1e3

        module_logger.debug("LCOE: {}".format(self.lcoe))

    def make_umbilical_table(self):
        """Quick fix to make the umbilical data table.

        For each umbilical get: the marker, db ref, device, seabed connection
        point and length. Then Collate in a pandas table.

        Attributes:
            marker (list): Umbilical cable markers.
            db_key (list): Umbilical cable database keys.
            device (list): Associated device.
            seabed_connection_point (list): Umbilical cable connection points.
            length (list): Umbilical cable lengths.
            umbilical_dict (dict): Structured dictionary for converting to
                pandas dataframe.

        Note:
            This could be improved by making the list creation a function?

        """

        marker = []
        db_key = []
        device = []
        seabed_connection_point = []
        length = []

        for cable in self.umbilical_cables:
            marker.append(cable.marker)
            db_key.append(cable.db_key)
            device.append(cable.device)
            seabed_connection_point.append(
                np.array(
                    [
                        cable.seabed_termination_x,
                        cable.seabed_termination_y,
                        cable.seabed_termination_z,
                    ]
                )
            )
            length.append(cable.length)

        umbilical_dict = {
            "marker": marker,
            "db ref": db_key,
            "device": device,
            "seabed_connection_point": seabed_connection_point,
            "length": length,
        }

        self.umbilical_cable_design = pd.DataFrame(umbilical_dict)

    def _make_result_str(self) -> str:
        msg = "\n"
        msg += "Annual yield: {}\n\n".format(self.annual_yield)
        msg += "Bill of Materials:\n\n"
        msg += "{}\n\n".format(self.economics_data)
        msg += "Component Data:\n\n"
        msg += "{}\n\n".format(self.b_o_m)
        msg += "Hierarchy:\n\n"
        msg += "{}\n\n".format(pformat(self.hierarchy))
        msg += "Network design:\n\n"
        msg += "{}\n\n".format(pformat(self.network_design))
        msg += "Cable routes:\n\n"
        msg += "{}\n\n".format(self.cable_routes)
        msg += "Collection points:\n\n"
        msg += "{}\n\n".format(self.collection_points_design)
        msg += "Umbilical cables:\n\n"
        msg += "{}\n\n".format(self.umbilical_cable_design)

        return msg

    def print_result(self):
        print(self._make_result_str())

    def log_result(self):
        msg = self._make_result_str()
        module_logger.info(msg)

    def _convert_path_to_coordinates(
        self,
        grid_id: Sequence[int],
        all_x: Sequence[float],
        all_y: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        """Convert a list of grid points into x and y coordinates.

        Args:
            grid_id (list) [-]: List of grid point ids.
            all_x (list) [m]: List of all x coordinates in the area.
            all_y (list) [m]: List of all y coordinates in the area.

        Attributes:
            x (list) [m]: List of x coordinates traversed by cables.
            y (list) [m]: List of y coordinates traversed by cables.

        Returns:
            x
            y

        Note:
            Faster to perform two list separate list comprehensions?
            Grid point id is set at input of module for internal use only.

        """

        x, y = zip(*[(all_x[point], all_y[point]) for point in grid_id])
        return list(x), list(y)

    def _add_connector(
        self,
        connection_type: str,
        wet_mate_idx: int,
        dry_mate_idx: int,
        marker: int,
        location: tuple[float, ...],
        components: dict[str, int],
    ) -> tuple[int, int, int]:
        """Define connector in the network."""

        match connection_type:
            case "wet-mate":
                db_key = components["wet_connector"]
                self.wet_mate.append(
                    WetMateConnector(wet_mate_idx, db_key, marker, location)
                )
                wet_mate_idx += 1

            case "dry-mate":
                db_key = components["dry_connector"]
                self.dry_mate.append(
                    DryMateConnector(dry_mate_idx, db_key, marker, location)
                )
                dry_mate_idx += 1

            case _:
                raise ValueError("device_connection value not recognised")

        return db_key, wet_mate_idx, dry_mate_idx

    def __str__(self):
        return (
            "This "
            + "network has: "
            + str(self.n_cp)
            + " collection point(s), "
            + str(len(self.array_cables))
            + " array cable(s) and "
            + str(len(self.export_cables))
            + " export cable(s)."
        )
