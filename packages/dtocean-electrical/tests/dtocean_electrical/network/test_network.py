# -*- coding: utf-8 -*-

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

from typing import Any
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from dtocean_electrical.grid.grid import Grid
from dtocean_electrical.inputs import ElectricalComponentDatabase
from dtocean_electrical.network.cable import (
    ArrayCable,
    ExportCable,
    UmbilicalCable,
)
from dtocean_electrical.network.collection_point import (
    CollectionPoint,
    PassiveHub,
    Substation,
)
from dtocean_electrical.network.connector import (
    DryMateConnector,
    WetMateConnector,
)
from dtocean_electrical.network.network import Network


class NullNetwork(Network):
    def __init__(self):
        self.export_cables: list[ExportCable] = []
        self.array_cables: list[ArrayCable] = []
        self.umbilical_cables: list[UmbilicalCable] = []
        self.collection_points: list[CollectionPoint] = []
        self.wet_mate: list[WetMateConnector] = []
        self.dry_mate: list[DryMateConnector] = []


@pytest.fixture
def null_network() -> Network:
    return NullNetwork()


def test_Network_add_collection_points_substation(
    component_database: ElectricalComponentDatabase,
    null_network: Network,
):
    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = [11]

    null_network._init_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    assert len(null_network.collection_points) == 1
    test = null_network.collection_points[0]

    assert isinstance(test, Substation)
    assert test.location == sub_cp_locs[0]
    assert test.db_key == sub_db_key[0]

    passive_cp_locs = [(1.0, 1.0, 1.0)]
    passive_db_key = [23]

    null_network._init_collection_points(
        passive_cp_locs,
        passive_db_key,
        component_database.collection_points,
    )

    assert len(null_network.collection_points) == 2
    test = null_network.collection_points[1]

    assert isinstance(test, PassiveHub)
    assert test.location == passive_cp_locs[0]
    assert test.db_key == passive_db_key[0]


def test_Network_add_collection_points_empty(
    component_database: ElectricalComponentDatabase,
    null_network: Network,
):
    cp_locs = [(0.0, 0.0, 0.0)]
    db_key = [-1]

    with pytest.raises(ValueError) as exc:
        null_network._init_collection_points(
            cp_locs,
            db_key,
            component_database.collection_points,
        )

    assert "db_key not found in db" in str(exc)


@pytest.fixture
def hub_null_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    network = NullNetwork()
    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = [23]

    network._init_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    return network


@pytest.fixture
def substation_null_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    network = NullNetwork()
    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = [11]

    network._init_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    return network


@pytest.fixture
def cluster() -> dict[str, Any]:
    return {"layout": []}


def test_Network_add_export_cable(
    grid: Grid,
    substation_null_network: Network,
    cluster: dict[str, Any],
):
    marker = 4
    connection = 0
    export_idx = 1
    export_route = [36, 37]
    export_length = 2.0
    db_key = 3
    burial_depth = 10.0
    n_export_cables = len(substation_null_network.export_cables)

    test_marker, test_export_idx = substation_null_network._add_export_cable(
        cluster,
        marker,
        connection,
        export_idx,
        export_route,
        export_length,
        db_key,
        grid.grid_pd,
        burial_depth,
    )

    assert test_marker == marker + 1
    assert test_export_idx == export_idx + 1

    assert "Export cable" in cluster
    assert cluster["Export cable"] == [(db_key, marker)]

    assert len(substation_null_network.export_cables) == n_export_cables + 1

    new_export = substation_null_network.export_cables[-1]
    assert isinstance(new_export, ExportCable)
    assert new_export.id_ == export_idx
    assert new_export.db_key == db_key
    assert new_export.length == export_length
    assert new_export.marker == marker
    assert new_export.route == export_route
    assert new_export.split_pipe == [burial_depth < 0] * len(export_route)
    assert new_export.target_burial_depth == [burial_depth] * len(export_route)
    assert new_export.upstream_type == "collection point"
    assert new_export.upstream_id == connection


# def test_Network_add_substation_passive(
#     hub_null_network: Network,
#     cluster: dict[str, Any],
# ):
#     hierarchy: dict[str, Any] = {}
#     marker = 1
#     cp_idx = 0
#     wet_mate_idx = 2
#     dry_mate_idx = 3
#     components = {"wet_connector": 4}
#     subhub_key = f"subhub{str(cp_idx).zfill(3)}"
#     cp = hub_null_network.collection_points[cp_idx]

#     test_marker, test_wet_mate_idx, test_dry_mate_idx = (
#         hub_null_network._add_cps(
#             cluster,
#             hierarchy,
#             marker,
#             cp_idx,
#             wet_mate_idx,
#             dry_mate_idx,
#             components,
#         )
#     )

#     assert test_marker == marker + 1
#     assert test_wet_mate_idx == wet_mate_idx
#     assert test_dry_mate_idx == dry_mate_idx

#     assert cluster["layout"] == [subhub_key]
#     assert "Substation" in cluster
#     assert cluster["Substation"] == ["Ideal"]

#     assert subhub_key in hierarchy
#     subhub_hier = hierarchy[subhub_key]

#     assert "Elec sub-system" in subhub_hier
#     assert not subhub_hier["Elec sub-system"]

#     assert "Substation" in subhub_hier
#     assert subhub_hier["Substation"] == [(cp.db_key, marker)]
#     assert cp.marker == marker


# def test_Network_add_substation_active(
#     substation_null_network: Network,
#     cluster: dict[str, Any],
# ):
#     cluster["Export cable"] = [(-1, -1)]
#     hierarchy: dict[str, Any] = {}
#     marker = 1
#     cp_idx = 0
#     wet_mate_idx = 2
#     dry_mate_idx = 3
#     components = {"wet_connector": 4}
#     cp = substation_null_network.collection_points[cp_idx]

#     test_marker, test_wet_mate_idx, test_dry_mate_idx = (
#         substation_null_network._add_cps(
#             cluster,
#             hierarchy,
#             marker,
#             cp_idx,
#             wet_mate_idx,
#             dry_mate_idx,
#             components,
#         )
#     )

#     assert test_marker == marker + 2
#     assert test_wet_mate_idx == wet_mate_idx + 1
#     assert test_dry_mate_idx == dry_mate_idx

#     assert len(cluster["Export cable"]) == 2
#     connector = cluster["Export cable"][1]
#     assert connector == (4, marker)

#     assert "Substation" in cluster
#     assert cluster["Substation"] == [(cp.db_key, marker + 1)]

#     assert not hierarchy
#     assert cp.marker == marker + 1

#     assert len(substation_null_network.wet_mate) == 1
#     wet_mate = substation_null_network.wet_mate[0]

#     assert wet_mate.id_ == wet_mate_idx
#     assert wet_mate.db_key == 4
#     assert wet_mate.marker == marker
#     assert wet_mate.utm_x == cp.location[0]
#     assert wet_mate.utm_y == cp.location[1]


def test_add_connector_wet(substation_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    db_key, test_wet_mate_idx, test_dry_mate_idx = (
        substation_null_network._add_connector(
            "wet-mate",
            wet_mate_idx,
            dry_mate_idx,
            marker,
            location,
            components,
        )
    )

    assert db_key == wet_mate_key
    assert test_wet_mate_idx == wet_mate_idx + 1
    assert test_dry_mate_idx == dry_mate_idx

    assert len(substation_null_network.wet_mate) == 1
    wet_mate = substation_null_network.wet_mate[0]

    assert wet_mate.id_ == wet_mate_idx
    assert wet_mate.db_key == wet_mate_key
    assert wet_mate.marker == marker
    assert wet_mate.utm_x == location[0]
    assert wet_mate.utm_y == location[1]


def test_add_connector_dry(substation_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    db_key, test_wet_mate_idx, test_dry_mate_idx = (
        substation_null_network._add_connector(
            "dry-mate",
            wet_mate_idx,
            dry_mate_idx,
            marker,
            location,
            components,
        )
    )

    assert db_key == dry_mate_key
    assert test_wet_mate_idx == wet_mate_idx
    assert test_dry_mate_idx == dry_mate_idx + 1

    assert len(substation_null_network.dry_mate) == 1
    dry_mate = substation_null_network.dry_mate[0]

    assert dry_mate.id_ == dry_mate_idx
    assert dry_mate.db_key == dry_mate_key
    assert dry_mate.marker == marker
    assert dry_mate.utm_x == location[0]
    assert dry_mate.utm_y == location[1]


def test_add_connector_bad(substation_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    with pytest.raises(ValueError) as exc:
        substation_null_network._add_connector(
            "hi-mate",
            wet_mate_idx,
            dry_mate_idx,
            marker,
            location,
            components,
        )

    assert "connection_type value not recognised" in str(exc)


def test_Network_add_device_fixed(substation_null_network: Network):
    elec_sub_system = []
    marker = 0
    dev_idx = 1
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    array_cable_length = 6.0
    array_route = [0, 1, 2, 3]
    burial = [10.0, 5.0, 0.0, 0.0]
    array_downstream_type = "mock_down"
    array_downstream_id = 7
    device_connection = "wet-mate"
    dev2_x = 24.0
    dev2_y = 1354.0
    device_layout = {"Device002": (dev2_x, dev2_y), "Device003": (2.0, 2.0)}
    array_key = 8
    wet_mate_key = 9
    components = {"array": array_key, "wet_connector": wet_mate_key}

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = substation_null_network._add_device(
        False,
        elec_sub_system,
        marker,
        dev_idx,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
        array_cable_length,
        array_route,
        burial,
        array_downstream_type,
        array_downstream_id,
        device_connection,
        device_layout,
        components,
    )

    assert test_marker == marker + 2
    assert test_array_idx == array_idx + 1
    assert test_wet_mate_idx == wet_mate_idx + 1
    assert test_dry_mate_idx == dry_mate_idx
    assert test_umbilical_idx == umbilical_idx

    assert elec_sub_system == [(array_key, marker), (wet_mate_key, marker + 1)]

    assert len(substation_null_network.array_cables) == 1
    array_cable_device002 = substation_null_network.array_cables[0]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == array_cable_length
    assert array_cable_device002.upstream_id == dev_idx
    assert array_cable_device002.upstream_type == "device"
    assert array_cable_device002.downstream_id == array_downstream_id
    assert array_cable_device002.downstream_type == array_downstream_type
    assert array_cable_device002.route == array_route
    assert array_cable_device002.split_pipe == [not bool(d) for d in burial]
    assert array_cable_device002.target_burial_depth == burial

    assert len(substation_null_network.wet_mate) == 1
    wet_mate_device002 = substation_null_network.wet_mate[0]

    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert wet_mate_device002.utm_x == dev2_x
    assert wet_mate_device002.utm_y == dev2_y


def test_Network_add_device_floating(substation_null_network: Network):
    elec_sub_system = []
    marker = 0
    dev_idx = 1
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    array_cable_length = 6.0
    array_route = [0, 1, 2, 3]
    burial = [10.0, 5.0, 0.0, 0.0]
    array_downstream_type = "mock_down"
    array_downstream_id = 7
    device_connection = "wet-mate"
    dev2_x = 24.0
    dev2_y = 1354.0
    device_layout = {"Device002": (dev2_x, dev2_y), "Device003": (2.0, 2.0)}
    array_key = 8
    wet_mate_key = 9
    components = {"array": array_key, "wet_connector": wet_mate_key}
    umbilical_db_key = 8
    umbilical_length = 50.0
    umbilical_design = {
        "Device002": {
            "device": "Device002",
            "length": umbilical_length,
            "x coords": [0.0, 1.0, 2.0],
            "z coords": [0.0, 10.0, 20.0],
            "termination": (dev2_x * 2, dev2_y, -30.0),
            "db_key": umbilical_db_key,
        },
        "Device003": {
            "device": "Device003",
            "length": umbilical_length,
            "x coords": [0.0, 1.0, 2.0],
            "z coords": [0.0, 10.0, 20.0],
            "termination": (1.0, 1.0, -30.0),
            "db_key": umbilical_db_key,
        },
    }

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = substation_null_network._add_device(
        True,
        elec_sub_system,
        marker,
        dev_idx,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
        array_cable_length,
        array_route,
        burial,
        array_downstream_type,
        array_downstream_id,
        device_connection,
        device_layout,
        components,
        umbilical_design,
    )

    assert test_marker == marker + 4
    assert test_array_idx == array_idx + 1
    assert test_wet_mate_idx == wet_mate_idx + 2
    assert test_dry_mate_idx == dry_mate_idx
    assert test_umbilical_idx == umbilical_idx + 1

    assert elec_sub_system == [
        (array_key, marker),
        (wet_mate_key, marker + 1),
        (umbilical_db_key, marker + 2),
        (wet_mate_key, marker + 3),
    ]

    assert len(substation_null_network.array_cables) == 1
    array_cable_device002 = substation_null_network.array_cables[0]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == array_cable_length
    assert array_cable_device002.upstream_id == marker + 1
    assert array_cable_device002.upstream_type == "connector"
    assert array_cable_device002.downstream_id == array_downstream_id
    assert array_cable_device002.downstream_type == array_downstream_type
    assert array_cable_device002.route == array_route
    assert array_cable_device002.split_pipe == [not bool(d) for d in burial]
    assert array_cable_device002.target_burial_depth == burial

    assert len(substation_null_network.wet_mate) == 2
    wet_mate_device002 = substation_null_network.wet_mate[0]

    assert isinstance(wet_mate_device002, WetMateConnector)
    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert (
        wet_mate_device002.utm_x
        == umbilical_design["Device002"]["termination"][0]
    )
    assert (
        wet_mate_device002.utm_y
        == umbilical_design["Device002"]["termination"][1]
    )

    assert len(substation_null_network.umbilical_cables) == 1
    umbilical_device002 = substation_null_network.umbilical_cables[0]

    assert isinstance(umbilical_device002, UmbilicalCable)
    assert umbilical_device002.id_ == umbilical_idx
    assert umbilical_device002.marker == marker + 2
    assert umbilical_device002.db_key == umbilical_db_key
    assert umbilical_device002.length == umbilical_length
    assert umbilical_device002.upstream_id is None
    assert umbilical_device002.downstream_id is None
    assert umbilical_device002.upstream_type is None
    assert umbilical_device002.downstream_type is None
    assert (
        umbilical_device002.seabed_termination_x
        == umbilical_design["Device002"]["termination"][0]
    )
    assert (
        umbilical_device002.seabed_termination_y
        == umbilical_design["Device002"]["termination"][1]
    )
    assert (
        umbilical_device002.seabed_termination_z
        == umbilical_design["Device002"]["termination"][2]
    )
    assert umbilical_device002.device == "Device002"
    assert (
        umbilical_device002.x_coordinates
        == umbilical_design["Device002"]["x coords"]
    )
    assert (
        umbilical_device002.z_coordinates
        == umbilical_design["Device002"]["z coords"]
    )


def test_Network_device_to_device_fixed(
    grid: Grid,
    substation_null_network: Network,
):
    hierarchy: dict[str, Any] = {}
    layout = []
    visited_nodes = []
    dev_idx = 0
    marker = 1
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    device_connection = "wet-mate"
    dev2_x = 24.0
    dev2_y = 1354.0
    device_layout = {"Device002": (dev2_x, dev2_y), "Device003": (2.0, 2.0)}
    device_to_device = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dev002_to_dev003 = 13
    cp_device_distance = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, dev002_to_dev003, 1],
            [2, dev002_to_dev003, 0, 1],
            [3, 1, 1, 0],
        ]
    )
    cp_device_paths = np.array(
        [
            [[], [0, 1], [0, 2], [0, 2, 3]],
            [[0, 1], [], [1, 2], [1, 3]],
            [[0, 2], [0, 2, 3], [], [2, 3]],
            [[0, 2, 3], [1, 3], [2, 3], []],
        ],
        dtype="object",
    )
    array_key = 6
    wet_mate_key = 7
    components = {"array": array_key, "wet_connector": wet_mate_key}

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = substation_null_network._device_to_device(
        False,
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
        grid.grid_pd,
        10,
        None,
    )

    assert test_marker == marker + 4
    assert test_array_idx == array_idx + 2
    assert test_wet_mate_idx == wet_mate_idx + 2
    assert test_dry_mate_idx == dry_mate_idx
    assert test_umbilical_idx == umbilical_idx

    assert layout == ["device002", "device003"]

    assert "device002" in hierarchy
    device002 = hierarchy["device002"]

    assert "Elec sub-system" in device002
    device002_elec = device002["Elec sub-system"]
    assert device002_elec == [(array_key, marker), (wet_mate_key, marker + 1)]

    assert "device003" in hierarchy
    device003 = hierarchy["device003"]

    assert "Elec sub-system" in device003
    device003_elec = device003["Elec sub-system"]
    assert device003_elec == [
        (array_key, marker + 2),
        (wet_mate_key, marker + 3),
    ]

    assert len(substation_null_network.array_cables) == 2
    array_cable_device002 = substation_null_network.array_cables[0]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == dev002_to_dev003
    assert array_cable_device002.upstream_id == dev_idx + 1
    assert array_cable_device002.downstream_id == dev_idx
    assert array_cable_device002.upstream_type == "device"
    assert array_cable_device002.downstream_type == "device"

    assert len(substation_null_network.wet_mate) == 2
    wet_mate_device002 = substation_null_network.wet_mate[0]

    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert wet_mate_device002.utm_x == dev2_x
    assert wet_mate_device002.utm_y == dev2_y


def test_Network_device_to_device_floating(
    grid: Grid,
    substation_null_network: Network,
):
    hierarchy: dict[str, Any] = {}
    layout = []
    visited_nodes = []
    dev_idx = 0
    marker = 10
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    device_connection = "wet-mate"
    dev2_x = 24.0
    dev2_y = 1354.0
    device_layout = {"Device002": (dev2_x, dev2_y), "Device003": (2.0, 2.0)}
    device_to_device = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    dev002_to_dev003 = 13
    cp_device_distance = np.array(
        [
            [0, 1, 2, 3],
            [1, 0, dev002_to_dev003, 1],
            [2, dev002_to_dev003, 0, 1],
            [3, 1, 1, 0],
        ]
    )
    cp_device_paths = np.array(
        [
            [[], [0, 1], [0, 2], [0, 2, 3]],
            [[0, 1], [], [1, 2], [1, 3]],
            [[0, 2], [0, 2, 3], [], [2, 3]],
            [[0, 2, 3], [1, 3], [2, 3], []],
        ],
        dtype="object",
    )
    array_key = 6
    wet_mate_key = 7
    components = {"array": array_key, "wet_connector": wet_mate_key}
    umbilical_db_key = 8
    umbilical_length = 50.0
    umbilical_design = {
        "Device002": {
            "device": "Device002",
            "length": umbilical_length,
            "x coords": [0.0, 1.0, 2.0],
            "z coords": [0.0, 10.0, 20.0],
            "termination": (dev2_x * 2, dev2_y, -30.0),
            "db_key": umbilical_db_key,
        },
        "Device003": {
            "device": "Device003",
            "length": umbilical_length,
            "x coords": [0.0, 1.0, 2.0],
            "z coords": [0.0, 10.0, 20.0],
            "termination": (1.0, 1.0, -30.0),
            "db_key": umbilical_db_key,
        },
    }

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = substation_null_network._device_to_device(
        True,
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
        grid.grid_pd,
        10,
        umbilical_design,
    )

    assert test_marker == marker + 8
    assert test_array_idx == array_idx + 2
    assert test_wet_mate_idx == wet_mate_idx + 4
    assert test_dry_mate_idx == dry_mate_idx
    assert test_umbilical_idx == umbilical_idx + 2

    assert layout == ["device002", "device003"]

    assert "device002" in hierarchy
    device002 = hierarchy["device002"]

    assert "Elec sub-system" in device002
    device002_elec = device002["Elec sub-system"]
    assert device002_elec == [
        (array_key, marker),
        (wet_mate_key, marker + 1),
        (umbilical_db_key, marker + 2),
        (wet_mate_key, marker + 3),
    ]

    assert "device003" in hierarchy
    device003 = hierarchy["device003"]

    assert "Elec sub-system" in device003
    device003_elec = device003["Elec sub-system"]
    assert device003_elec == [
        (array_key, marker + 4),
        (wet_mate_key, marker + 5),
        (umbilical_db_key, marker + 6),
        (wet_mate_key, marker + 7),
    ]

    assert len(substation_null_network.array_cables) == 2
    array_cable_device002 = substation_null_network.array_cables[0]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == dev002_to_dev003
    assert array_cable_device002.upstream_id == marker + 1
    assert (
        array_cable_device002.downstream_id == marker - 3
    )  # previous array to umbilical connector
    assert array_cable_device002.upstream_type == "connector"
    assert array_cable_device002.downstream_type == "connector"

    assert len(substation_null_network.wet_mate) == 4
    wet_mate_device002 = substation_null_network.wet_mate[0]

    assert isinstance(wet_mate_device002, WetMateConnector)
    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert (
        wet_mate_device002.utm_x
        == umbilical_design["Device002"]["termination"][0]
    )
    assert (
        wet_mate_device002.utm_y
        == umbilical_design["Device002"]["termination"][1]
    )

    assert len(substation_null_network.umbilical_cables) == 2
    umbilical_device002 = substation_null_network.umbilical_cables[0]

    assert isinstance(umbilical_device002, UmbilicalCable)
    assert umbilical_device002.id_ == umbilical_idx
    assert umbilical_device002.marker == marker + 2
    assert umbilical_device002.db_key == umbilical_db_key
    assert umbilical_device002.length == umbilical_length
    assert umbilical_device002.upstream_id is None
    assert umbilical_device002.downstream_id is None
    assert umbilical_device002.upstream_type is None
    assert umbilical_device002.downstream_type is None
    assert (
        umbilical_device002.seabed_termination_x
        == umbilical_design["Device002"]["termination"][0]
    )
    assert (
        umbilical_device002.seabed_termination_y
        == umbilical_design["Device002"]["termination"][1]
    )
    assert (
        umbilical_device002.seabed_termination_z
        == umbilical_design["Device002"]["termination"][2]
    )
    assert umbilical_device002.device == "Device002"
    assert (
        umbilical_device002.x_coordinates
        == umbilical_design["Device002"]["x coords"]
    )
    assert (
        umbilical_device002.z_coordinates
        == umbilical_design["Device002"]["z coords"]
    )


def test_Network_cp_to_devices_substation(substation_null_network: Network):
    cluster: dict[str, Any] = {"layout": []}
    hierarchy: dict[str, Any] = {}
    marker = 1
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    device_connection = "wet-mate"
    dev1_x = 24.0
    dev1_y = 1354.0
    device_layout = {
        "Device001": (dev1_x, dev1_y),
        "Device002": (2000.0, 2000.0),
        "Device003": (100.0, 100.0),
    }
    cp_to_device = np.array([[1, 0, 1]])
    device_to_device = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    cp_to_dev001 = 1300
    cp_device_distance = np.array(
        [
            [0, cp_to_dev001, 2, 3],
            [cp_to_dev001, 0, 1000, 1],
            [2, 1000, 0, 1],
            [3, 1, 1, 0],
        ]
    )
    cp_device_paths = np.array(
        [
            [[], [0, 1], [0, 2], [0, 2, 3]],
            [[0, 1], [], [1, 2], [1, 3]],
            [[0, 2], [0, 2, 3], [], [2, 3]],
            [[0, 2, 3], [1, 3], [2, 3], []],
        ],
        dtype="object",
    )
    array_key = 6
    wet_mate_key = 7
    dry_mate_key = 8
    components = {
        "array": array_key,
        "wet_connector": wet_mate_key,
        "dry_connector": dry_mate_key,
    }

    burial_targets_dict = {
        "id": [0, 1, 2, 3],
        "Target burial depth": [10, 10, 10, 10],
    }
    burial_targets = pd.DataFrame(burial_targets_dict)
    burial_array = 10

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = substation_null_network._cp_to_devices(
        False,
        cluster,
        hierarchy,
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
        device_connection,
        device_layout,
        cp_to_device,
        device_to_device,
        cp_device_distance,
        cp_device_paths,
        components,
        burial_targets,
        burial_array,
    )

    assert test_marker == marker + 8
    assert test_array_idx == array_idx + 3
    assert test_wet_mate_idx == wet_mate_idx + 3
    assert test_dry_mate_idx == dry_mate_idx + 2
    assert test_umbilical_idx == umbilical_idx

    assert cluster["layout"] == [["device001", "device002"], ["device003"]]

    for dev_idx in range(3):
        assert f"device00{dev_idx + 1}" in hierarchy
        dev_hier = hierarchy[f"device00{dev_idx + 1}"]
        assert "Elec sub-system" in dev_hier

    device001_elec = hierarchy["device001"]["Elec sub-system"]
    assert device001_elec == [
        (dry_mate_key, marker),
        (array_key, marker + 1),
        (wet_mate_key, marker + 2),
    ]

    device002_elec = hierarchy["device002"]["Elec sub-system"]
    assert len(device002_elec) == 2

    device003_elec = hierarchy["device003"]["Elec sub-system"]
    assert len(device003_elec) == 3

    assert len(substation_null_network.dry_mate) == 2

    for dry_mate in substation_null_network.dry_mate:
        assert dry_mate.id_ in [dry_mate_idx, dry_mate_idx + 1]
        assert dry_mate.db_key == dry_mate_key
        assert (
            dry_mate.utm_x
            == substation_null_network.collection_points[0].location[0]
        )
        assert (
            dry_mate.utm_y
            == substation_null_network.collection_points[0].location[1]
        )


def test_Network_cp_to_devices_hub(hub_null_network: Network):
    cluster: dict[str, Any] = {"layout": []}
    subhub_key = "subhub001"
    hierarchy: dict[str, Any] = {subhub_key: {}}
    marker = 1
    array_idx = 2
    wet_mate_idx = 3
    dry_mate_idx = 4
    umbilical_idx = 5
    device_connection = "wet-mate"
    dev1_x = 24.0
    dev1_y = 1354.0
    device_layout = {
        "Device001": (dev1_x, dev1_y),
        "Device002": (2000.0, 2000.0),
        "Device003": (100.0, 100.0),
    }
    cp_to_device = np.array([[1, 0, 1]])
    device_to_device = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    cp_to_dev001 = 1300
    cp_device_distance = np.array(
        [
            [0, cp_to_dev001, 2, 3],
            [cp_to_dev001, 0, 1000, 1],
            [2, 1000, 0, 1],
            [3, 1, 1, 0],
        ]
    )
    cp_device_paths = np.array(
        [
            [[], [0, 1], [0, 2], [0, 2, 3]],
            [[0, 1], [], [1, 2], [1, 3]],
            [[0, 2], [0, 2, 3], [], [2, 3]],
            [[0, 2, 3], [1, 3], [2, 3], []],
        ],
        dtype="object",
    )
    array_key = 6
    wet_mate_key = 7
    dry_mate_key = 8
    components = {
        "array": array_key,
        "wet_connector": wet_mate_key,
        "dry_connector": dry_mate_key,
    }

    burial_targets_dict = {
        "id": [0, 1, 2, 3],
        "Target burial depth": [10, 10, 10, 10],
    }
    burial_targets = pd.DataFrame(burial_targets_dict)
    burial_array = 10

    (
        test_marker,
        test_array_idx,
        test_wet_mate_idx,
        test_dry_mate_idx,
        test_umbilical_idx,
    ) = hub_null_network._cp_to_devices(
        False,
        cluster,
        hierarchy,
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
        device_connection,
        device_layout,
        cp_to_device,
        device_to_device,
        cp_device_distance,
        cp_device_paths,
        components,
        burial_targets,
        burial_array,
    )

    assert test_marker == marker + 8
    assert test_array_idx == array_idx + 3
    assert test_wet_mate_idx == wet_mate_idx + 3
    assert test_dry_mate_idx == dry_mate_idx + 2
    assert test_umbilical_idx == umbilical_idx

    assert not cluster["layout"]
    assert "layout" in hierarchy[subhub_key]
    assert hierarchy[subhub_key]["layout"] == [
        ["device001", "device002"],
        ["device003"],
    ]

    for dev_idx in range(3):
        assert f"device00{dev_idx + 1}" in hierarchy
        dev_hier = hierarchy[f"device00{dev_idx + 1}"]
        assert "Elec sub-system" in dev_hier

    device001_elec = hierarchy["device001"]["Elec sub-system"]
    assert device001_elec == [
        (dry_mate_key, marker),
        (array_key, marker + 1),
        (wet_mate_key, marker + 2),
    ]

    device002_elec = hierarchy["device002"]["Elec sub-system"]
    assert len(device002_elec) == 2

    device003_elec = hierarchy["device003"]["Elec sub-system"]
    assert len(device003_elec) == 3

    assert len(hub_null_network.dry_mate) == 2

    for dry_mate in hub_null_network.dry_mate:
        assert dry_mate.id_ in [dry_mate_idx, dry_mate_idx + 1]
        assert dry_mate.db_key == dry_mate_key
        assert (
            dry_mate.utm_x == hub_null_network.collection_points[0].location[0]
        )
        assert (
            dry_mate.utm_y == hub_null_network.collection_points[0].location[1]
        )


def test_Network_get_cable_routes(null_network: Network):
    grid_dict = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "x": [0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20, 0, 10, 20],
        "y": [
            0,
            0,
            0,
            10,
            10,
            10,
            20,
            20,
            20,
            30,
            30,
            30,
            40,
            40,
            40,
            50,
            50,
            50,
        ],
        "layer 1 start": [
            0,
            0,
            0,
            10,
            10,
            10,
            20,
            20,
            20,
            30,
            30,
            30,
            40,
            40,
            40,
            50,
            50,
            50,
        ],
        "layer 1 type": [
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
            "hard rock",
        ],
    }

    grid = Mock()
    grid.grid_pd = pd.DataFrame(grid_dict)
    grid.all_x = grid.grid_pd.x
    grid.all_y = grid.grid_pd.y

    array_cable = ArrayCable(
        0,
        20.0,
        0,
        0,
        [14, 11, 8],
        [0.1] * 3,
        [True] * 3,
        "device",
        "collection point",
        0,
        0,
    )

    export_cable = ExportCable(
        0,
        20.0,
        0,
        1,
        [8, 5, 2],
        [0.1] * 3,
        [True] * 3,
        "collection point",
        0,
    )

    null_network.array_cables = [array_cable]
    null_network.export_cables = [export_cable]

    cable_routes = null_network._get_cable_routes(grid)

    assert cable_routes is not None
    assert len(cable_routes) == 6

    markers = cable_routes.marker
    assert all(x <= y for x, y in zip(markers, markers[1:]))


def test_Network__map_component_types():
    types = [
        "export",
        "array",
        "wet-mate",
        "dry-mate",
        "substation",
        "passive hub",
        "umbilical",
    ]

    result = Network._map_component_types(types)

    assert result == [
        "export_cable",
        "array_cable",
        "wet_mate_connectors",
        "dry_mate_connectors",
        "collection_points",
        "collection_points",
        "dynamic_cable",
    ]


def test_Network_calculate_annual_yield():
    power_histogram = [0.5, 0.5]
    array_power_output = [2, 0.5]

    network = NullNetwork()
    network.power_histogram = power_histogram
    network.array_power_output = array_power_output

    annual_yield = network._calculate_annual_yield()

    assert annual_yield == 8760000000.0 + 8760000000.0 / 4


def test_Network_calculate_annual_yield_zero():
    power_histogram = [0.5, 0.5]
    array_power_output = [2, np.nan]

    network = NullNetwork()
    network.power_histogram = power_histogram
    network.array_power_output = array_power_output

    annual_yield = network._calculate_annual_yield()

    assert annual_yield == 0.0


def test_Network_get_lcoe(null_network: Network):
    null_network._total_cost = 10
    null_network._calculate_annual_yield = MagicMock(return_value=2)

    assert null_network._get_lcoe() == 5e3


def test_Network_get_lcoe_inf(null_network: Network):
    null_network._total_cost = 10
    null_network._calculate_annual_yield = MagicMock(return_value=0)

    assert null_network._get_lcoe() == np.inf
