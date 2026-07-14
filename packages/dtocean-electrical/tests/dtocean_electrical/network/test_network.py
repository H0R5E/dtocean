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


def test_add_connector_wet(star_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    db_key, test_wet_mate_idx, test_dry_mate_idx = (
        star_null_network._add_connector(
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

    assert len(star_null_network.wet_mate) == 1
    wet_mate = star_null_network.wet_mate[0]

    assert wet_mate.id_ == wet_mate_idx
    assert wet_mate.db_key == wet_mate_key
    assert wet_mate.marker == marker
    assert wet_mate.utm_x == location[0]
    assert wet_mate.utm_y == location[1]


def test_add_connector_dry(star_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    db_key, test_wet_mate_idx, test_dry_mate_idx = (
        star_null_network._add_connector(
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

    assert len(star_null_network.dry_mate) == 1
    dry_mate = star_null_network.dry_mate[0]

    assert dry_mate.id_ == dry_mate_idx
    assert dry_mate.db_key == dry_mate_key
    assert dry_mate.marker == marker
    assert dry_mate.utm_x == location[0]
    assert dry_mate.utm_y == location[1]


def test_add_connector_bad(star_null_network: Network):
    marker = 0
    wet_mate_idx = 1
    dry_mate_idx = 2
    wet_mate_key = 10
    dry_mate_key = 20
    location = (0.0, 0.0)
    components = {"dry_connector": dry_mate_key, "wet_connector": wet_mate_key}

    with pytest.raises(ValueError) as exc:
        star_null_network._add_connector(
            "hi-mate",
            wet_mate_idx,
            dry_mate_idx,
            marker,
            location,
            components,
        )

    assert "connection_type value not recognised" in str(exc)


@pytest.fixture
def cluster() -> dict[str, Any]:
    return {"layout": []}


def test_Network_add_export_cable(
    grid: Grid,
    star_null_network: Network,
    cluster: dict[str, Any],
):
    marker = 4
    connection = 0
    export_idx = 1
    export_route = [36, 37]
    export_length = 2.0
    db_key = 3
    burial_depth = 10.0
    n_export_cables = len(star_null_network.export_cables)

    test_marker, test_export_idx = star_null_network._add_export_cable(
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

    assert len(star_null_network.export_cables) == n_export_cables + 1

    new_export = star_null_network.export_cables[-1]
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


@pytest.fixture
def star_null_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    network = NullNetwork()
    sub_cp_locs = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (-10.0, 0.0, 0.0)]
    sub_db_key = [11, 23, 23]

    network._init_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    return network


def test_Network_cp_to_cp(
    grid: Grid,
    star_null_network: Network,
    cluster: dict[str, Any],
):
    cluster["Export cable"] = [(-1, -1)]
    hierarchy: dict[str, Any] = {}
    marker = 1
    cp_idx = 0
    array_idx = 4
    wet_mate_idx = 2
    dry_mate_idx = 3
    cp_to_cp = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    subhub002_to_subhub003 = 13
    cp_cp_distance = np.array(
        [
            [0, subhub002_to_subhub003, 2],
            [subhub002_to_subhub003, 0, 1],
            [2, 0, 1],
        ]
    )
    cp_cp_paths = np.array(
        [
            [[], [0, 1], [0, 2]],
            [[0, 1], [], [1, 2]],
            [[0, 2], [1, 2], []],
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

    test_marker, test_array_idx, test_wet_mate_idx, test_dry_mate_idx = (
        star_null_network._cp_to_cp(
            cluster,
            hierarchy,
            cp_idx,
            marker,
            array_idx,
            wet_mate_idx,
            dry_mate_idx,
            cp_to_cp,
            cp_cp_distance,
            cp_cp_paths,
            components,
            grid.grid_pd,
            10,
        )
    )

    assert test_marker == marker + 8
    assert test_array_idx == array_idx + 2
    assert test_wet_mate_idx == wet_mate_idx + 2
    assert test_dry_mate_idx == dry_mate_idx + 2

    assert len(cluster["layout"]) == 1
    first = cluster["layout"][0]
    assert first == ["subhub002"]

    assert "subhub002" in hierarchy
    subhub002 = hierarchy["subhub002"]

    assert "Elec sub-system" in subhub002
    subhub002_elec = subhub002["Elec sub-system"]
    assert subhub002_elec == [
        (dry_mate_key, marker),
        (array_key, marker + 1),
        (wet_mate_key, marker + 2),
    ]

    assert "Substation" in subhub002
    subhub002_sub = subhub002["Substation"]
    assert subhub002_sub == [(23, marker + 3)]

    assert "layout" in subhub002
    subhub002_layout = subhub002["layout"]
    assert subhub002_layout == [["subhub003"]]

    assert "subhub003" in hierarchy
    subhub003 = hierarchy["subhub003"]

    assert "Elec sub-system" in subhub003
    subhub003_elec = subhub003["Elec sub-system"]
    assert subhub003_elec == [
        (dry_mate_key, marker + 4),
        (array_key, marker + 5),
        (wet_mate_key, marker + 6),
    ]

    assert "Substation" in subhub003
    subhub003_sub = subhub003["Substation"]
    assert subhub003_sub == [(23, marker + 7)]

    assert "layout" in subhub003
    subhub003_layout = subhub003["layout"]
    assert not subhub003_layout

    assert len(star_null_network.array_cables) == 2
    array_cable_device002 = star_null_network.array_cables[0]

    assert len(star_null_network.dry_mate) == 2
    dry_mate = star_null_network.dry_mate[0]

    assert dry_mate.id_ == dry_mate_idx
    assert dry_mate.db_key == dry_mate_key
    assert dry_mate.marker == marker
    assert dry_mate.utm_x == star_null_network.collection_points[0].location[0]
    assert dry_mate.utm_y == star_null_network.collection_points[0].location[1]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker + 1
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == subhub002_to_subhub003
    assert array_cable_device002.upstream_id == 0
    assert array_cable_device002.downstream_id == 1
    assert array_cable_device002.upstream_type == "collection point"
    assert array_cable_device002.downstream_type == "collection point"

    assert len(star_null_network.wet_mate) == 2
    wet_mate = star_null_network.wet_mate[0]

    assert wet_mate.id_ == wet_mate_idx
    assert wet_mate.db_key == wet_mate_key
    assert wet_mate.marker == marker + 2
    assert wet_mate.utm_x == star_null_network.collection_points[1].location[0]
    assert wet_mate.utm_y == star_null_network.collection_points[1].location[1]


def test_Network_add_device_fixed(star_null_network: Network):
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
    ) = star_null_network._add_device(
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

    assert len(star_null_network.array_cables) == 1
    array_cable_device002 = star_null_network.array_cables[0]

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

    assert len(star_null_network.wet_mate) == 1
    wet_mate_device002 = star_null_network.wet_mate[0]

    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert wet_mate_device002.utm_x == dev2_x
    assert wet_mate_device002.utm_y == dev2_y


def test_Network_add_device_floating(star_null_network: Network):
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
    ) = star_null_network._add_device(
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

    assert len(star_null_network.array_cables) == 1
    array_cable_device002 = star_null_network.array_cables[0]

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

    assert len(star_null_network.wet_mate) == 2
    wet_mate_device002 = star_null_network.wet_mate[0]

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

    assert len(star_null_network.umbilical_cables) == 1
    umbilical_device002 = star_null_network.umbilical_cables[0]

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
    star_null_network: Network,
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
    ) = star_null_network._device_to_device(
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

    assert len(star_null_network.array_cables) == 2
    array_cable_device002 = star_null_network.array_cables[0]

    assert isinstance(array_cable_device002, ArrayCable)
    assert array_cable_device002.id_ == array_idx
    assert array_cable_device002.marker == marker
    assert array_cable_device002.db_key == array_key
    assert array_cable_device002.length == dev002_to_dev003
    assert array_cable_device002.upstream_id == dev_idx + 1
    assert array_cable_device002.downstream_id == dev_idx
    assert array_cable_device002.upstream_type == "device"
    assert array_cable_device002.downstream_type == "device"

    assert len(star_null_network.wet_mate) == 2
    wet_mate_device002 = star_null_network.wet_mate[0]

    assert wet_mate_device002.id_ == wet_mate_idx
    assert wet_mate_device002.db_key == wet_mate_key
    assert wet_mate_device002.marker == marker + 1
    assert wet_mate_device002.utm_x == dev2_x
    assert wet_mate_device002.utm_y == dev2_y


def test_Network_device_to_device_floating(
    grid: Grid,
    star_null_network: Network,
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
    ) = star_null_network._device_to_device(
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

    assert len(star_null_network.array_cables) == 2
    array_cable_device002 = star_null_network.array_cables[0]

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

    assert len(star_null_network.wet_mate) == 4
    wet_mate_device002 = star_null_network.wet_mate[0]

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

    assert len(star_null_network.umbilical_cables) == 2
    umbilical_device002 = star_null_network.umbilical_cables[0]

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


def test_Network_cp_to_devices_substation(star_null_network: Network):
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
    ) = star_null_network._cp_to_devices(
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

    assert len(star_null_network.dry_mate) == 2

    for dry_mate in star_null_network.dry_mate:
        assert dry_mate.id_ in [dry_mate_idx, dry_mate_idx + 1]
        assert dry_mate.db_key == dry_mate_key
        assert (
            dry_mate.utm_x == star_null_network.collection_points[0].location[0]
        )
        assert (
            dry_mate.utm_y == star_null_network.collection_points[0].location[1]
        )


def test_Network_cp_to_devices_hub(hub_null_network: Network):
    cluster: dict[str, Any] = {"layout": []}
    subhub_key = "subhub001"
    hierarchy: dict[str, Any] = {subhub_key: {"layout": []}}
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def substation_null_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    """NullNetwork with a single substation CP (id=11, wet-mate in / dry-mate out)."""
    network = NullNetwork()
    network._init_collection_points(
        [(0.0, 0.0, 0.0)],
        [11],
        component_database.collection_points,
    )
    return network


@pytest.fixture
def burial_depths_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"id": list(range(10)), "Target burial depth": [1.0] * 10}
    )


@pytest.fixture
def simple_components() -> dict:
    return {
        "export": 1,  # static_cable id=1
        "array": 1,  # static_cable id=1
        "wet_connector": 5,  # wet_mate id=5
        "dry_connector": 7,  # dry_mate id=7
    }


# =============================================================================
# Network.__init__ validation
# =============================================================================


def _make_mock_py_power(
    shore_to_cp,
    cp_to_device,
    device_to_device,
    cp_to_cp=None,
):
    m = Mock()
    m.shore_to_cp = shore_to_cp
    m.cp_to_device = cp_to_device
    m.device_to_device = device_to_device
    m.cp_to_cp = cp_to_cp
    return m


def _minimal_network_kwargs(py_power, cp_locs):
    """Return keyword arguments for Network.__init__ with a mocked py_power."""
    return dict(
        index=0,
        floating=False,
        export_voltage=11000.0,
        array_power_output=[1.0],
        elec_array=Mock(),
        elec_db=Mock(),
        export_constraints=Mock(),
        array_constraints=Mock(),
        py_power=py_power,
        cp_locs=cp_locs,
        cp_db_keys=[Mock()],
        cp_db=Mock(),
        cp_device_distance=Mock(),
        cp_cp_distance=Mock(),
        cp_device_paths=Mock(),
        cp_cp_paths=Mock(),
        export_route=[0, 1],
        export_length=1000.0,
        components={},
        burial_depths=Mock(),
        burial_array=None,
        burial_export=None,
        grid=Mock(),
    )


def test_Network_init_shore_to_cp_length_mismatch():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1, 1]),  # length 2 ≠ len(cp_locs)=1
        cp_to_device=np.zeros((1, 1)),
        device_to_device=np.zeros((1, 1)),
    )
    with pytest.raises(
        ValueError, match="Length of shore_to_cp must equal n_cp"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


def test_Network_init_cp_to_device_shape_mismatch():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1]),
        cp_to_device=np.zeros((2, 1)),  # shape[0]=2 ≠ len(cp_locs)=1
        device_to_device=np.zeros((1, 1)),
    )
    with pytest.raises(
        ValueError, match="First dimension of cp_to_device must equal n_cp"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


def test_Network_init_device_to_device_not_square():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1]),
        cp_to_device=np.zeros((1, 2)),
        device_to_device=np.zeros((2, 3)),  # not square
    )
    with pytest.raises(
        ValueError, match="device_to_device must have equal dimensions"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


def test_Network_init_device_to_device_cp_to_device_mismatch():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1]),
        cp_to_device=np.zeros((1, 3)),  # shape[1]=3
        device_to_device=np.zeros((2, 2)),  # shape[0]=2 ≠ 3
    )
    with pytest.raises(
        ValueError, match="device_to_device must have equal dimensions"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


def test_Network_init_cp_to_cp_not_square():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1]),
        cp_to_device=np.zeros((1, 1)),
        device_to_device=np.zeros((1, 1)),
        cp_to_cp=np.zeros((1, 2)),  # not square
    )
    with pytest.raises(
        ValueError, match="If given, cp_to_cp must have equal dimensions"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


def test_Network_init_cp_to_cp_size_mismatch():
    cp_locs = [(0.0, 0.0, 0.0)]
    py_power = _make_mock_py_power(
        shore_to_cp=np.array([1]),
        cp_to_device=np.zeros((1, 1)),
        device_to_device=np.zeros((1, 1)),
        cp_to_cp=np.zeros((2, 2)),  # square but shape[0]=2 ≠ len(cp_locs)=1
    )
    with pytest.raises(
        ValueError, match="If given, cp_to_cp must have equal dimensions"
    ):
        Network(**_minimal_network_kwargs(py_power, cp_locs))


# =============================================================================
# Network._cp_to_devices — empty cp_to_device (line 781)
# =============================================================================


def test_Network_cp_to_devices_empty(star_null_network: Network):
    cluster: dict[str, Any] = {"layout": []}
    hierarchy: dict[str, Any] = {}
    marker = 5
    array_idx = 1
    wet_mate_idx = 2
    dry_mate_idx = 3
    umbilical_idx = 4
    cp_to_device = np.zeros((0, 0))  # size == 0

    result = star_null_network._cp_to_devices(
        False,
        cluster,
        hierarchy,
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
        "wet-mate",
        {},
        cp_to_device,
        np.zeros((0, 0)),
        np.zeros((0, 0)),
        np.array([], dtype=object).reshape(0, 0),
        {},
        pd.DataFrame({"id": [], "Target burial depth": []}),
        None,
    )

    assert result == (
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
    )


# =============================================================================
# Network._cp_to_cp — empty cp_to_cp (line 676)
# =============================================================================


def test_Network_cp_to_cp_empty(
    star_null_network: Network,
    cluster: dict[str, Any],
):
    cluster["Export cable"] = [(-1, -1)]
    hierarchy: dict[str, Any] = {}
    marker = 3
    cp_idx = 0
    array_idx = 1
    wet_mate_idx = 2
    dry_mate_idx = 4

    result = star_null_network._cp_to_cp(
        cluster,
        hierarchy,
        cp_idx,
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        np.zeros((0, 0)),  # size == 0  → early return
        np.zeros((0, 0)),
        np.array([], dtype=object).reshape(0, 0),
        {},
        pd.DataFrame({"id": [], "Target burial depth": []}),
        None,
    )

    assert result == (marker, array_idx, wet_mate_idx, dry_mate_idx)


# =============================================================================
# Network._add_cps
# =============================================================================


def test_Network_add_cps(
    substation_null_network: Network,
    burial_depths_df: pd.DataFrame,
    simple_components: dict,
):
    """Happy-path: _add_cps adds the import connector, updates cluster and sets
    the CP marker."""
    cluster: dict[str, Any] = {"Export cable": [(-1, -1)], "layout": []}
    hierarchy: dict[str, Any] = {}
    cp_idx = 0
    marker = 5
    array_idx = 1
    wet_mate_idx = 2
    dry_mate_idx = 3

    test_marker, test_array_idx, test_wet_mate_idx, test_dry_mate_idx = (
        substation_null_network._add_cps(
            cluster,
            hierarchy,
            cp_idx,
            marker,
            array_idx,
            wet_mate_idx,
            dry_mate_idx,
            np.zeros((0, 0)),  # no CP-to-CP
            np.zeros((0, 0)),
            np.array([], dtype=object).reshape(0, 0),
            simple_components,
            burial_depths_df,
            1.0,
        )
    )

    # CP id=11 has input_connector="wet-mate"
    assert test_wet_mate_idx == wet_mate_idx + 1
    assert test_dry_mate_idx == dry_mate_idx
    # marker advances by 2: one for connector, one for CP
    assert test_marker == marker + 2

    # Second entry added to Export cable (connector on import side of CP)
    assert len(cluster["Export cable"]) == 2
    assert cluster["Export cable"][1] == (
        simple_components["wet_connector"],
        marker,
    )

    cp = substation_null_network.collection_points[0]
    assert cp.marker == marker + 1
    assert cluster["Substation"] == [(cp.db_key, cp.marker)]


def test_Network_add_cps_missing_export_cable(
    substation_null_network: Network,
    burial_depths_df: pd.DataFrame,
    simple_components: dict,
):
    """RuntimeError is raised when 'Export cable' is absent from cluster."""
    cluster: dict[str, Any] = {"layout": []}  # no "Export cable"
    hierarchy: dict[str, Any] = {}

    with pytest.raises(
        RuntimeError, match="Export cable must exist in cluster"
    ):
        substation_null_network._add_cps(
            cluster,
            hierarchy,
            0,
            0,
            0,
            0,
            0,
            np.zeros((0, 0)),
            np.zeros((0, 0)),
            np.array([], dtype=object).reshape(0, 0),
            simple_components,
            burial_depths_df,
            1.0,
        )


# =============================================================================
# Network._add_device — error paths
# =============================================================================


def test_Network_add_device_floating_missing_umbilical_data(
    star_null_network: Network,
):
    """ValueError when floating=True but umbilical_data is None."""
    with pytest.raises(
        ValueError, match="umbilical_data must be set if 'floating' is True"
    ):
        star_null_network._add_device(
            True,
            [],
            0,
            0,
            0,
            0,
            0,
            0,
            10.0,
            [0, 1],
            [1.0, 1.0],
            "device",
            0,
            "wet-mate",
            {"Device001": (0.0, 0.0)},
            {"array": 1, "wet_connector": 5},
            None,  # umbilical_data is None → raises immediately
        )


def test_Network_add_device_floating_device_missing_from_umbilical(
    star_null_network: Network,
):
    """ValueError when floating=True and device key absent from umbilical_data."""
    umbilical_data = {
        "Device999": {
            "device": "Device999",
            "length": 50.0,
            "x coords": [0.0],
            "z coords": [0.0],
            "termination": (0.0, 0.0, -10.0),
            "db_key": 1,
        }
    }
    with pytest.raises(
        ValueError, match="Umbilical data not defined for device Device001"
    ):
        star_null_network._add_device(
            True,
            [],
            0,
            0,  # dev_idx=0 → "Device001"
            0,
            0,
            0,
            0,
            10.0,
            [0, 1],
            [1.0, 1.0],
            "device",
            0,
            "wet-mate",
            {"Device001": (0.0, 0.0)},
            {"array": 1, "wet_connector": 5},
            umbilical_data,
        )


def test_Network_add_device_fixed_device_missing_from_layout(
    star_null_network: Network,
):
    """ValueError when floating=False and device key absent from device_layout."""
    with pytest.raises(
        ValueError, match="Layout not defined for device Device001"
    ):
        star_null_network._add_device(
            False,
            [],
            0,
            0,  # dev_idx=0 → "Device001"
            0,
            0,
            0,
            0,
            10.0,
            [0, 1],
            [1.0, 1.0],
            "device",
            0,
            "wet-mate",
            {"Device999": (0.0, 0.0)},  # Device001 absent
            {"array": 1, "wet_connector": 5},
        )


# =============================================================================
# Network._device_to_device — all nodes already visited (line 921)
# =============================================================================


def test_Network_device_to_device_all_visited(star_null_network: Network):
    """When every next-device candidate is in visited_nodes the loop exits early."""
    hierarchy: dict[str, Any] = {}
    layout: list[str] = []
    # Mark device 1 as already visited so when dev_idx=0 tries to reach it,
    # it finds nothing new and returns immediately.
    visited_nodes = [1, 2]
    dev_idx = 0
    marker = 3
    array_idx = 0
    wet_mate_idx = 0
    dry_mate_idx = 0
    umbilical_idx = 0
    device_to_device = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    cp_device_distance = np.zeros((4, 4))
    cp_device_paths = np.array(
        [
            [[], [], [], []],
            [[], [], [], []],
            [[], [], [], []],
            [[], [], [], []],
        ],
        dtype=object,
    )
    components = {"array": 1, "wet_connector": 5}

    result = star_null_network._device_to_device(
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
        "wet-mate",
        {"Device001": (0.0, 0.0), "Device002": (1.0, 0.0)},
        device_to_device,
        cp_device_distance,
        cp_device_paths,
        components,
        pd.DataFrame({"id": list(range(4)), "Target burial depth": [1.0] * 4}),
        1.0,
        None,
    )

    assert result == (
        marker,
        array_idx,
        wet_mate_idx,
        dry_mate_idx,
        umbilical_idx,
    )
    assert layout == []  # nothing was added
    assert hierarchy == {}  # nothing was added


# =============================================================================
# Network._get_all_connections integration (lines 424-497)
# =============================================================================


def test_Network_get_all_connections(
    substation_null_network: Network,
    burial_depths_df: pd.DataFrame,
    simple_components: dict,
):
    """Integration test covering _get_all_connections and _add_cps body."""
    shore_to_cp = np.array([1])
    cp_to_cp = np.zeros((0, 0))
    cp_to_device = np.array([[1]])  # CP 0 → device 0
    device_to_device = np.zeros((1, 1))
    cp_device_distance = np.array([[0, 200], [200, 0]], dtype=float)
    cp_cp_distance = np.zeros((0, 0))
    cp_device_paths = np.array([[[], [0, 1]], [[1, 0], []]], dtype=object)
    cp_cp_paths = np.array([], dtype=object).reshape(0, 0)
    device_layout = {"Device001": (500.0, 600.0)}

    result = substation_null_network._get_all_connections(
        floating=False,
        shore_to_cp=shore_to_cp,
        cp_to_cp=cp_to_cp,
        cp_to_device=cp_to_device,
        device_to_device=device_to_device,
        cp_device_distance=cp_device_distance,
        cp_cp_distance=cp_cp_distance,
        device_connection="wet-mate",
        device_layout=device_layout,
        cp_device_paths=cp_device_paths,
        cp_cp_paths=cp_cp_paths,
        export_route=[0, 1],
        export_length=1000.0,
        components=simple_components,
        burial_depths=burial_depths_df,
        burial_array=1.0,
        burial_export=1.0,
        umbilical_data=None,
    )

    assert "array" in result
    assert len(result["array"]) == 1

    cluster = result["array"][0]
    assert "Export cable" in cluster
    assert "Substation" in cluster
    assert cluster["Substation"] == [(11, 2)]

    assert "device001" in result
    elec = result["device001"]["Elec sub-system"]
    assert (
        len(elec) == 3
    )  # dry-mate(CP out) + array cable + wet-mate(device in)

    # Cables and connectors were created
    assert len(substation_null_network.export_cables) == 1
    assert len(substation_null_network.array_cables) == 1
    assert len(substation_null_network.wet_mate) == 2  # export-side + device
    assert len(substation_null_network.dry_mate) == 1  # array-side of CP


# =============================================================================
# Properties (lines 283-331)
# =============================================================================


def test_Network_properties(null_network: Network):
    """Properties return stored private attributes."""
    null_network._n_devices = 3
    null_network._all_connections = {"array": [], "device001": {}}
    null_network._bom = pd.DataFrame()
    null_network._economics_data = pd.DataFrame()
    null_network._total_cost = 42.0
    null_network._cable_routes = pd.DataFrame()

    assert null_network.n_cp == 0
    assert null_network.n_devices == 3
    assert null_network.all_connections is null_network._all_connections
    assert null_network.bom is null_network._bom
    assert null_network.economics_data is null_network._economics_data
    assert null_network.total_cost == 42.0
    assert null_network.cable_routes is null_network._cable_routes


# =============================================================================
# Network._get_bom (lines 1292-1361)
# =============================================================================


def test_Network_get_bom(star_null_network: Network):
    """_get_bom produces a DataFrame with one row per component."""
    # Add one of each component type
    wet = WetMateConnector(0, 5, 10, (1.0, 2.0))
    dry = DryMateConnector(0, 7, 11, (3.0, 4.0))
    export = ExportCable(
        0,
        500.0,
        1,
        12,
        [0, 1],
        [1.0, 1.0],
        [False, False],
        "collection point",
        0,
    )
    array = ArrayCable(
        0,
        300.0,
        1,
        13,
        [0, 1],
        [1.0, 1.0],
        [False, False],
        "device",
        "collection point",
        0,
        0,
    )
    umbilical = UmbilicalCable(
        0, 50.0, 2, 14, (5.0, 6.0, -10.0), "Device001", [0.0, 1.0], [0.0, -10.0]
    )

    star_null_network.wet_mate = [wet]
    star_null_network.dry_mate = [dry]
    star_null_network.export_cables = [export]
    star_null_network.array_cables = [array]
    star_null_network.umbilical_cables = [umbilical]
    # collection_points already set (3 CPs from fixture)

    bom = star_null_network._get_bom()

    assert bom is not None
    assert (
        len(bom) == 1 + 1 + 1 + 1 + 1 + 3
    )  # wet + dry + export + array + umbilical + 3CPs
    assert set(bom.columns) == {
        "marker",
        "db ref",
        "install_type",
        "utm_x",
        "utm_y",
        "quantity",
    }
    assert (bom[bom["install_type"] == "export"]["quantity"] == 500.0).all()
    assert (bom[bom["install_type"] == "array"]["quantity"] == 300.0).all()
    assert (bom[bom["install_type"] == "umbilical"]["quantity"] == 50.0).all()


# =============================================================================
# Network._get_db_keys_from_pd, _get_economics_data, _get_total_cost
# (lines 1374-1375, 1400-1430, 1483-1489, 1498)
# =============================================================================


def test_Network_get_db_keys_from_pd(null_network: Network):
    null_network._bom = pd.DataFrame({"db ref": [1, 5, 1, 7]})
    result = null_network._get_db_keys_from_pd()
    assert set(result) == {1, 5, 7}


def test_Network_get_economics_data(
    component_database: ElectricalComponentDatabase,
    star_null_network: Network,
):
    """_get_economics_data looks up unit costs from the database."""
    # Build a BOM with one wet-mate connector (id=5) at quantity 1
    wet = WetMateConnector(0, 5, 10, (1.0, 2.0))
    star_null_network.wet_mate = [wet]
    star_null_network.dry_mate = []
    star_null_network.export_cables = []
    star_null_network.array_cables = []
    star_null_network.umbilical_cables = []

    # Remove the pre-set CPs so only the connector appears
    star_null_network.collection_points = []

    star_null_network._bom = star_null_network._get_bom()
    economics = star_null_network._get_economics_data(component_database)

    assert economics is not None
    assert len(economics) == 1
    assert economics["db ref"].iloc[0] == 5
    assert economics["cost"].iloc[0] == 150000  # from mock_db wet_mate id=5


def test_Network_get_economics_data_with_onshore_cost(
    component_database: ElectricalComponentDatabase,
    star_null_network: Network,
):
    """Onshore cost appended as a None-keyed row."""
    wet = WetMateConnector(0, 5, 10, (1.0, 2.0))
    star_null_network.wet_mate = [wet]
    star_null_network.dry_mate = []
    star_null_network.export_cables = []
    star_null_network.array_cables = []
    star_null_network.umbilical_cables = []
    star_null_network.collection_points = []

    star_null_network._bom = star_null_network._get_bom()
    economics = star_null_network._get_economics_data(
        component_database, onshore_cost=999.0
    )

    assert len(economics) == 2
    onshore_row = economics[economics["db ref"].isna()]
    assert onshore_row["cost"].iloc[0] == 999.0
    assert onshore_row["quantity"].iloc[0] == 1


def test_Network_get_total_cost(null_network: Network):
    null_network._economics_data = pd.DataFrame(
        {
            "cost": [100.0, 200.0],
            "quantity": [3, 2],
        }
    )
    total = null_network._get_total_cost()
    assert total == 700.0


# =============================================================================
# Network._get_hierarchy (lines 1134-1182)
# =============================================================================


def test_Network_get_hierarchy(null_network: Network):
    """_get_hierarchy builds the connection hierarchy dict."""
    export_key = 1
    wet_key = 5
    dry_key = 7
    cp_key = 11

    null_network._n_devices = 1
    null_network._all_connections = {
        "array": [
            {
                "Export cable": [(export_key, 0), (wet_key, 1)],
                "Substation": [(cp_key, 2)],
                "layout": [["device001"]],
            }
        ],
        "device001": {
            "Elec sub-system": [(dry_key, 3), (export_key, 4), (wet_key, 5)],
        },
    }

    hier = null_network._get_hierarchy()

    assert "device001" in hier
    dev001 = hier["device001"]
    assert dev001["Elec sub-system"] == [[dry_key, export_key, wet_key]]

    assert "array" in hier
    arr = hier["array"]
    assert arr["layout"] == [["device001"]]
    assert arr["Export cable"] == [[export_key, wet_key]]
    assert arr["Substation"] == [[cp_key]]


def test_Network_get_hierarchy_with_subhub(null_network: Network):
    """Sub-hub entries are included in hierarchy."""
    null_network._n_devices = 0
    null_network._all_connections = {
        "array": [
            {
                "Export cable": [(1, 0), (5, 1)],
                "Substation": [(11, 2)],
                "layout": [["subhub001"]],
            }
        ],
        "subhub001": {
            "Elec sub-system": [(7, 3), (1, 4), (5, 5)],
            "Substation": [(23, 6)],
            "layout": [["device001"]],
        },
    }

    hier = null_network._get_hierarchy()

    assert "subhub001" in hier
    subhub = hier["subhub001"]
    assert subhub["layout"] == [["device001"]]
    assert subhub["Substation"] == [[23]]
    assert subhub["Elec sub-system"] == [[7, 1, 5]]


# =============================================================================
# Network._get_network_design (lines 1187-1258)
# =============================================================================


def test_Network_get_network_design(null_network: Network):
    """_get_network_design builds the marker-based design dict."""
    export_key = 1
    wet_key = 5
    dry_key = 7
    cp_key = 11

    null_network._n_devices = 1
    null_network._all_connections = {
        "array": [
            {
                "Export cable": [(export_key, 0), (wet_key, 1)],
                "Substation": [(cp_key, 2)],
                "layout": [["device001"]],
            }
        ],
        "device001": {
            "Elec sub-system": [(dry_key, 3), (export_key, 4), (wet_key, 5)],
        },
    }

    design = null_network._get_network_design()

    assert "device001" in design
    dev001 = design["device001"]
    assert dev001["marker"] == [[3, 4, 5]]
    from collections import Counter

    assert dev001["quantity"] == Counter(
        {dry_key: 1, export_key: 1, wet_key: 1}
    )

    assert "array" in design
    arr = design["array"]
    assert arr["Export cable"]["marker"] == [[0, 1]]
    assert arr["Substation"]["marker"] == [[2]]


def test_Network_get_network_design_with_subhub(null_network: Network):
    """Sub-hub entries appear in network design."""
    null_network._n_devices = 0
    null_network._all_connections = {
        "array": [
            {
                "Export cable": [(1, 0), (5, 1)],
                "Substation": [(11, 2)],
                "layout": [],
            }
        ],
        "subhub001": {
            "Elec sub-system": [(7, 3), (1, 4)],
            "Substation": [(23, 5)],
            "layout": [],
        },
    }

    design = null_network._get_network_design()

    assert "subhub001" in design
    sub = design["subhub001"]
    assert sub["marker"] == [[3, 4, 5]]
    from collections import Counter

    assert sub["quantity"] == Counter({7: 1, 1: 1, 23: 1})


# =============================================================================
# Network._get_collection_point_design (lines 1619-1678)
# =============================================================================


def test_Network_get_collection_point_design(
    star_null_network: Network,
):
    """_get_collection_point_design returns a DataFrame with one row per CP."""
    # Set markers so the output is non-trivial
    for i, cp in enumerate(star_null_network.collection_points):
        cp.marker = i * 10

    df = star_null_network._get_collection_point_design()

    assert df is not None
    assert len(df) == 3  # star_null_network has 3 CPs
    assert "marker" in df.columns
    assert "origin" in df.columns
    assert list(df["marker"]) == [0, 10, 20]


# =============================================================================
# Network._get_umbilical_cable_design — non-empty (lines 1734-1766)
# =============================================================================


def test_Network_get_umbilical_cable_design_non_empty(null_network: Network):
    """When umbilical_cables is populated the design DataFrame is returned."""
    cable = UmbilicalCable(
        0,
        75.0,
        3,
        5,
        (100.0, 200.0, -30.0),
        "Device001",
        [0.0, 1.0],
        [0.0, -30.0],
    )
    null_network.umbilical_cables = [cable]

    df = null_network._get_umbilical_cable_design()

    assert df is not None
    assert len(df) == 1
    assert df["marker"].iloc[0] == 5
    assert df["db ref"].iloc[0] == 3
    assert df["device"].iloc[0] == "Device001"
    assert df["length"].iloc[0] == 75.0
    seabed = df["seabed_connection_point"].iloc[0]
    assert list(seabed) == [100.0, 200.0, -30.0]


# =============================================================================
# Network.calculate_annual_losses (lines 1783-1789)
# =============================================================================


def test_Network_calculate_annual_losses():
    network = NullNetwork()
    network.power_histogram = [0.5, 0.5]
    network.array_power_output = [2.0, 1.0]

    ideal_yield = 2 * 8760 * 1e6  # 2 MW × 100% occurrence × hours × W/MW

    losses, efficiency = network.calculate_annual_losses(ideal_yield)

    annual_yield = network._calculate_annual_yield()
    assert losses == pytest.approx(ideal_yield - annual_yield)
    assert efficiency == pytest.approx(annual_yield / ideal_yield)


def test_Network_calculate_annual_losses_none_yield():
    """When annual_yield is None, losses equal ideal and efficiency is 1."""
    network = NullNetwork()
    # Override annual_yield property to return None
    network._calculate_annual_yield = lambda: None  # type: ignore[assignment]
    ideal_yield = 1000.0

    losses, efficiency = network.calculate_annual_losses(ideal_yield)

    assert losses == ideal_yield
    assert efficiency == 1


# =============================================================================
# Network.calculate_histogram_losses (lines 1810-1819)
# =============================================================================


def test_Network_calculate_histogram_losses():
    network = NullNetwork()
    network.array_power_output = [2.0, 1.0]

    ideal_histogram = [2_000_000.0, 1_000_000.0]  # in W
    losses, efficiency = network.calculate_histogram_losses(ideal_histogram)

    assert losses == [0.0, 0.0]
    assert efficiency == [1.0, 1.0]


def test_Network_calculate_histogram_losses_with_loss():
    network = NullNetwork()
    network.array_power_output = [1.5, 0.8]  # MW

    ideal_histogram = [2_000_000.0, 1_000_000.0]  # W
    losses, efficiency = network.calculate_histogram_losses(ideal_histogram)

    assert losses[0] == pytest.approx(2_000_000.0 - 1.5e6)
    assert efficiency[0] == pytest.approx(1.5e6 / 2_000_000.0)


# =============================================================================
# Network.print_result, log_result, _make_result_str (lines 1822-1846)
# =============================================================================


def test_Network_print_result(null_network: Network, capsys):
    null_network._make_result_str = MagicMock(return_value="TEST OUTPUT")
    null_network.print_result()
    captured = capsys.readouterr()
    assert "TEST OUTPUT" in captured.out


def test_Network_log_result(null_network: Network):
    null_network._make_result_str = MagicMock(return_value="LOG OUTPUT")
    null_network.log_result()
    null_network._make_result_str.assert_called_once()


def test_Network_make_result_str(null_network: Network):
    """_make_result_str produces a non-empty string using all output properties."""
    null_network._n_devices = 0
    null_network._all_connections = {
        "array": [
            {
                "Export cable": [(1, 0)],
                "Substation": [(11, 1)],
                "layout": [],
            }
        ]
    }
    null_network._bom = pd.DataFrame(
        columns=[
            "marker",
            "db ref",
            "install_type",
            "utm_x",
            "utm_y",
            "quantity",
        ]
    )
    null_network._economics_data = pd.DataFrame(
        columns=["db ref", "quantity", "cost", "year"]
    )
    null_network._total_cost = 0.0
    null_network._cable_routes = pd.DataFrame()
    null_network.power_histogram = [1.0]
    null_network.array_power_output = [0.0]

    result = null_network._make_result_str()

    assert isinstance(result, str)
    assert "Annual yield" in result
    assert "Bill of Materials" in result
    assert "Hierarchy" in result


# =============================================================================
# Network.__str__ (line 1849)
# =============================================================================


def test_Network_str(null_network: Network):
    result = str(null_network)
    assert "network" in result.lower()
    assert "0 collection point" in result
    assert "0 array cable" in result
    assert "0 export cable" in result
