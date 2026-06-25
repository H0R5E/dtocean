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

import numpy as np
import pandas as pd
import pytest

from dtocean_electrical.grid.grid import Grid
from dtocean_electrical.inputs import ElectricalComponentDatabase
from dtocean_electrical.network.cable import ArrayCable, ExportCable
from dtocean_electrical.network.collection_point import PassiveHub, Substation
from dtocean_electrical.network.network import Network
from dtocean_electrical.optimiser.power_flow import ComponentLoading


@pytest.fixture
def mock_network() -> Network:
    return Network(
        0,
        [],
        [],
        False,
        ComponentLoading("mock", 0),
        ComponentLoading("mock", 0),
    )


def test_Network_add_collection_points_substation(
    component_database: ElectricalComponentDatabase,
    mock_network: Network,
):
    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = 11

    mock_network.add_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    assert len(mock_network.collection_points) == 1
    test = mock_network.collection_points[0]

    assert isinstance(test, Substation)
    assert test.location == sub_cp_locs[0]
    assert test.db_key == sub_db_key

    passive_cp_locs = [(1.0, 1.0, 1.0)]
    passive_db_key = 23

    mock_network.add_collection_points(
        passive_cp_locs,
        passive_db_key,
        component_database.collection_points,
    )

    assert len(mock_network.collection_points) == 2
    test = mock_network.collection_points[1]

    assert isinstance(test, PassiveHub)
    assert test.location == passive_cp_locs[0]
    assert test.db_key == passive_db_key


def test_Network_add_collection_points_empty(
    component_database: ElectricalComponentDatabase,
    mock_network: Network,
):
    cp_locs = [(0.0, 0.0, 0.0)]
    db_key = -1

    with pytest.raises(ValueError) as exc:
        mock_network.add_collection_points(
            cp_locs,
            db_key,
            component_database.collection_points,
        )

    assert "db_key not found in db" in str(exc)


@pytest.fixture
def hub_radial_fixed_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    network = Network(
        0,
        [],
        [],
        False,
        ComponentLoading("mock_export", 0),
        ComponentLoading("mock_array", 0),
    )

    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = 23

    network.add_collection_points(
        sub_cp_locs,
        sub_db_key,
        component_database.collection_points,
    )

    return network


@pytest.fixture
def substation_radial_fixed_network(
    component_database: ElectricalComponentDatabase,
) -> Network:
    network = Network(
        0,
        [],
        [],
        False,
        ComponentLoading("mock_export", 0),
        ComponentLoading("mock_array", 0),
    )

    sub_cp_locs = [(0.0, 0.0, 0.0)]
    sub_db_key = 11

    network.add_collection_points(
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
    substation_radial_fixed_network: Network,
    cluster: dict[str, Any],
):
    marker = 4
    connection = 0
    export_idx = 1
    export_route = [36, 37]
    export_length = 2.0
    db_key = 3
    burial_depth = 10.0
    n_export_cables = len(substation_radial_fixed_network.export_cables)

    test_marker, test_export_idx = (
        substation_radial_fixed_network._add_export_cable(
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
    )

    assert test_marker == marker + 1
    assert test_export_idx == export_idx + 1

    assert "Export cable" in cluster
    assert cluster["Export cable"] == [(db_key, marker)]

    assert (
        len(substation_radial_fixed_network.export_cables)
        == n_export_cables + 1
    )

    new_export = substation_radial_fixed_network.export_cables[-1]
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


def test_Network_add_substation_passive(
    hub_radial_fixed_network: Network,
    cluster: dict[str, Any],
):
    hierarchy: dict[str, Any] = {}
    marker = 1
    cp_idx = 0
    wet_mate_idx = 2
    dry_mate_idx = 3
    components = {"wet_connector": 4}
    subhub_key = f"subhub{str(cp_idx).zfill(3)}"
    cp = hub_radial_fixed_network.collection_points[cp_idx]

    test_marker, test_wet_mate_idx, test_dry_mate_idx = (
        hub_radial_fixed_network._add_substation(
            cluster,
            hierarchy,
            marker,
            cp_idx,
            wet_mate_idx,
            dry_mate_idx,
            components,
        )
    )

    assert test_marker == marker + 1
    assert test_wet_mate_idx == wet_mate_idx
    assert test_dry_mate_idx == dry_mate_idx

    assert cluster["layout"] == [subhub_key]
    assert "Substation" in cluster
    assert cluster["Substation"] == ["Ideal"]

    assert subhub_key in hierarchy
    subhub_hier = hierarchy[subhub_key]

    assert "Elec sub-system" in subhub_hier
    assert not subhub_hier["Elec sub-system"]

    assert "Substation" in subhub_hier
    assert subhub_hier["Substation"] == [(cp.db_key, marker)]
    assert cp.marker == marker


def test_Network_add_substation_active(
    substation_radial_fixed_network: Network,
    cluster: dict[str, Any],
):
    cluster["Export cable"] = [(-1, -1)]
    hierarchy: dict[str, Any] = {}
    marker = 1
    cp_idx = 0
    wet_mate_idx = 2
    dry_mate_idx = 3
    components = {"wet_connector": 4}
    cp = substation_radial_fixed_network.collection_points[cp_idx]

    test_marker, test_wet_mate_idx, test_dry_mate_idx = (
        substation_radial_fixed_network._add_substation(
            cluster,
            hierarchy,
            marker,
            cp_idx,
            wet_mate_idx,
            dry_mate_idx,
            components,
        )
    )

    assert test_marker == marker + 2
    assert test_wet_mate_idx == wet_mate_idx + 1
    assert test_dry_mate_idx == dry_mate_idx

    assert len(cluster["Export cable"]) == 2
    connector = cluster["Export cable"][1]
    assert connector == (4, marker)

    assert "Substation" in cluster
    assert cluster["Substation"] == [(cp.db_key, marker + 1)]

    assert not hierarchy
    assert cp.marker == marker + 1

    assert len(substation_radial_fixed_network.wet_mate) == 1
    wet_mate = substation_radial_fixed_network.wet_mate[0]

    assert wet_mate.id_ == wet_mate_idx
    assert wet_mate.db_key == 4
    assert wet_mate.marker == marker
    assert wet_mate.utm_x == cp.location[0]
    assert wet_mate.utm_y == cp.location[1]


def test_Network_make_cable_routes(mock_network: Network):
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

    grid = pd.DataFrame(grid_dict)
    all_x = grid.x.to_list()
    all_y = grid.y.to_list()

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
        0, 20.0, 0, 1, [8, 5, 2], [0.1] * 3, [True] * 3, "collection point", 0
    )

    mock_network.array_cables = [array_cable]
    mock_network.export_cables = [export_cable]

    mock_network.make_cable_routes(grid, all_x, all_y)

    assert mock_network.cable_routes is not None
    assert len(mock_network.cable_routes) == 6

    markers = mock_network.cable_routes.marker
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

    network = Network(
        0,
        power_histogram,
        array_power_output,
        False,
        ComponentLoading("mock", 0),
        ComponentLoading("mock", 0),
    )
    annual_yield = network.calculate_annual_yield()

    assert annual_yield == 8760000000.0 + 8760000000.0 / 4


def test_Network_calculate_annual_yield_zero():
    power_histogram = [0.5, 0.5]
    array_power_output = [2, np.nan]

    network = Network(
        0,
        power_histogram,
        array_power_output,
        False,
        ComponentLoading("mock", 0),
        ComponentLoading("mock", 0),
    )

    annual_yield = network.calculate_annual_yield()

    assert annual_yield == 0.0


def test_Network_calculate_lcoe(mock_network: Network):
    mock_network.total_cost = 10
    mock_network.annual_yield = 2
    mock_network.calculate_lcoe()

    assert mock_network.lcoe == 5e3


def test_Network_calculate_lcoe_inf(mock_network: Network):
    mock_network.total_cost = 10
    mock_network.annual_yield = 0
    mock_network.calculate_lcoe()

    assert mock_network.lcoe == np.inf
