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

import numpy as np
import pandas as pd
import pytest

from dtocean_electrical.inputs import ElectricalComponentDatabase
from dtocean_electrical.network.cable import ArrayCable, ExportCable
from dtocean_electrical.network.collection_point import PassiveHub, Substation
from dtocean_electrical.network.network import Network
from dtocean_electrical.optimiser.power_flow import ComponentLoading


@pytest.fixture
def mock_network() -> Network:
    return Network(
        0,
        "mock",
        [],
        [],
        False,
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
        "mock",
        power_histogram,
        array_power_output,
        False,
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
        "mock",
        power_histogram,
        array_power_output,
        False,
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
