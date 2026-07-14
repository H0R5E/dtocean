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

from dtocean_electrical.optimiser.optimiser import RadialNetwork, StarNetwork


def test_RadialNetwork_db_compatibility(mocker):
    # Fake a database
    df_dict = {
        "id": [1, 2],
        "v_rate": [6600, 11000],
        "v1": [11000, 6600],
        "v2": [6600, 11000],
    }

    df = pd.DataFrame(df_dict)

    database = mocker.Mock()
    database.array_cable = df
    database.export_cable = df
    database.collection_points = df
    database.wet_mate_connectors = df
    database.dry_mate_connectors = df
    database.dynamic_cable = df

    # Fake the meta
    meta = mocker.Mock()
    meta.array_data.machine_data.technology = "floating"
    meta.options.user_umbilical = None

    # Options
    oec_voltage = 6600
    export_voltage = 11000
    array_voltage = 6600

    test = RadialNetwork(meta)

    result = test.db_compatibility(
        database,
        oec_voltage,
        export_voltage,
        array_voltage,
        1,
    )

    assert result["wet_connector"] == 1
    assert result["dry_connector"] == 1
    assert result["array"][0] == 1
    assert result["cp"] == [2]
    assert result["export"][0] == 2
    assert result["umbilical"] == 1


def test_StarNetwork_db_compatibility(mocker):
    # Fake a database
    df_dict = {
        "id": [1, 2],
        "v_rate": [6600, 11000],
        "v1": [6600, 6600],
        "v2": [6600, 11000],
        "a_air": [50, 50],
    }

    df = pd.DataFrame(df_dict)

    database = mocker.Mock()
    database.array_cable = df
    database.export_cable = df
    database.collection_points = df
    database.wet_mate_connectors = df
    database.dry_mate_connectors = df
    database.dynamic_cable = df

    # Fake the meta
    meta = mocker.Mock()
    meta.array_data.machine_data.technology = "fixed"
    meta.array_data.machine_data.max_current = 20

    # Options
    oec_voltage = 6600
    export_voltage = 11000
    array_voltage = 6600

    test = StarNetwork(meta)

    result = test.db_compatibility(
        database,
        oec_voltage,
        export_voltage,
        array_voltage,
        2,
    )

    assert result["wet_connector"] == 1
    assert result["dry_connector"] == 1
    assert result["array"][0] == 1
    assert result["cp"] == [2, 1]
    assert result["export"][0] == 2
    assert result["device"] == 1


def test_RadialNetwork_cable_impedance(mocker):
    # Fake a database
    df_dict = {"id": [1], "r_ac": [1], "xl": [2], "c": [3]}

    df = pd.DataFrame(df_dict)

    database = mocker.Mock()
    database.array_cable = df

    # Fake the meta
    meta = mocker.Mock()
    meta.database = database

    test = RadialNetwork(meta)
    result = test.cable_impedance(1, "array")

    assert result == (1, 2, 3)


def test_RadialNetwork_cable_rating(mocker):
    # Fake a database
    df_dict = {"id": [1], "a_air": [1]}

    df = pd.DataFrame(df_dict)

    database = mocker.Mock()
    database.export_cable = df

    # Fake the meta
    meta = mocker.Mock()
    meta.database = database

    test = RadialNetwork(meta)
    result = test.cable_rating(1, "export")

    assert result == 1


def test_RadialNetworkr__get_cable_db_error(mocker):
    # Fake the meta
    meta = mocker.Mock()
    test = RadialNetwork(meta)

    with pytest.raises(KeyError):
        test._get_cable_db("wrong")


def test_RadialNetworkr_check_lcoe(mocker):
    # Fake the meta
    meta = mocker.Mock()
    test = RadialNetwork(meta)

    # Fake the results
    test.lcoe = [2, 1]
    networks = [mocker.Mock(), mocker.Mock()]

    for i, network in enumerate(networks):
        network.i = i
        network.array_constraints.flag = False
        network.export_constraints.flag = False
        network.lcoe = test.lcoe[i]

    test.networks = networks

    result = test.check_lcoe()

    assert result == 1


def test_RadialNetwork_check_lcoe_error(mocker):
    # Fake the meta
    meta = mocker.Mock()
    test = RadialNetwork(meta)

    # Fake the results
    test.lcoe = [np.inf, np.inf]
    networks = [mocker.Mock(), mocker.Mock()]

    for i, network in enumerate(networks):
        network.i = i
        network.array_constraints.flag = False
        network.export_constraints.flag = False
        network.lcoe = test.lcoe[i]

    test.networks = networks

    with pytest.raises(ValueError):
        test.check_lcoe()
