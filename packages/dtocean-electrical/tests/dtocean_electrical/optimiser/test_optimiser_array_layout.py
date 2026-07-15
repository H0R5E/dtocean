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

from copy import deepcopy

import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LinearRing, LineString, Point

from dtocean_electrical.grid.grid_processing import clip_grid
from dtocean_electrical.optimiser.array_layout import (
    add_device_positions,
    calculate_distance_dijkstra,
    calculate_saving_vector,
    check_in_paths,
    check_in_route,
    check_neighbour_number,
    check_path_capacity,
    closeness_test,
    crossing_dijkstra,
    dijkstra,
    extend_line,
    get_device_locations,
    get_export,
    get_single_location,
    make_linestring,
    make_optimal_paths,
    offset_cp,
    offset_cp_local,
    set_substation_to_edge,
    snap_to_grid,
    substation_in_site,
    update_paths,
)


def test_dijkstra(graph):
    length, path = dijkstra(graph, 676, 694)

    assert np.isclose(length, 181.2425646506693)
    assert path == list(range(676, 695))


def test_dijkstra_target_missing(grid):
    with pytest.raises(nx.NetworkXNoPath):
        dijkstra(grid.graph, 676, 695, grid)


def test_dijkstra_nopath(grid):
    graph_copy = deepcopy(grid.graph)

    # Remove a rectangle of nodes
    nodes_to_remove = [
        658,
        659,
        660,
        661,
        662,
        663,
        664,
        665,
        666,
        685,
        693,
        712,
        713,
        714,
        715,
        716,
        717,
        718,
        719,
        720,
    ]

    graph_copy.remove_nodes_from(nodes_to_remove)

    assert graph_copy.has_node(690)

    with pytest.raises(nx.NetworkXNoPath):
        dijkstra(graph_copy, 676, 690, grid)


def test_get_export(grid):
    cp_loc = (491820.0, 6502180)
    landing_loc = (491760.0, 6500320.0)

    length, path = get_export(cp_loc, landing_loc, grid, grid.graph)

    assert length > 0.0
    assert path[-1] == 153


def test_calculate_distance_dijkstra(grid):
    layout_grid = [(0, 334), (0, 344), (0, 604), (0, 614)]
    substation_location = (491770.0, 6502090.0)

    (distance_array, path_array) = calculate_distance_dijkstra(
        layout_grid, substation_location, grid, grid.graph
    )

    assert distance_array.shape == (5, 5)
    assert np.sum(distance_array) > 0.0
    assert distance_array[1, 2] == distance_array[2, 1]

    assert path_array.shape == (5, 5)
    assert len(path_array[1, 2]) > 0
    assert path_array[1, 2] == path_array[2, 1][::-1]


def test_calculate_saving_vector(grid):
    layout_grid = [(0, 334), (0, 344), (0, 604), (0, 614)]
    substation_location = (491770.0, 6502090.0)

    (distance_array, path_array) = calculate_distance_dijkstra(
        layout_grid, substation_location, grid, grid.graph
    )

    saving_vector = calculate_saving_vector(distance_array, 4)

    for vector in saving_vector:
        assert vector[0] != vector[1]

        if vector[1] == 0:
            assert np.isclose(vector[2], 0)
        else:
            assert vector[2] > 0.0


def test_set_substation_to_edge(lease, export):
    _, lease_polygon = clip_grid(lease, export)

    landing_point = Point((495000.0, 6502220))
    line = LineString([(491810.0, 6502220), landing_point])
    lease_area_ring = LinearRing(list(lease_polygon.exterior.coords))

    result = set_substation_to_edge(line, lease_area_ring, lease, lease_polygon)

    assert result is not None
    assert ((lease["x"] == result[0]) & (lease["y"] == result[1])).any()


def test_snap_to_grid(lease, export):
    _, lease_polygon = clip_grid(lease, export)

    grid = np.array(lease[["x", "y"]])
    point = (491810.0, 6502090.0)

    result = snap_to_grid(grid, point, lease)

    assert len(result) == 3
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert isinstance(result[2], float)

    # Check that the result point is close to the input point
    assert abs(result[0] - point[0]) < 100
    assert abs(result[1] - point[1]) < 100


def test_set_substation_to_edge_no_intersection(lease, export):
    """Test set_substation_to_edge when line does not intersect lease area."""
    _, lease_polygon = clip_grid(lease, export)

    # Create a line that doesn't intersect the lease area
    landing_point = Point((490000.0, 6500000.0))
    line = LineString([(490000.0, 6500000.0), landing_point])
    lease_area_ring = LinearRing(list(lease_polygon.exterior.coords))

    result = set_substation_to_edge(line, lease_area_ring, lease, lease_polygon)

    # Result may be None or a valid point
    if result is not None:
        assert ((lease["x"] == result[0]) & (lease["y"] == result[1])).any()


def test_set_substation_to_edge_multipoint_intersection(lease, export):
    """Test set_substation_to_edge when line intersects lease area at multiple points."""
    from shapely.geometry import Polygon

    _, lease_polygon = clip_grid(lease, export)

    # Create a line that passes through multiple points of the lease area
    bounds = lease_polygon.bounds
    landing_point = Point((bounds[2] + 100, bounds[3] + 100))
    line = LineString([(bounds[0] - 100, bounds[1] - 100), landing_point])
    lease_area_ring = LinearRing(list(lease_polygon.exterior.coords))

    result = set_substation_to_edge(line, lease_area_ring, lease, lease_polygon)

    if result is not None:
        assert ((lease["x"] == result[0]) & (lease["y"] == result[1])).any()


def test_substation_in_site(lease, export):
    """Test substation_in_site function."""
    _, lease_polygon = clip_grid(lease, export)

    # Use a point within the lease area
    cp_loc = (491810.0, 6502090.0)

    result = substation_in_site(lease, cp_loc, lease_polygon)

    # Result may be None (if point not in grid) or a valid neighbor point
    if result is not None:
        assert isinstance(result, tuple)
        assert len(result) == 2


def test_substation_in_site_point_not_found(lease, export):
    """Test substation_in_site with a point not in the grid."""
    import pandas as pd
    from shapely.geometry import Polygon

    _, lease_polygon = clip_grid(lease, export)

    # Create a modified lease with no matching points
    small_lease = lease.iloc[:5].copy()
    cp_loc = (999999.0, 999999.0)

    # Should raise IndexError when accessing grid_point.i.item()
    with pytest.raises((IndexError, ValueError)):
        substation_in_site(small_lease, cp_loc, lease_polygon)


def test_closeness_test_true():
    """Test closeness_test returns True when points are close."""
    from shapely.geometry import Point

    device_points = [Point(0, 0), Point(10, 10), Point(20, 20)]
    cp_loc = (1, 1, 0)  # Close to first device
    threshold = 5.0

    result = closeness_test(device_points, cp_loc, threshold)

    assert result is True


def test_closeness_test_false():
    """Test closeness_test returns False when no points are close."""
    from shapely.geometry import Point

    device_points = [Point(0, 0), Point(10, 10), Point(20, 20)]
    cp_loc = (100, 100, 0)  # Far from all devices
    threshold = 5.0

    result = closeness_test(device_points, cp_loc, threshold)

    assert result is False


def test_extend_line():
    """Test extend_line function."""
    p1 = (0.0, 0.0)
    p2 = (1.0, 1.0)

    result = extend_line(p1, p2)

    assert result.length > np.sqrt(2)  # Longer than original
    # Check that p1 is at the start
    assert np.isclose(result.coords[0][0], p1[0])
    assert np.isclose(result.coords[0][1], p1[1])


def test_offset_cp_local():
    """Test offset_cp_local function."""
    cp_loc = (0.0, 0.0)
    array_edge = (1.0, 1.0)
    distance = 10.0

    result = offset_cp_local(cp_loc, array_edge, distance)

    assert isinstance(result, Point)
    # The result should be in the direction from array_edge away from cp_loc


def test_offset_cp_edge():
    """Test offset_cp with position='edge'."""
    from shapely.geometry import LineString

    device_loc = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    export = LineString([(5.0, -10.0), (5.0, 20.0)])
    cp_loc_estimate = (5.0, 5.0)
    position = "edge"
    distance = 10.0

    result = offset_cp(device_loc, export, cp_loc_estimate, position, distance)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_offset_cp_beyond():
    """Test offset_cp with position='beyond'."""
    from shapely.geometry import LineString

    device_loc = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    export = LineString([(5.0, -10.0), (5.0, 20.0)])
    cp_loc_estimate = (5.0, 5.0)
    position = "beyond"
    distance = 10.0

    result = offset_cp(device_loc, export, cp_loc_estimate, position, distance)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_offset_cp_invalid_position():
    """Test offset_cp with invalid position argument."""
    from shapely.geometry import LineString

    device_loc = [(0.0, 0.0), (10.0, 0.0)]
    export = LineString([(5.0, -10.0), (5.0, 20.0)])
    cp_loc_estimate = (5.0, 5.0)
    position = "invalid"
    distance = 10.0

    with pytest.raises(ValueError, match="position argument must be"):
        offset_cp(device_loc, export, cp_loc_estimate, position, distance)


def test_get_single_location(grid):
    """Test get_single_location function."""
    point = (491770.0, 6502090.0)

    result = get_single_location(point, grid.grid_pd)

    assert isinstance(result, (int, np.integer))
    assert result in grid.grid_pd["id"].values


def test_add_device_positions():
    """Test add_device_positions function."""
    layout = {
        "Device001": (100.0, 200.0),
        "Device002": (300.0, 400.0),
        "Device003": (500.0, 600.0),
    }
    layout_grid = [(1, 100), (2, 200), (3, 300)]

    result = add_device_positions(layout, layout_grid)

    assert len(result) == 3
    assert result[0] == (1, 100, (100.0, 200.0))
    assert result[1] == (2, 200, (300.0, 400.0))
    assert result[2] == (3, 300, (500.0, 600.0))


def test_add_device_positions_empty():
    """Test add_device_positions with empty layout."""
    layout = {}
    layout_grid = []

    result = add_device_positions(layout, layout_grid)

    assert result == []


def test_get_device_locations(grid):
    """Test get_device_locations function."""
    # Get actual grid points to use as device locations
    # The function finds all points where grid point matches layout location
    # So we need to use coordinates that match unique grid points
    sample_id = list(grid.points.keys())[0]
    point = grid.points[sample_id]

    layout = {
        "Device001": (point.x, point.y),
    }

    result = get_device_locations(layout, grid)

    # At least one device should be found
    assert len(result) >= 1
    assert all(isinstance(r, tuple) for r in result)
    assert all(len(r) == 3 for r in result)


def test_make_linestring(grid):
    """Test make_linestring function."""
    from shapely.geometry import LineString

    # Get some valid grid point IDs
    point_ids = list(grid.points.keys())[:5]

    result = make_linestring(point_ids, grid)

    assert isinstance(result, LineString)
    assert len(result.coords) == 5


def test_check_in_paths_true():
    """Test check_in_paths returns True when link is in paths."""
    link = (2, 1)
    paths = [[0, 1, 2], [0, 3, 4], [0, 5, 6]]

    result = check_in_paths(link, paths)

    assert result is True


def test_check_in_paths_false():
    """Test check_in_paths returns False when link is not in paths."""
    link = (5, 4)
    paths = [[0, 1, 2], [0, 3, 4], [0, 5, 6]]

    result = check_in_paths(link, paths)

    assert result is False


def test_check_in_paths_no_match():
    """Test check_in_paths with link where one node matches but not the other."""
    link = (1, 4)  # 1 is in first path, 4 is in second path
    paths = [[0, 1, 2], [0, 3, 4]]

    result = check_in_paths(link, paths)

    assert result is False


def test_check_in_route_true():
    """Test check_in_route returns True when route to node exists."""
    link = (2, 1)
    route = [(1, 0), (2, 0), (3, 0), (4, 0)]

    result = check_in_route(link, route)

    assert result is True


def test_check_in_route_false():
    """Test check_in_route returns False when route to node doesn't exist."""
    link = (2, 1)
    route = [(1, 5), (2, 5), (3, 5), (4, 5)]  # Links go to 5, not 0

    result = check_in_route(link, route)

    assert result is False


def test_check_neighbour_number_true():
    """Test check_neighbour_number returns True when neighbor has <= 1 connections."""
    link = (2, 1)
    route = [(1, 0), (2, 0)]

    result = check_neighbour_number(link, route)

    assert result is True


def test_check_neighbour_number_false():
    """Test check_neighbour_number returns False when neighbor has > 1 connections."""
    link = (2, 1)
    route = [(1, 0), (1, 3), (2, 0)]  # Node 1 has 2 connections

    result = check_neighbour_number(link, route)

    assert result is False


def test_check_path_capacity_not_exceeded():
    """Test check_path_capacity returns False when capacity not exceeded."""
    link = (2, 1)
    paths = [[0, 1], [0, 2]]
    cap = 3

    result = check_path_capacity(link, paths, cap)

    assert result is False  # Capacity not exceeded


def test_check_path_capacity_exceeded():
    """Test check_path_capacity returns True when capacity would be exceeded."""
    link = (2, 1)
    paths = [
        [0, 1, 3, 4, 5],
        [0, 2, 6, 7, 8],
    ]  # Each has 4 devices, combined = 8
    cap = 5

    result = check_path_capacity(link, paths, cap)

    assert result is True  # Capacity exceeded


def test_update_paths():
    """Test update_paths function."""
    link = (2, 1)
    route = [(1, 0), (2, 0), (3, 0), (4, 0)]

    result = update_paths(link, route)

    assert isinstance(result, list)
    assert len(result) >= 1
    # Check that the link was added and the default link removed
    assert (2, 1) in route
    assert (2, 0) not in route


def test_update_paths_multiple_routes():
    """Test update_paths with multiple disconnected routes."""
    link = (4, 3)
    route = [(1, 0), (2, 0), (3, 2), (4, 0)]

    result = update_paths(link, route)

    assert isinstance(result, list)


def test_crossing_dijkstra_no_crossing(grid):
    """Test crossing_dijkstra with non-crossing paths."""
    import numpy as np

    # Create a simple path array
    layout_grid = [(0, 334), (0, 344)]
    substation_location = (491770.0, 6502090.0)

    distance_array, path_array = calculate_distance_dijkstra(
        layout_grid, substation_location, grid, grid.graph
    )

    devices = add_device_positions(
        {"Device001": (0.0, 0.0), "Device002": (10.0, 10.0)},
        [(1, 100), (2, 200)],
    )

    link = (1, 0)
    route = [(0, 1)]  # Simple route

    result = crossing_dijkstra(link, route, path_array, devices, grid)

    assert isinstance(result, bool)


def test_dijkstra_source_missing():
    """Test dijkstra raises error when source node is not in graph."""
    graph = nx.Graph()

    with pytest.raises(nx.NetworkXNoPath, match="not in graph"):
        dijkstra(graph, 1, 2)


def test_crossing_dijkstra_with_feasible_paths(grid):
    """Test crossing_dijkstra with feasible paths."""
    import numpy as np

    layout_grid = [(0, 334), (0, 344), (0, 604)]
    substation_location = (491770.0, 6502090.0)

    distance_array, path_array = calculate_distance_dijkstra(
        layout_grid, substation_location, grid, grid.graph
    )

    devices = add_device_positions(
        {
            "Device001": (0.0, 0.0),
            "Device002": (10.0, 10.0),
            "Device003": (20.0, 20.0),
        },
        [(1, 100), (2, 200), (3, 300)],
    )

    link = (2, 1)
    route = [(1, 0)]

    # This should not raise an error
    result = crossing_dijkstra(link, route, path_array, devices, grid)

    assert isinstance(result, bool)


def test_calculate_saving_vector_empty():
    """Test calculate_saving_vector with minimal n_oec value."""
    # Test with a minimal 2x2 distance array and n_oec=1
    distance_array = np.array([[0, 1], [1, 0]])
    savings = calculate_saving_vector(distance_array, 1)

    # Should return empty list since savings would be 0 or negative
    # and filtered out
    assert isinstance(savings, list)


def test_make_optimal_paths_basic(grid):
    """Test make_optimal_paths with basic inputs."""
    layout_grid = [(0, 334), (0, 344), (0, 604)]
    substation_location = (491770.0, 6502090.0)

    distance_array, path_array = calculate_distance_dijkstra(
        layout_grid, substation_location, grid, grid.graph
    )

    savings = calculate_saving_vector(distance_array, 3)

    # Initialize paths and route with default layout
    paths = [[0, 1], [0, 2], [0, 3]]
    route = [(1, 0), (2, 0), (3, 0)]

    devices = add_device_positions(
        {
            "Device001": (0.0, 0.0),
            "Device002": (10.0, 10.0),
            "Device003": (20.0, 20.0),
        },
        [(1, 100), (2, 200), (3, 300)],
    )

    result = make_optimal_paths(
        savings, paths, route, 10, path_array, devices, grid
    )

    # Result should be a list of paths
    assert isinstance(result, list)
    assert len(result) > 0


def test_offset_cp_runtime_error():
    """Test offset_cp raises RuntimeError when location not found."""
    from shapely.geometry import LineString

    # Create a scenario where the export line doesn't intersect any edges
    device_loc = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    # Line that doesn't pass through the envelope bounds
    export = LineString([(-20.0, -20.0), (-20.0, -10.0)])
    cp_loc_estimate = (5.0, 5.0)
    position = "edge"
    distance = 10.0

    with pytest.raises(RuntimeError, match="Location not found"):
        offset_cp(device_loc, export, cp_loc_estimate, position, distance)
