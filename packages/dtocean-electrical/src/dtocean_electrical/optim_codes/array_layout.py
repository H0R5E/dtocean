# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Adam Collin
#    Copyright (C) 2017-2021 Mathew Topper
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
This module defines the DTOcean electrical subsystems array routing functions.

.. module:: array_layout
   :platform: Windows
   :synopsis: Intra-array cable routing functions.

.. moduleauthor:: Adam Collin <adam.collin@ieee.org>
.. moduleauthor:: Mathew Topper <mathew.topper@dataonlygreater.com>
"""

import itertools
import logging
from typing import Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.special import comb
from shapely.geometry import LinearRing, LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points

from ..grid.grid import Grid

module_logger = logging.getLogger(__name__)

DevicePositions = list[tuple[int, int, tuple[float, float]]]
Link = tuple[int, int, float]
Paths = list[list[int]]
Route = list[tuple[int, int]]


def snap_to_grid(
    grid: np.ndarray,
    point: tuple[float, float],
    lease: pd.DataFrame,
) -> tuple[float, ...]:
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

    new_coords = grid[spatial.KDTree(grid).query(np.array(point))[1]].tolist()

    # and add z coord
    z = lease[(lease.x == new_coords[0]) & (lease.y == new_coords[1])][
        "layer 1 start"
    ].values[0]

    new_coords.append(z)
    new_coords = [float(i) for i in new_coords]

    return tuple(new_coords)


def set_substation_to_edge(
    line: LineString,
    lease_area_ring: LinearRing,
    lease_bathymetry: pd.DataFrame,
    lease: Polygon,
) -> tuple[float, float] | None:
    interim_estimate = None

    # find poi between line of intial to shore and lease area
    lease_x = lease_bathymetry.x.tolist()
    lease_y = lease_bathymetry.y.tolist()

    grid_to_search = np.array([lease_x, lease_y]).T

    if line.intersects(lease_area_ring):
        poi = lease_area_ring.intersection(line)

        if isinstance(poi, MultiPoint):
            line_end = Point(line.coords[-1])
            poi_candidates = list(poi.geoms)

            distances = [p.distance(line_end) for p in poi_candidates]
            min_distance_idx = distances.index(min(distances))

            poi = poi_candidates[min_distance_idx]

        assert isinstance(poi, Point)
        poi = (poi.x, poi.y)

    else:
        coord = nearest_points(line, lease_area_ring)[1].coords[0]
        poi = (coord[0], coord[1])

    # snap to nearest point
    interim_estimate_snapped = snap_to_grid(
        grid_to_search,
        poi,
        lease_bathymetry,
    )

    # then shift

    # find cp_loc in list of points
    grid_point = lease_bathymetry[
        (lease_bathymetry.x == interim_estimate_snapped[0])
        & (lease_bathymetry.y == interim_estimate_snapped[1])
    ]

    # get neighbours
    check_x_direction = [0, 0, -1, 1, -1, 1, -1, 1]
    check_y_direction = [-1, 1, 0, 0, -1, 1, 1, -1]

    neighbour_ids = [
        (grid_point.i.item() - i_shift, grid_point.j.item() - j_shift)
        for i_shift, j_shift in zip(check_x_direction, check_y_direction)
    ]

    for neighbour in neighbour_ids:
        neighbour = lease_bathymetry[
            (lease_bathymetry.i == neighbour[0])
            & (lease_bathymetry.j == neighbour[1])
        ]

        neighbour_shapely = Point(neighbour.x, neighbour.y)

        if neighbour_shapely.within(lease):
            interim_estimate = (neighbour.x.item(), neighbour.y.item())
            break

    return interim_estimate


def substation_in_site(
    grid_df: pd.DataFrame,
    cp_loc: tuple[float, float],
    lease: Polygon,
) -> tuple[float, float] | None:
    # find cp_loc in list of points
    cp_estimate = None
    grid_point = grid_df[(grid_df.x == cp_loc[0]) & (grid_df.y == cp_loc[1])]

    # get neighbours
    check_x_direction = [0, 0, -1, +1, -1, +1, -1, +1]
    check_y_direction = [-1, +1, 0, 0, -1, +1, -1, +1]

    neighbour_ids = [
        (grid_point.i.item() - i_shift, grid_point.j.item() - j_shift)
        for i_shift, j_shift in zip(check_x_direction, check_y_direction)
    ]

    for neighbour in neighbour_ids:
        neighbour = grid_df[
            (grid_df.i == neighbour[0]) & (grid_df.j == neighbour[1])
        ]

        neighbour_shapely = Point(neighbour.x, neighbour.y)

        if neighbour_shapely.within(lease):
            cp_estimate = (neighbour.x.item(), neighbour.y.item())

            break

    return cp_estimate


def closeness_test(
    device_points: Sequence[Point],
    cp_loc: tuple[float, ...],
    threshold: float,
) -> bool:
    close = False

    for oec in device_points:
        if oec.distance(Point(cp_loc[:2])) < threshold:
            close = True

    return close


def offset_cp(
    device_loc: Sequence[tuple[float, float]],
    export: LineString,
    cp_loc_estimate: tuple[float, float],
    position: str,
    distance: float,
) -> tuple[float, float]:
    # make points
    device_points = []

    for item in device_loc:
        device_points.append(Point(item[0], item[1]))

    point_collection = MultiPoint(device_points)
    envelope = point_collection.envelope

    if not isinstance(envelope, Polygon):
        raise RuntimeError("Could not find enclosing geometry for devices")

    edges = list(zip(*envelope.exterior.xy))
    edge_strings = []

    for id_, point in enumerate(edges[:-1]):
        # make line
        edge_strings.append(LineString([point, edges[id_ + 1]]))

    new_cp_loc = None

    for item in edge_strings:
        poi = export.intersection(item)

        if poi:
            area_centre = (item.centroid.xy[0][0], item.centroid.xy[1][0])

            if position == "edge":
                new_cp_loc = Point(area_centre[0], area_centre[1])
            elif position == "beyond":
                new_cp_loc = offset_cp_local(
                    cp_loc_estimate,
                    area_centre,
                    distance,
                )
            else:
                raise ValueError("position argument must be 'edge or 'beyond'")

            break

    if new_cp_loc is None:
        raise RuntimeError("Location not found")

    return (new_cp_loc.x, new_cp_loc.y)


def offset_cp_local(
    cp_loc: tuple[float, float],
    array_edge: tuple[float, float],
    distance: float,
) -> Point:
    """Offset the collection point beyond the array boundary by value specified
    in distance.

    Args:
        cp_loc
        array_edge

    """

    extended_line = extend_line(cp_loc, array_edge)
    # make line from edge to end of extended line
    # first_point = (array_edge[0], array_edge[1])
    last_point = (extended_line.xy[0][1], extended_line.xy[1][1])
    new_end = LineString([array_edge, last_point]).interpolate(distance)

    return new_end


def extend_line(p1: tuple[float, float], p2: tuple[float, float]) -> LineString:
    """Extend line in p1 -> p2 direction.

    http://stackoverflow.com/questions/33159833/shapely-extending-line-feature

    Args:
        p1
        p2

    """

    ratio = 5
    a = p1
    b = (p1[0] + ratio * (p2[0] - p1[0]), p1[1] + ratio * (p2[1] - p1[1]))

    return LineString([a, b])


def dijkstra(
    graph: nx.Graph,
    a: int,
    b: int,
    grid: Optional[Grid] = None,
) -> tuple[float, list[int]]:
    def log_error():
        if grid is None:
            return

        id_grid_pd = grid.grid_pd.set_index("id")
        target_missing = b not in id_grid_pd.index

        if target_missing:
            logMsg = "Target node '{}' is not in grid".format(b)

        else:
            source_node = id_grid_pd.loc[a]
            target_node = id_grid_pd.loc[b]
            logMsg = (
                "Point ({}, {}) not reachable from point "
                "({}, {}), with node numbers {} and {}"
            ).format(
                target_node["x"],
                target_node["y"],
                source_node["x"],
                source_node["y"],
                b,
                a,
            )

        module_logger.debug(logMsg)

    if not graph.has_node(a):
        raise nx.NetworkXNoPath("node {} not in graph".format(a))

    try:
        length, path = nx.single_source_dijkstra(graph, a, b)
        assert isinstance(length, float)
        assert isinstance(path, list)
    except nx.NetworkXNoPath:
        if grid is not None:
            log_error()
        raise nx.NetworkXNoPath("node {} not reachable from {}".format(b, a))

    return length, path


def get_export(
    cp_loc: tuple[float, ...],
    landing_loc: tuple[float, ...],
    grid: Grid,
    graph: nx.Graph,
) -> tuple[float, list[int]]:
    """Get the export cable route and length."""

    a = get_single_location(landing_loc[:2], grid.grid_pd)
    b = get_single_location(cp_loc[:2], grid.grid_pd)

    length, route = dijkstra(graph, a, b, grid)

    return length, route


def calculate_distance_dijkstra(
    layout_grid: list[tuple[int, int]],
    substation_location: tuple[float, ...],
    grid: Grid,
    graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the distance between all devices in layout_grid and between
    all devices and a fixed point defined by substation_location. This uses
    dijkstras algorithm to calculate the seabed distance.

    Args:
        layout_grid
        substation_location (float) [-]: substation_location as x,y,z.
        grid (object) [-]: grid object of seabed.
        graph

    Returns:
        distance_array
        path_array

    """

    layout_list = [x[1] for x in layout_grid]

    # Insert substation location
    substation_point = get_single_location(
        substation_location[:2],
        grid.grid_pd,
    )
    layout_list.insert(0, substation_point)

    # Initialise output arrays
    nlocs = len(layout_list)
    distance_array = np.zeros([nlocs, nlocs])
    path_array = np.empty([nlocs, nlocs], dtype=object)

    ncombos = int(comb(nlocs, 2))
    module_logger.debug("{} node combinations found".format(ncombos))

    layout_ids = range(nlocs)

    for i, j in itertools.combinations(layout_ids, 2):
        module_logger.debug(
            "Evaluating node combination " "({}, {})".format(i, j)
        )

        a = layout_list[i]
        b = layout_list[j]

        ij_length, ij_path = dijkstra(graph, a, b, grid)

        if len(ij_path) < 2:
            err_str = (
                "The path between two devices contains less than two "
                "grid points. Consider increasing spacing between "
                "devices or grid resolution."
            )
            raise RuntimeError(err_str)

        distance_array[i, j] = ij_length
        path_array[i, j] = ij_path

        # Reverse paths
        distance_array[j, i] = ij_length
        path_array[j, i] = ij_path[::-1]

    return distance_array, path_array


def get_single_location(
    point: tuple[float, ...],
    grid_points: pd.DataFrame,
) -> int:
    """Get the location of a single component on the grid."""

    grid = np.array(grid_points[["x", "y"]])

    new_coords = grid[
        spatial.KDTree(grid).query(np.array(point[:2]))[1]
    ].tolist()

    # and add z coord
    idx = grid_points[
        (grid_points.x == new_coords[0]) & (grid_points.y == new_coords[1])
    ]["id"].values[0]

    #    new_coords.append(z)
    #    new_coords = [float(i) for i in new_coords]

    return idx


def add_device_positions(
    layout: dict[str, tuple[float, ...]],
    layout_grid: list[tuple[int, int]],
) -> DevicePositions:
    """Add the device positions to the given layout_grid argument as the
    third item in each tuple.
    """

    new: DevicePositions = []

    for item in layout_grid:
        for key, value in layout.items():
            if key == "Device" + str(item[0]).zfill(3):
                new.append((item[0], item[1], (value[0], value[1])))

    return new


def get_device_locations(
    layout: dict[str, tuple[float, ...]],
    site_grid: Grid,
) -> DevicePositions:
    """
    Note:
        This assumes direct overlapping. Will this always be the case?
        Unlikely but check this.

    """

    device_on_grid = []

    for key, value in layout.items():
        for point in site_grid.points.values():
            if np.isclose(point.x, value[0]) and np.isclose(point.y, value[1]):
                device_on_grid.append(
                    (int(key[6:]), point.index, (value[0], value[1]))
                )

    device_on_grid = sorted(device_on_grid, key=lambda x: x[0])

    return device_on_grid


def calculate_saving_vector(
    distance_vector: np.ndarray,
    n_oec: int,
) -> list[Link]:
    """Calculate distance saving of going from node j to i rather than directly
    to i from the origin. Remove any combinations that do not result in savings."""
    saving_vector: list[Link] = [
        (i, j, float(distance_vector[0][i] - distance_vector[j][i]))
        for i in range(0, n_oec + 1)
        for j in range(0, n_oec + 1)
        if i != j
    ]

    saving_vector_sorted = sorted(
        saving_vector, key=lambda i: i[2], reverse=True
    )
    # tidy up savings vector by removing self connecting nodes
    saving_vector_filtered: list[Link] = []
    for point in saving_vector_sorted:
        if point[0] != point[1] and point[2] >= 0.0:
            saving_vector_filtered.append(point)

    return saving_vector_filtered


def make_optimal_paths(
    savings_vector: Sequence[Link],
    paths: Paths,
    route: Route,
    string_max: int,
    path_array: np.ndarray,
    devices: DevicePositions,
    site_grid: Grid,
) -> Paths:
    """Update the given paths based on the links in the savings_vector

    paths contains a list of lists where the inner lists are sequences of
    connected nodes, starting from zero. This should be initialised with
    the default layout (e.g [[0, 1], [0, 2], ...])

    route is list of valid links between nodes where each link is a tuple
    with the end node first and the start second. This should be initialised
    to match the existing paths (e.g. [(1, 0), (2, 0), ...])

    saving_vector contain a list of potential links that are used to update
    the default paths. If a link in saving_vector is valid it is added to the
    route (replacing the default) and the paths are recalculated. Assuming the
    savings vector is calculated using calculate_saving_vector then it
    represents paths between nodes that save distance compared to the default
    layout.
    """
    for link in savings_vector:
        if (
            not check_in_paths(link, paths)
            and check_in_route(link, route)
            and check_neighbour_number(link, route)
            and not check_path_capacity(link, paths, string_max)
            and not crossing_dijkstra(
                link,
                route,
                path_array,
                devices,
                site_grid,
            )
        ):
            paths = update_paths(link, route)

    return paths


def check_in_paths(link: Link, paths: Paths) -> bool:
    """Check if the start and end nodes of the link are already contained
    in any of the paths."""
    in_path = []

    for i in range(0, len(paths)):
        if link[0] in paths[i] and link[1] in paths[i]:
            in_path.append(True)
        else:
            in_path.append(False)

    return any(in_path)


def check_in_route(link: Link, route: Route) -> bool:
    """
    Check if the default route to the node still exists (i.e. from the
    substation)
    """

    return (link[0], 0) in route


def check_neighbour_number(link: Link, route: Route) -> bool:
    """Return true if link[1] is not already connected to two other nodes"""

    counter_u = 0

    for arc in route:
        if link[1] in arc:
            counter_u += 1

    return counter_u <= 1


def check_path_capacity(link: Link, paths: Paths, cap: int) -> bool:
    """Check if adding the link would exceed the maximum number of nodes for a
    path"""
    path_length_k = 0
    path_length_u = 0

    for i in paths:
        if link[0] in i:
            path_length_k = len(i) - 1
        if link[1] in i:
            path_length_u = len(i) - 1

    # sum path lengths and check capacity
    total = path_length_k + path_length_u

    if total > cap:
        cap_exceed = True
    else:
        cap_exceed = False

    return cap_exceed


def crossing_dijkstra(
    link: Link,
    route: Route,
    path_array: np.ndarray,
    devices: DevicePositions,
    site_grid: Grid,
) -> bool:
    """Check if new link will cross any of the existing links in the current
    route using Dijkstra's algorithm generated paths.
    """

    # make line
    new_path = path_array[link[0]][link[1]]
    new_line = make_linestring(new_path, site_grid)

    # initiate crossing vector
    cross = []

    for edge in route:
        # make line
        existing_path = path_array[edge[0]][edge[1]]

        # Route is infeasible
        if len(existing_path) < 2:
            cross.append(True)
            continue

        existing_line = make_linestring(existing_path, site_grid)
        link_0_point = Point(*devices[link[0] - 1][2])
        link_1_point = Point(*devices[link[1] - 1][2])

        if new_line.intersection(existing_line).is_empty:
            cross.append(False)
        elif new_line.intersection(existing_line) == link_0_point:
            cross.append(False)
        elif new_line.intersection(existing_line) == link_1_point:
            cross.append(False)
        elif new_line.distance(link_0_point) < 1e-3:
            cross.append(False)
        elif new_line.distance(link_1_point) < 1e-3:
            cross.append(False)
        else:
            cross.append(True)

    return any(cross)


def make_linestring(path: Sequence[int], site_grid: Grid) -> LineString:
    path_points = [(site_grid.points[i].x, site_grid.points[i].y) for i in path]
    return LineString(path_points)


def update_paths(link: Link, route: Route) -> Paths:
    """Add the link to the route vector, remove the default link (i.e from
    the substation to link[0]) and recalculate the paths from the links in the
    route"""

    route.append((link[0], link[1]))
    # Remove link to 0 point
    if (link[0], 0) in route:
        route.remove((link[0], 0))

    paths: Paths = []
    route_copy = list(route)

    first = route_copy.pop(0)
    start = first[1]
    end = first[0]
    path_interim = [start, end]

    start_check = []

    while route_copy:
        for i in route_copy:
            start_check.append(i[1])

        # If end node equals any start nodes then append to the current path
        if end in start_check:
            for i in route_copy:
                if i[1] == end:
                    path_interim.append(i[0])
                    start_check = []

                    route_copy.remove(i)
                    end = i[0]
                    break

        # Finish the current path and start a new one
        else:
            paths.append(path_interim)
            start_check = []

            next = route_copy.pop(0)
            start = next[1]
            end = next[0]
            path_interim = [start, end]

    # Add the final path to the list
    paths.append(path_interim)

    return paths
