# -*- coding: utf-8 -*-

#    Copyright (C) 2016 Sam Weller
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
Umbilical cable calculations

.. moduleauthor:: Sam Weller <S.Weller@exeter.ac.uk>
.. moduleauthor:: Mathew Topper <mathew.topper@dataonlygreater.com>
"""

import logging
import math
from collections import namedtuple
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import spatial
from shapely import LineString, Point

if TYPE_CHECKING:
    from ..main import Electrical


PointTuple = tuple[float, float, float]

# Start logging
module_logger = logging.getLogger(__name__)


class UmbilicalDesign:
    """Design umbilical cable for floating devices."""

    def __init__(self, data: "Electrical", reuse_lengths: bool = True):
        """
        Args:
            reuse_lengths (bool, optional) [-]: Reuse length calculations from
                previous run, unless the cable db_key has changed.
                Defaults to True.
        """

        self.designs: dict[str, dict[str, Any]] | None = None
        self._meta_data = data
        self._reuse_lengths = reuse_lengths
        self._db_key: int | None = None
        self._umbilical_data: pd.DataFrame | None = None

        return

    def umbilical_design(
        self,
        paths: np.ndarray,
        db_key: int,
        sol: list[list[int]],
    ):
        """Call code to design the umbilical.

        Args:
            paths (np.ndarray) [-]: Path of seabed cables between devices and
                point of connection.
            db_key (int) [-]: DB key of selected umibilical.

        Attributes:
            all_umbilical_data (pd.DataFrame) [-]: Filtered copy of the
                electrical component db, containing only the selected
                umibilical cable.
            umbilical_parameters (dict) [-]: Electrical component db converted
                into format required by umbilical design module.
            termination_points (dict) [m]: Seabed connection point of each
                device as (x, y, z) coordinates; key = device number, value =
                (x, y, z).
            umbilical_vars (object) [-]: Variables object.
            umbilical (object) [-]: Umbilical object.
            all_cable_designs (dict) [-]: All umbilical designs, key is the
                device id.

        Returns:
            all_cable_designs

        Note:
            Device type 'wavefloat' is always passed as the results are
            independent of if 'wavefloat' or 'tidefloat' is specified.

        """

        if self._reuse_lengths:
            if self._db_key is None or self._db_key != db_key:
                reuse_lengths = False
            else:
                reuse_lengths = True

            self._db_key = db_key

        dynamic_cable_db = self._meta_data.database.dynamic_cable
        array_data = self._meta_data.array_data
        options = self._meta_data.options

        self._umbilical_data = dynamic_cable_db[dynamic_cable_db.id == db_key]

        umbilical_parameters = self._umbilical_map()
        devices = self._get_device_ids(sol)
        termination_dict: dict[str, list[float]] = {}

        for device_n in devices:
            cable_termination = self._set_umbilical_termination(
                paths,
                device_n,
                sol,
            )

            device_id = "Device" + str(device_n).zfill(3)

            if reuse_lengths:
                assert isinstance(self.designs, dict)
                self.designs[device_id]["termination"] = cable_termination
                self.designs[device_id]["db_key"] = db_key

            else:
                termination_dict[device_id] = cable_termination

        if reuse_lengths:
            return

        logMsg = ("Calculating umbilical lengths using cable id: " "{}").format(
            db_key
        )
        module_logger.info(logMsg)

        umbilical_vars = Variables(
            list(termination_dict.keys()),
            options.gravity,
            umbilical_parameters,
            "wavefloat",
            array_data.layout,
            array_data.machine_data.connection_point,
            array_data.orientation_angle,
            db_key,
            options.umbilical_safety_factor,
            termination_dict,
            array_data.machine_data.draft,
        )

        umbilical = Umbilical(umbilical_vars)

        all_cable_designs: dict[str, dict[str, Any]] = {}

        for device_id in umbilical_vars.devices:
            dev_orig = umbilical_vars.sysorig[device_id]

            (umbleng, umbxcoords, umbzcoords) = umbilical.umbdes(
                device_id,
                dev_orig,
            )

            result = {
                "device": device_id,
                "length": umbleng,
                "x coords": umbxcoords,
                "z coords": umbzcoords,
                "termination": umbilical_vars.subcabconpt[device_id],
                "db_key": db_key,
            }

            all_cable_designs[device_id] = result

        self.designs = all_cable_designs

    def _umbilical_map(self):
        """Convert electrical component database into format required by
        umbilical design module.

        Args:
            data (pd.DataFrame) [-]: DB entry of umbilical cable.

        Attributes
            umbilical_db (dict) [-]: Collection of only the data required for
                the umbilical design module.

        """

        assert self._umbilical_data is not None
        data = self._umbilical_data

        umbilical_db = {
            data.id.values[0]: {
                "item3": None,
                "item5": [data.mbl.values[0], data.mbr.values[0]],
                "item6": [data.diameter.values[0]],
                "item7": [data.dry_mass.values[0], data.wet_mass.values[0]],
            }
        }

        return umbilical_db

    def _set_umbilical_termination(
        self,
        path: np.ndarray,
        device: int,
        sol: list[list[int]],
    ) -> list[float]:
        """Logic to set the umbilical termination point. This defines a fixed
        point along the seabed cable projection. Two values are compared -
        1.5 x sea depth and 0.5 x seabed cable projection length - and the
        largest value selected.

        Attributes:
            line (Shapely LineString): LineString representation of cable
                route.
            termination_approximation (Shapely Point):
            termination_fixed ()

        Note:
            This currently only uses 1.5.

        """

        initial_guess = 1.5

        connect = [item for item in sol if device in item][0]
        downstream = connect.index(device) - 1

        line_path = path[connect[downstream]][device]
        points = self._make_shapely_point_list(line_path)

        if len(points) > 1:
            line = LineString(points)
            depth = self._meta_data.site_data.min_water_depth * initial_guess
            termination_approximation = line.interpolate(depth)

        else:
            termination_approximation = points[0]

        x, y = zip(
            *[
                (
                    self._meta_data.grid.points[point].x,
                    self._meta_data.grid.points[point].y,
                )
                for point in line_path
            ]
        )

        grid_to_search = np.array([x, y]).T
        point_to_check = (
            termination_approximation.x,
            termination_approximation.y,
        )

        termination_fixed = self.snap_to_grid(grid_to_search, point_to_check)

        # Check that z is negative
        assert np.sign(termination_fixed[2]) == -1.0

        return list(termination_fixed)

    def _make_shapely_point_list(self, path: np.ndarray) -> list[Point]:
        """Description to be added.

        Args:
            path () [-]:

        Return:
            list () [-]: List of Shapely Point objects.

        """

        return [
            self._meta_data.grid.points[point].shapely_point
            for point in path[::-1]
        ]

    def _get_device_ids(self, sol) -> list[int]:
        """Convert chain into unique device ids and remove central collection
        point at zero.

        Args:
            sol (list, tuples) [-]: List of connection tuples.

        Attributes:
            unique_values (set) [-]: Set of unique device ids.
            unique_values_as_list (list) [-]:

        Returns:
            unique_values_as_list

        """

        unique_values = set([val for item in sol for val in item])
        unique_values_as_list = list(unique_values)
        unique_values_as_list.remove(0)

        return unique_values_as_list

    def snap_to_grid(
        self,
        grid: np.ndarray,
        point: tuple[float, float],
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

        new_coords = grid[
            spatial.KDTree(grid).query(np.array(point))[1]
        ].tolist()

        # and add z coord
        z = self._meta_data.grid.grid_pd[
            (self._meta_data.grid.grid_pd.x == new_coords[0])
            & (self._meta_data.grid.grid_pd.y == new_coords[1])
        ]["layer 1 start"].values[0]

        new_coords.append(z)
        new_coords = [float(i) for i in new_coords]

        return tuple(new_coords)

    def umbilical_impedance_table(
        self,
        override: Optional[dict[str, dict[str, Any]]] = None,
    ) -> list[tuple[float, ...]]:
        """Calculate impedance of each umbilical cable."""

        if self._umbilical_data is None:
            return []

        impedance_values: list[tuple[float, ...]] = []
        keys: list[int] = []

        z_data = (
            self._umbilical_data.r_ac.item(),
            self._umbilical_data.xl.item(),
            self._umbilical_data.c.item(),
        )

        if override:
            designs = override
        else:
            designs = self.designs

        if designs is None:
            return []

        for key, val in designs.items():
            length = val["length"] / 1000  # m to km
            impedance = [length * item for item in z_data]

            idx = int(key.strip("Device"))

            impedance_values.append(tuple(impedance))
            keys.append(idx)

        sorted_impedance_values = [
            z for (_, z) in sorted(zip(keys, impedance_values))
        ]

        return sorted_impedance_values


class Variables:
    """Collect input data for the umbilical design module.

    Args:
        devices (list) [-]: List of device identification numbers.
        gravity (float) [m/s2]: Acceleration due to gravity.
        compdict (dict) [-]: Representation of db data for umbilical design
            module:
                key = 'item3', value = None,
                key = 'item5', value = [minimum break load,
                                        minimum bend radius]
                key = 'item6', value = [diameter]
                key = 'item7', value = [dry mass, wet mass]
        systype (str) [-]: Device type, from: 'tidefloat', 'tidefixed',
            'wavefloat', 'wavefixed'.
        sysorig () [-]: Seabed connection point of each device as (x, y, z)
            coordinates; key = device number, value = (x, y, z).
        umbconpt (np.ndarray) [m]: Umbilical connection point as (x, y, z)
            coordinates. x and y given with respect to device origin, z with
            respect to MSL.
        sysorienang (float) [deg]: System orientation angle.
        preumb (int) [-]: DB key of the selected umbilical.
        umbsf (float) [-]: Umbilical safety factor.
        subcabconpt (dict) [m]: Subsea cable connection point for each device
            as (x, y, z) coordinates.
        sysdraft (float) [m]: Device equilibrium draft without mooring/cable
            system.

    Attributes:
        None

    """

    def __init__(
        self,
        devices: Sequence[str],
        gravity: float,
        compdict: dict[int, dict[str, Any]],
        systype: str,
        sysorig: dict[str, tuple[float, ...]],
        umbconpt: PointTuple,
        sysorienang: float,
        preumb: int,
        umbsf: float,
        subcabconpt: dict[str, list[float]],
        sysdraft: float,
    ):
        self.devices = devices
        self.gravity = gravity
        self.compdict = compdict
        self.systype = systype
        self.sysorig = sysorig
        self.umbconpt = umbconpt
        self.sysorienang = sysorienang
        self.preumb = preumb
        self.umbsf = umbsf
        self.subcabconpt = subcabconpt
        self.sysdraft = sysdraft

        return


class Umbilical:
    """Umbilical geometry specification submodule

    Args:
        variables (object) [-]: Instance of Variables class.

    Attributes:
        selumbtyp: (str) [-]: Selected umbilical type.
        umbgeolw (numpy.ndarray): Not yet fully defined.
        umbleng (float) [m]: Umbilical length.
        totumbcost (float) [euros]: Umbilical total cost.
        umbbomat (dict) [-]: Umbilical bill of materials, keys:
            'umbilical type' [str], 'length' [m], 'cost' [euros]
            'total weight' [kg], 'diameter' [m].
        umbhier (list) [-]: Umbilical hierarchy.

    Functions:
        umbdes: Specifies umbilical geometry.
        umbinst: Calculates installation parameters.
        umbcost: Calculates umbilical capital cost.
        umbbom: Creates umbilical bill of materials.
        umbhier: Creates umbilical hierarchy.

    """

    def __init__(self, variables: Variables):
        self._variables = variables

    def umbdes(self, deviceid: str, syspos: tuple[float, ...]):
        """This method will be used to look-up umbilical properties"""

        # Define geometry
        # Umbilical defined by WP3
        preumb_record = self._variables.compdict[self._variables.preumb]
        self.selumbtyp = preumb_record["item3"]

        if self._variables.systype in ("wavefloat", "tidefloat"):
            # """ Lazy-wave geometry comprises three sections; hang-off,
            # buoyancy and decline """
            compblocks = ["hang off", "buoyancy", "decline"]
        elif self._variables.systype in ("wavefixed", "tidefixed"):
            compblocks = ["hang off"]

        prop_cols = [
            "compid",
            "size",
            "length",
            "dry_mass",
            "wet_mass",
            "mbl",
            "mbr",
        ]
        UmpProps = namedtuple("UmpProps", prop_cols)

        umbwetmass = [0.0 for row in range(len(compblocks))]
        umbconpt_rotated = [0.0 for row in range(0, 2)]
        umbtopconn = [0.0 for row in range(0, 3)]
        angle_rads = -self._variables.sysorienang * math.pi / 180.0
        pre_rotated = self._variables.umbconpt[:2]

        # Rotate the connection point
        umbconpt_rotated[0] = pre_rotated[0] * math.cos(
            angle_rads
        ) - pre_rotated[1] * math.sin(angle_rads)
        umbconpt_rotated[1] = pre_rotated[0] * math.sin(
            angle_rads
        ) + pre_rotated[1] * math.cos(angle_rads)

        # Move the connection point to global coordinates
        umbtopconn[0] = round(umbconpt_rotated[0] + syspos[0], 3)
        umbtopconn[1] = round(umbconpt_rotated[1] + syspos[1], 3)

        if self._variables.systype in ("wavefloat", "tidefloat"):
            umbtopconn[2] = (
                self._variables.umbconpt[2] - self._variables.sysdraft
            )
            klim = 1
        elif self._variables.systype in ("wavefixed", "tidefixed"):
            umbtopconn[2] = self._variables.umbconpt[2]
            klim = 500

        # Umbilical length set initially as 1.15 x the shortest distance
        # between the upper and lower connection points
        subcabconpts = self._variables.subcabconpt[deviceid]

        umbleng = 1.15 * math.sqrt(
            (umbtopconn[0] - subcabconpts[0]) ** 2.0
            + (umbtopconn[1] - subcabconpts[1]) ** 2.0
            + (umbtopconn[2] - subcabconpts[2]) ** 2.0
        )
        umblengcheck = "False"

        for k in range(0, klim):
            # logmsg = ('Umbilical total length {}').format(umbleng)
            # module_logger.debug(logmsg)

            # If any z-coordinate is below the global subsea cable connection
            # point reduce umbilical length by 0.5%
            if (
                self._variables.systype in ("wavefixed", "tidefixed")
                and k > 0
                and umblengcheck == "False"
            ):
                umbleng = 0.999 * umbleng

            if self._variables.systype in ("wavefloat", "tidefloat"):
                # Initial lengths of hang-off, buoyancy and decline sections
                # (40%, 20% and 40%
                umbsecleng = [0.4 * umbleng, 0.2 * umbleng, 0.4 * umbleng]
            elif self._variables.systype in ("wavefixed", "tidefixed"):
                self.umbleng = umbleng
                umbsecleng = [umbleng]

            umbcomptab = {}

            for i, block_name in enumerate(compblocks):
                if compblocks[i] == "buoyancy":
                    umbwetmass[i] = -1.4 * preumb_record["item7"][1]
                else:
                    umbwetmass[i] = preumb_record["item7"][1]

                umbprops = UmpProps(
                    self._variables.preumb,
                    preumb_record["item6"][0],
                    umbsecleng[i],
                    preumb_record["item7"][0],
                    umbwetmass[i],
                    preumb_record["item5"][0],
                    preumb_record["item5"][1],
                )

                umbcomptab[block_name] = umbprops

            mlim = 5000
            # """ Catenary tolerance """
            tol = 0.01
            # """ Distance tolerance """
            disttol = 0.001
            # """ Number of segements along cable """
            numseg = 50
            flipzumb = [0.0 for row in range(0, numseg)]
            # """ Segment length """
            ds = umbleng / numseg
            umbxf = math.sqrt(
                (umbtopconn[0] - subcabconpts[0]) ** 2
                + (umbtopconn[1] - subcabconpts[1]) ** 2
            )
            umbzf = umbtopconn[2] - subcabconpts[2]

            Humb = [0.0 for row in range(numseg)]
            Vumb = [0.0 for row in range(numseg)]
            Tumb = [0.0 for row in range(numseg)]
            thetaumb = [0.0 for row in range(numseg)]
            xumb = [0.0 for row in range(numseg)]
            zumb = [0.0 for row in range(numseg)]
            leng = [0.0 for row in range(numseg)]

            # Approximate catenary profile used in first instance to estimate
            # top end loads
            if umbxf == 0:
                lambdacat = 1e6
            elif math.sqrt(umbxf**2 + umbzf**2) >= umbleng:
                lambdacat = 0.2
            else:
                lambdacat = math.sqrt(
                    3 * (((umbleng**2 - umbzf**2) / umbxf**2) - 1)
                )

            c1 = 0.5 * umbcomptab["hang off"].wet_mass * self._variables.gravity
            Hf = max(math.fabs(c1 * umbxf / lambdacat), tol)
            Vf = c1 * ((umbzf / math.tanh(lambdacat)) + umbleng)
            Tf = math.sqrt(Hf**2.0 + Vf**2.0)
            theta_0 = math.atan(Vf / Hf)
            Tumb[0] = Tf
            Humb[0] = Hf
            Vumb[0] = Vf
            thetaumb[0] = theta_0
            xumb[0] = 0.0
            zumb[0] = 0.0

            errumbzf = None
            errumbxf = None

            for m in range(0, mlim):
                if m >= 1:
                    assert errumbzf is not None
                    assert errumbxf is not None

                    if math.fabs(errumbzf) > disttol * umbzf:
                        if (
                            np.diff(zumb[k - 2 : k + 1 : 2]) == 0.0
                            and np.diff(zumb[k - 3 : k : 2]) == 0.0
                        ):
                            Tffactor = 0.0001
                        else:
                            Tffactor = 0.001
                        if errumbzf > 0.0:
                            Tumb[0] = Tumb[0] + Tffactor * Tf
                        elif errumbzf < 0.0:
                            Tumb[0] = Tumb[0] - Tffactor * Tf
                        Vumb[0] = math.sqrt(Tumb[0] ** 2.0 - Humb[0] ** 2.0)
                        thetaumb[0] = math.atan(Vumb[0] / Humb[0])

                    if math.fabs(errumbxf) > disttol * umbxf:
                        if errumbxf > 0.0:
                            Humb[0] = Humb[0] + 0.001 * Hf
                        elif errumbxf < 0.0:
                            Humb[0] = Humb[0] - 0.001 * Hf

                        Vumb[0] = Vumb[0]
                        thetaumb[0] = math.atan(Vumb[0] / Humb[0])
                        if Humb[0] < 0.0:
                            thetaumb[0] = math.pi / 2.0
                            Humb[0] = 0.0

                c2 = (
                    umbcomptab["hang off"].wet_mass
                    * self._variables.gravity
                    * ds
                )
                c3 = (
                    umbcomptab["buoyancy"].wet_mass
                    * self._variables.gravity
                    * ds
                )
                c4 = (
                    umbcomptab["decline"].wet_mass
                    * self._variables.gravity
                    * ds
                )

                for k in range(1, numseg):
                    leng[k] = ds * k
                    if leng[k] <= umbsecleng[0]:
                        # """ Hang-off section """
                        Vumb[k] = Vumb[k - 1] - c2
                        Humb[k] = Humb[k - 1]
                        thetaumb[k] = math.atan(Vumb[k] / Humb[k])
                        xumb[k] = xumb[k - 1] + ds * math.cos(thetaumb[k - 1])
                        zumb[k] = zumb[k - 1] + ds * math.sin(thetaumb[k - 1])
                    elif leng[k] > umbsecleng[0] and leng[k] <= sum(
                        umbsecleng[0:2]
                    ):
                        # """ Buoyancy section """
                        Vumb[k] = Vumb[k - 1] - c3
                        Humb[k] = Humb[k - 1]
                        thetaumb[k] = math.atan(Vumb[k] / Humb[k])
                        xumb[k] = xumb[k - 1] + ds * math.cos(thetaumb[k - 1])
                        zumb[k] = zumb[k - 1] + ds * math.sin(thetaumb[k - 1])
                    elif leng[k] > sum(umbsecleng[0:2]) and leng[k] <= umbleng:
                        # """ Decline section """
                        Vumb[k] = Vumb[k - 1] - c4
                        Humb[k] = Humb[k - 1]
                        thetaumb[k] = math.atan(Vumb[k] / Humb[k])
                        if Vumb[k] < 0.0:
                            thetaumb[k] = 0.0
                            Vumb[k] = 0.0
                        xumb[k] = xumb[k - 1] + ds * math.cos(thetaumb[k - 1])
                        zumb[k] = zumb[k - 1] + ds * math.sin(thetaumb[k - 1])
                    Tumb[k] = math.sqrt(Humb[k] ** 2.0 + Vumb[k] ** 2.0)
                    # """ Allow cable to embed by up to 0.5m """
                    if (
                        self._variables.systype in ("wavefixed", "tidefixed")
                        and zumb[k] > umbzf + 0.5
                    ):
                        umblengcheck = "False"
                        break
                    else:
                        umblengcheck = "True"

                errumbxf = umbxf - xumb[-1]
                errumbzf = umbzf - zumb[-1]
                if umblengcheck == "False":
                    break

                if (
                    math.fabs(errumbxf) < disttol * umbxf
                    and math.fabs(errumbzf) < disttol * umbzf
                ):
                    # logmsg = ('Solution found, umbilical length '
                    #           '{}').format(umbleng)
                    # module_logger.debug(logmsg)
                    break

            if umblengcheck == "True":
                break

        # """ Maximum tension """
        self.Tumbmax = max(Tumb)
        # """ Radius of curvature along umbilical (starting at device end) """
        dzdx = np.diff(zumb) / np.diff(xumb)
        d2zdx2 = np.diff(dzdx) / np.diff(xumb[:-1])
        umbradcurv = [
            abs(number)
            for number in (((1 + dzdx[0:-1] ** 2.0) ** 1.5) / d2zdx2)
        ]
        self.umbradmin = min(umbradcurv)

        for zind, zval in enumerate(zumb):
            flipzumb[zind] = umbtopconn[2] - zval

        return umbleng, xumb, flipzumb
