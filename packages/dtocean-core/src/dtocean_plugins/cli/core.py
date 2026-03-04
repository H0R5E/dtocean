import numpy as np
from cartopy.feature import AdaptiveScaler, NaturalEarthFeature

from dtocean_core.utils.config import init_config
from dtocean_core.utils.execute import main

from .shared import SmartFormatter

COLORS = {
    "land": np.array((240, 240, 220)) / 256.0,
    "land_alt1": np.array((220, 220, 220)) / 256.0,
    "water": np.array((152, 183, 226)) / 256.0,
}


def init(args):
    print("+ Downloading cartopy plot data...")

    auto_scaler = AdaptiveScaler("110m", (("50m", 50), ("10m", 15)))

    # Automatically scaled country boundaries
    NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        auto_scaler,
        edgecolor="black",
        facecolor="never",
    )

    # Automatically scaled coastline, including major islands
    NaturalEarthFeature(
        "physical",
        "coastline",
        auto_scaler,
        edgecolor="black",
        facecolor="never",
    )

    # Automatically scaled land polygons, including major islands.
    NaturalEarthFeature(
        "physical",
        "land",
        auto_scaler,
        edgecolor="none",
        facecolor=COLORS["land"],
        zorder=-1,
    )

    # Automatically scaled ocean polygons
    NaturalEarthFeature(
        "physical",
        "ocean",
        auto_scaler,
        edgecolor="none",
        facecolor=COLORS["water"],
        zorder=-1,
    )


def subcommand(subparser):
    parser = subparser.add_parser("core", help="dtocean-core utilities")
    sp = parser.add_subparsers(required=True)
    _setup_run(sp)
    _setup_config(sp)


def _setup_run(subparser):
    desStr = (
        "Execute DTOcean .dtop project files. By default, the next "
        "module scheduled is executed. All scheduled modules can also "
        "be run using the appropriate option. Completed simulations are "
        "saved to a new project file with '_complete' appended to the "
        "file path."
    )

    parser = subparser.add_parser(
        "run",
        description=desStr,
        help="run simulation",
    )

    parser.add_argument(
        "fpath", help=("path to DTOcean project file"), type=str
    )

    parser.add_argument(
        "-o",
        "--out",
        help=("specify output path for results file"),
        type=str,
        default=None,
    )

    parser.add_argument(
        "-f",
        "--full",
        help=("execute all scheduled modules"),
        action="store_true",
    )

    parser.add_argument(
        "-n", "--no-save", help=("do not save the results"), action="store_true"
    )

    parser.add_argument(
        "-w",
        "--warnings",
        help=("show tracebacks for all warnings"),
        action="store_true",
    )

    parser.add_argument(
        "-l",
        "--logging",
        help=("activate the DTOcean logging system"),
        action="store_true",
    )

    parser.set_defaults(func=lambda args: _main(args))


def _main(args):
    fpath = args.fpath
    rpath = args.out
    full = args.full
    warn = args.warnings
    no_save = args.no_save
    log = args.logging

    if no_save is True:
        save = False
    elif rpath is not None:
        save = rpath.strip()
    else:
        save = True

    main(fpath, save, full, warn, log)


def _setup_config(subparser):
    """Command line parser for init_config.

    Example:

        To get help::

            dtocean core config -h

    """

    desStr = (
        "Copy user modifiable configuration files to "
        r"<UserName>\AppData\Roaming\DTOcean\dtocean-core\config"
    )

    parser = subparser.add_parser(
        "config",
        description=desStr,
        help="copy config files",
        formatter_class=SmartFormatter,
    )

    parser.add_argument(
        "action",
        choices=["logging", "database"],
        help="R|Select an action, where\n"
        " logging = copy logging configuration\n"
        " database = copy database configuration",
    )

    parser.add_argument(
        "--overwrite",
        help=("overwrite any existing configuration files"),
        action="store_true",
    )

    parser.set_defaults(func=lambda args: _config(args))


def _config(args):
    action = args.action
    overwrite = args.overwrite

    kwargs = {
        "logging": False,
        "database": False,
        "overwrite": overwrite,
    }

    kwargs[action] = True
    dir_path = init_config(**kwargs)

    if dir_path is not None:
        print("Copying configuration files to {}".format(dir_path))
