[![dtocean actions](https://github.com/DTOcean/dtocean/actions/workflows/dtocean.yml/badge.svg?branch=main)](https://github.com/DTOcean/dtocean/actions/workflows/dtocean.yml)
[![codecov](https://codecov.io/gh/DTOcean/dtocean/graph/badge.svg?token=Y3GR22fUJ8)](https://codecov.io/gh/DTOcean/dtocean)

***

:loudspeaker: This project is currently in beta while I finish converting the
Python2 modules to Python3. Please consider [sponsoring my
work](https://github.com/sponsors/H0R5E). Many thanks.

***

<p align="center">
  <img width="385" height="125" src="https://media.githubusercontent.com/media/H0R5E/dtocean/refs/heads/main/images/dtocean2plus._padded.png">
</p>

# DTOcean

**DTOcean is an open-source tool for design and techno-economic assessment of
marine renewable energy arrays.**

DTOcean can calculate:

- Optimal ocean energy converter (OEC) positioning
- <s>Energy export infrastructure</s>
- <s>Station keeping requirements based on OEC performance and site conditions</s>
- <s>Installation planning with weather effects</s>
- <s>Maintenance planning, simulating OEC downtime</s>
- <s>Environmental impact assessment (experimental)</s>

And features include:

- <s>A unique statistical approach to calculating levelized cost of energy (LCOE)</s>
- <s>OEC reliability influenced at component level</s>
- Graphical user interface
- Persistent database

(Features currently unavailable but planned for reimplementation are struck out.)

## Installation

Install the DTOcean suite of packages using pip:

```sh
pip install dtocean
```

After installation, download necessary data files and setup desktop shortcuts
using the following command:

```sh
dtocean init
```

## Database

See the [dtocean-database](https://github.com/DTOcean/dtocean-database)
repository for download and installation instructions.

## Usage

Open the DTOcean GUI using the DTOcean shortcut or from a command prompt:

```sh
dtocean app
```

### Command Line Tools

The `dtocean` command provides universal access to the command line tools for
all of the install modules. Documentation for each subcommand can be found in
the table below.

| subcommand    | help                                                     |
| ------------- | -------------------------------------------------------- |
| init          | Run module initialization (requires internet connection) |
| app           | See [dtocean-app]                                        |
| core          | See [dtocean-core]                                       |
| database      | See [dtocean-core]                                       |
| docs          | See [dtocean-docs]                                       |
| hydrodynamics | See [dtocean-hydrodynamics]                              |

### Example Files

The `examples` archive (in zip or tar.gz format) can be downloaded from the
[latest release](https://github.com/DTOcean/dtocean-examples/releases/latest)
of the [dtocean-examples](https://github.com/DTOcean/dtocean-examples/)
repository. See the "Getting Started 1: Example Project" chapter of the
[docs](https://dtocean.github.io/dtocean) for usage instructions.

## Documentation

See [https://dtocean.github.io/dtocean](https://dtocean.github.io/dtocean) for
the latest documentation. The documentation can also be accessed from the
`Help` menu of the GUI using the `Index...` command or using the `dtocean docs`
command line tool.

Various video tutorials can also be found on the Data Only Greater
[YouTube Channel](https://www.youtube.com/@dataonlygreater).

## Credits

<img align="left" width="301" height="159" src="https://media.githubusercontent.com/media/H0R5E/dtocean/refs/heads/main/images/dog_logo_wide_300.png">

This version of DTOcean was developed and published by Mathew Topper at [Data
Only Greater](https://www.dataonlygreater.com/) as a continuation of the
[EU FP7 DTOcean project](https://cordis.europa.eu/project/id/608597).

Also, please check out the [EU H2020 DTOceanPlus project](https://cinea.ec.europa.eu/featured-projects/dtoceanplus_en), which
expanded the scope of the DTOcean tools. The source code for DTOceanPlus is
available [here](https://gitlab.com/dtoceanplus).

## Licence and Attributions

[GPL-3.0](https://choosealicense.com/licenses/gpl-3.0/)

Other licenses may apply to individual components. Please see the source code
for full licensing information.

While not required by the terms of the license, if you would like to
acknowledge the use of DTOcean in a publication, please cite:

> Topper, M. B. R., Nava, V., Collin, A. J., Bould, D., Ferri, F., Olson, S. S.,
> ... & Jeffrey, H. F. (2019). Reducing variability in the cost of energy of
> ocean energy arrays. Renewable and Sustainable Energy Reviews, 112, 263-279.
> Retrieved from https://authors.elsevier.com/sd/article/S1364032119303454

[dtocean-app]: https://pypi.org/project/dtocean-app/
[dtocean-core]: https://pypi.org/project/dtocean-core/
[dtocean-docs]: https://pypi.org/project/dtocean-docs/
[dtocean-hydrodynamics]: https://pypi.org/project/dtocean-hydrodynamics/
