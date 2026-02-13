# Contributing to DTOcean

This page describes the structure of the DTOcean monorepo, the tools used to
manage it and how you can contribute to the project.

## Repository Structure

This repository is structured as a monorepo, in that all of the individual
packages that make up the DTOcean suite of tools are all stored in a single
git repo. Details of the files and folders within the repository are given
in the tables below.

### Top-Level Files

The table below describes the purposes of the files found at the root level of
the repository.

| File            | Purpose                                                       |
| --------------- | ------------------------------------------------------------- |
| .codecov.yml    | Configuration for the [Codecov] test coverage service         |
| .gitattributes  | git attributes for files, used mainly for [LFS] configuration |
| .gitignore      | Lists files to be ignored (globally) by git                   |
| CHANGELOG.md    | A record of changes at root level or for docs[^1]             |
| CONTRIBUTING.md | This document                                                 |
| poetry.lock     | Global (and only tracked) [Poetry] lock file                  |
| pyproject.toml  | Global configuration file                                     |
| README.md       | Top level README[^2]                                          |

### Top-Level Folders

The table below describes the contents of the folders found at the root level
of the repository.

| Folder   | Purpose                                                     |
| -------- | ----------------------------------------------------------- |
| .github  | GitHub configuration files including Actions workflows      |
| .vscode  | Visual Studio Code workspace settings                       |
| docs     | Global documentation source in [Sphinx] format              |
| images   | Assets (e.g. images) for use in the top level README        |
| packages | Folders containing the Python packages that make up DTOcean |
| scripts  | Top level Python scripts, e.g., version calculation         |

### Package Folders

The table below describes the contents of the sub-folders found in the
`packages` folder.

| Folder                | Purpose                                                          |
| --------------------- | ---------------------------------------------------------------- |
| dtocean               | Meta package for installing the whole DTOcean suite              |
| dtocean-app           | Graphical representation of the dtocean-core functionality       |
| dtocean-core          | Main orchestration and data management module                    |
| dtocean-docs          | Provides an offline copy of the global documentation             |
| dtocean-dummy-module  | Basic engineering assessment module for testing dtocean-core     |
| dtocean-hydrodynamics | Device positioning and mechanical power calculation module       |
| dtocean-qt            | Support module providing specialised QT6 widgets for dtocean-app |
| mdo-engine            | General data management, execution ordering and plugin framework |
| polite-config         | Helper utilities for setting up logging, config files, etc.      |
| dtocean-economics     | Economic assessment module[^3]                                   |
| dtocean-electrical    | Electrical network design module[^3]                             |
| dtocean-environment   | Environmental impact assessment module[^3]                       |
| dtocean-installation  | Farm installation logistics design module[^3]                    |
| dtocean-logistics     | Support module used by other logistics modules[^3]               |
| dtocean-maintenance   | Operations and maintenance logistics design module[^3]           |
| dtocean-moorings      | Station keeping / foundation design module[^3]                   |
| dtocean-reliability   | Farm system reliability assessment module[^3]                    |

### Files and Folders in a Package

The table below describes the files and folder consistently found in a package
folder.

| File / Folder          | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| .vsode                 | Visual Studio Code workspace settings for the package      |
| src                    | Python source code                                         |
| tests                  | [pytest] test code                                         |
| .gitignore             | Lists files to be ignored by git within the package folder |
| CHANGELOG.md           | A record of changes for the package                        |
| LICENSES / LICENSE.txt | License(s) governing the use of the package code           |
| pyproject.toml         | Configuration file for the package                         |
| README.md              | Package specific README                                    |

Other folders containing examples (with names such as `examples`, `notebooks`,
etc.) or additional data for test or examples may or may not be included within
the package folders.

[^1]: Changes to packages are recorded in their own change logs

[^2]: Also used as the README for the `dtocean` meta package

[^3]: Not yet implemented in this version but planned (stub package)

## Tools

[Poetry](https://python-poetry.org/) is used to manage manage package
dependencies and builds, while the
[poetry-monoranger-plugin](https://github.com/ag14774/poetry-monoranger-plugin)
is used to ensure dependency compatibility between the packages and to replace
path dependencies with concrete version specifiers at build time.

## Setting up

## Contributing

[Codecov]: https://about.codecov.io/
[LFS]: https://git-lfs.com/
[Poetry]: https://python-poetry.org/
[Sphinx]: https://www.sphinx-doc.org/
[pytest]: https://docs.pytest.org/