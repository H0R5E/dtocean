# Contributing to DTOcean

This page describes the structure of the DTOcean monorepo, the tools used to
manage it and how you can contribute to the project.

## Repository Structure

This repository is structured as a monorepo, in that all of the individual
packages that make up the DTOcean suite of tools are all stored in a single
git repo.

## Tools

[Poetry](https://python-poetry.org/) is used to manage manage package
dependencies and builds, while the
[poetry-monoranger-plugin](https://github.com/ag14774/poetry-monoranger-plugin)
is used to ensure dependency compatibility between the packages and to replace
path dependencies with concrete version specifiers at build time.

## Setting up

## Contributing
