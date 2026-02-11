# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- version list -->

## v1.1.3 (2026-02-11)

### Bug Fixes

- Redo includes and excludes in poetry configs
  ([`6f14ee7`](https://github.com/H0R5E/dtocean/commit/6f14ee75ea21f51343fb59dd383c9b924bb5c430))


## v1.1.2 (2026-02-11)

### Bug Fixes

- Bump version
  ([`695a1d9`](https://github.com/H0R5E/dtocean/commit/695a1d901df047f499734a0a9ce6227fb51254fe))


## v1.1.1 (2026-02-11)

### Bug Fixes

- Enable project deps in pre-build
  ([`60cdb79`](https://github.com/H0R5E/dtocean/commit/60cdb79307592f3987843c82d7f066d85a28ad21))


## v1.1.0 (2026-02-11)

### Features

- Test multi-arch publish
  ([`61002c5`](https://github.com/H0R5E/dtocean/commit/61002c5d88b649fd573f0dc874c7ff8138bc25c7))


## v1.0.1 (2026-02-09)

### Bug Fixes

- Add changelog pragmas
  ([`e29140c`](https://github.com/H0R5E/dtocean/commit/e29140cf1bd06cab1108f5c023af347d179db062))

## v0.10.1 - 2022-04-12

### Changed

- Use the appveyor configuration file as the single source for the version
  number.

### Removed

- Removed `__build__` and `__version__` dunders.

### Fixed

- Fixed minimum pandas dependency version.
- Fixed codacy configuration.

## v0.10.0 - 2019-03-12

### Added

- Added change log.
- Added CI files
- Added python-magic as a dependency.
- Added pagination to DataFrameModel to accelerate loading times.

### Removed

- Removed unused ui module (which contained a copy of easygui) and german
  translations.
- Removed packaged libmagic library due to version conflicts.

### Fixed

- Fixed incorrect use of Pandas ix method which read wrong table rows.
- Fixed reference to pandasqt in MANIFEST.in which caused crash.
- Fixed various bugs with use of QVariant v2 API, which is not default for
  Python 2.
- Fixed bugs with comparison of QStrings to Python strings.
- Fixed incorrect format (python 3) for validator return value in remove
  column dialog of a DataTable widget.
- Fixed various depreciated pandas API issues.

## v0.9.0 - 2017-02-23

### Added

- Initial import of pandas-qt from SETIS.

### Changed

- Changed package name to dtocean-qt.
