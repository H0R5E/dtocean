# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

<!-- version list -->

## v2.0.1 (2026-02-11)

### Bug Fixes

- Bump version
  ([`df995ff`](https://github.com/H0R5E/dtocean/commit/df995ff9b20c007181a608d5ef6badd82fb35cf8))


## v2.0.0 (2026-02-06)

### Features

- Update install instructions
  ([`51b0d54`](https://github.com/H0R5E/dtocean/commit/51b0d540db41a85043088d4de8c74442da6cf819))

## v1.0.4 (2026-02-05)

### Bug Fixes

- Be stricter with python version for now
  ([`9b172b2`](https://github.com/H0R5E/dtocean/commit/9b172b202e8826393fe0e87b7ab4dca9780e1635))

## v1.0.3 (2026-02-05)

### Bug Fixes

- Force bump
  ([`eee3fa6`](https://github.com/H0R5E/dtocean/commit/eee3fa6caeee2372ee941bd798049d2df53cc72f))

## v1.0.2 (2026-02-05)

### Bug Fixes

- Don't add GitHub release
  ([`e110e7e`](https://github.com/H0R5E/dtocean/commit/e110e7e518aee499af73c9a332b627faab387adb))

## v1.0.1 (2026-02-05)

### Bug Fixes

- **polite-config**: Attempt an automated release
  ([`d40793f`](https://github.com/H0R5E/dtocean/commit/d40793f8524ffa44e30bbff1e997390f9725e8da))

## v0.10.3 - 2022-07-14

### Changed

- Made EtcDirectory guess the path based on the system architecture only.
  It won't raise an error now if it's not found.

## v0.10.2 - 2022-07-13

### Added

- Added EtcDirectory class to the paths module for locating the Python
  distribution's etc directory.

## v0.10.1 - 2019-07-08

### Changed

- Added Loader and Dumper arguments to pyyaml calls as required by pyyaml
  version 5.1 to improve safety.

## v0.10.0 - 2019-03-01

### Added

- Added dummy logging configuration file to config/logging.yaml and added
  usage example to README.
- Added Directory.list_files method to list the files in a directory if it
  exists.
- Added DirectoryMap.copy_all and DirectoryMap.safe_copy_all to copy all files
  in the source directory to the target directory.
- Directory objects now returns their path when printed.

### Changed

- Made Logger.configure_logger require the configuration dictionary as an
  input.

### Fixed

- Ensured that the target directory exists when calling ReadYAML.write().

## v0.9.0 - 2017-01-04

### Added

- Initial import of dtocean-gui from SETIS.
