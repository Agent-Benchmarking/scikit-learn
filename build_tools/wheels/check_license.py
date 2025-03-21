"""Checks the bundled license is installed with the wheel."""

import platform
import site
import os
from itertools import chain
from pathlib import Path

site_packages = site.getsitepackages()
print(f"Site packages: {site_packages}")

site_packages_path = (Path(p) for p in site_packages)

try:
    distinfo_path = next(
        chain(
            s
            for site_package in site_packages_path
            for s in site_package.glob("scikit_learn-*.dist-info")
        )
    )
    print(f"Found dist-info path: {distinfo_path}")
except StopIteration as e:
    raise RuntimeError("Unable to find scikit-learn's dist-info") from e

license_file = distinfo_path / "COPYING"
print(f"License file path: {license_file}")
print(f"License file exists: {license_file.exists()}")

license_text = license_file.read_text()
print(f"License text length: {len(license_text)}")
print(f"First 100 chars: {license_text[:100]}")
print(f"Platform system: {platform.system()}")

# Check if the system environment variable is available
runner_os = os.environ.get('RUNNER_OS', 'Unknown')
print(f"RUNNER_OS environment variable: {runner_os}")

assert "Copyright (c)" in license_text, "Missing copyright information in license"

expected_text = "This binary distribution of scikit-learn also bundles the following software"
assert expected_text in license_text, f"Unable to find bundled license for {platform.system()}"

print("License check passed successfully!")
