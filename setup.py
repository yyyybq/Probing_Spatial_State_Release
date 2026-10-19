from setuptools import find_packages, setup

setup(
    name="vidfm3d",
    version="1.0",
    description="Diagnostic probes for spatial information in frozen video and vision foundation-model features",
    python_requires=">=3.11",
    author="Probing Spatial State authors",
    license=(
        "CC-BY-NC-4.0; vendored third-party code keeps its own license, "
        "see THIRD_PARTY.md"
    ),
    # Only root-level files here: setuptools < 77 copies license_files into
    # dist-info by basename, so the nested LICENSE files would overwrite the
    # root one. Those ship next to their code through package_data instead.
    license_files=["LICENSE", "THIRD_PARTY.md"],
    packages=find_packages(include=["vidfm3d", "vidfm3d.*"]),
    package_data={
        "vidfm3d.vggt": ["LICENSE.txt"],
        "vidfm3d.vggt.layers": ["LICENSE"],
        "vidfm3d.dust3r": ["LICENSE", "NOTICE"],
    },
)
