"""On-disk maps for DeepEvolution.

One directory per map, in the same layout as ``utilities/maps/<name>``:
``<directory>/<name>.yaml``, ``<name>.png`` and ``<name>_wp.csv``.
Nothing outside this folder is registered, so ``--MAP_NAME`` still lists only
``utilities/maps``. Point ``MAP_PATH`` at one of these directories to load it.

``maps/<name>/``
    ``<name>.png``   occupancy bitmap. 0 is wall, 255 is free. The loader
                     flips the image vertically and treats row 0 as the
                     lowest y, so save it that way.
    ``<name>.yaml``  image, resolution [m/px], origin [x, y, yaw],
                     negate, occupied_thresh, free_thresh, and
                     ``starting_positions: [[x, y, yaw], ...]``.
    ``<name>_wp.csv``  raceline. Header:
                     s_m,x_m,y_m,psi_rad,kappa_radpm,vx_mps,ax_mps2,d_right_iqp,d_left_iqp

Optional, same names as the standard maps: ``centerline.csv``,
``<name>_wp_reverse.csv``, ``<name>_speed_scaling.csv``.
"""

from pathlib import Path

MAPS_DIR = Path(__file__).resolve().parent / "maps"


def map_dir(name: str) -> Path:
    return MAPS_DIR / name


def has_map(name: str) -> bool:
    folder = map_dir(name)
    return (folder / f"{name}.yaml").is_file()


def list_maps() -> list[str]:
    if not MAPS_DIR.is_dir():
        return []
    return sorted(
        entry.name
        for entry in MAPS_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / f"{entry.name}.yaml").is_file()
    )
