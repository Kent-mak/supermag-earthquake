import xarray as xr
import numpy as np
import pandas as pd

def get_station_coords(geomag: xr.Dataset) -> np.ndarray:
    glats = geomag["glat"].isel(time=0).values
    glons = geomag["glon"].isel(time=0).values

    coords = np.stack((glats, glons), axis=-1)
    return coords


def normalize_coordinates(coords: np.ndarray) -> np.ndarray:

    assert coords.shape[1] == 2, f"Invalid shape: {coords.shape}, expected (N, 2)"

    lat, lon = coords[:, 0], coords[:, 1]

    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("Latitude values must be between -90 and 90.")

    lon = ((lon + 180) % 360) - 180

    return np.column_stack((lat, lon))
