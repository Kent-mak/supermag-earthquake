import os
import xarray as xr
import pandas as pd


def assign_time_as_coordinate(ds: xr.Dataset) -> xr.Dataset:
    time_as_datetime = pd.to_datetime({
        "year": ds["time_yr"].values,
        "month": ds["time_mo"].values,
        "day": ds["time_dy"].values,
        "hour": ds["time_hr"].values,
        "minute": ds["time_mt"].values
    })

    ds = ds.assign_coords(time=("block", time_as_datetime))
    ds = ds.drop_vars(["time_yr", "time_mo", "time_dy", "time_hr", "time_mt", "time_sc"])
    ds = ds.swap_dims({"block": "time"})
    
    return ds



def load_geomag_data(netcdf_path: str) -> xr.Dataset:
    """Load geomagnetic data from a NetCDF file."""
    try:
        print("loading raw data...")
        ds = xr.open_dataset(netcdf_path, chunks="auto")
        ds = assign_time_as_coordinate(ds)
        return ds
    except Exception as e:
        raise RuntimeError(f"Error loading NetCDF file {netcdf_path}: {e}")


def load_eq_catalog(csv_path: str) -> pd.DataFrame:
    """Load earthquake catalog from a CSV file."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Error loading earthquake catalog {csv_path}: {e}")



def save_ds_to_netcdf(dir: str, year: int, ds: xr.Dataset) -> str:
    os.makedirs(dir, exist_ok=True)
    filename = f"{year}_dataset.nc"
    filepath = os.path.join(dir, filename)
    ds.to_netcdf(filepath)
    return filepath
