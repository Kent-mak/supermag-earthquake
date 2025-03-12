import xarray as xr
import pandas as pd

def load_geomag_data(netcdf_path: str) -> xr.Dataset:
    """Load geomagnetic data from a NetCDF file."""
    try:
        print("loading raw data...")
        ds = xr.open_dataset(netcdf_path, chunks="auto")
        print(ds)
        return ds
    except Exception as e:
        raise RuntimeError(f"Error loading NetCDF file {netcdf_path}: {e}")


def load_eq_catalog(csv_path: str) -> pd.DataFrame:
    """Load earthquake catalog from a CSV file."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Error loading earthquake catalog {csv_path}: {e}")


def load_station_info(csv_path: str) -> pd.DataFrame:
    """Load geomagnetic station information from a CSV file."""
    try:
        return pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Error loading station info {csv_path}: {e}")

if __name__=="__main__":
    geomag_path = r"D:\earthquake-prediction\data\geomagnetic_data\supermag\all_stations_all2024.netcdf"
    ds = load_geomag_data(geomag_path)
    print(ds)