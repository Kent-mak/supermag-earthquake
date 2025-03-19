# from supermag_eq import get_sample_ranges, build_dataset, save_ds_to_netcdf, load_eq_catalog, load_geomag_data
import pandas as pd
import supermag_eq as se
import numpy as np


def process_1_year(
        catalog_path: str, 
        geomag_path: str, 
        time_period: np.timedelta64,
        year: int,
        radius: float,
        save_dir: str):
    catalog =  se.load_eq_catalog(catalog_path)
    geomag = se.load_geomag_data(geomag_path)

    time_ranges_dict, pos_count, neg_count = se.get_sample_ranges(
        catalog=catalog,
        geomag=geomag,
        time_period=time_period,
        year=year,
        radius=radius
    )
    print(f"positives: {pos_count}")
    print(f"negatives: {neg_count}")

    # print(time_ranges_dict)

    dataset = se.build_dataset(time_ranges_dict, geomag=geomag, vars=[ "dbn_nez", "dbe_nez", "dbz_nez"])
    
    ds_path = se.save_ds_to_netcdf(dir=save_dir, year=year, ds=dataset)

    print(f"{year} dataset saved to {ds_path}")

    return 

if __name__=="__main__":
    # year = int(input("year:"))
    period = np.timedelta64(7, 'D')
    # rad = int(input("station radius:") )
    years = list(range(2000, 2025))
    print(years)

    for year in years:
        print(f"year: {year}")
        process_1_year(
            catalog_path=fr"D:\earthquake-prediction\data\catalog\csv\usgs\{year}.csv",
            geomag_path=fr"D:\earthquake-prediction\data\geomagnetic_data\supermag\all_stations_all{year}.netcdf",
            time_period=period,
            year=year,
            radius=200,
            save_dir=r"D:\earthquake-prediction\data\dataset\supermag"
        )