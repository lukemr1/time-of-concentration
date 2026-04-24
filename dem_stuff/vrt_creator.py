from osgeo import gdal
import geopandas as gpd
from shapely.geometry import box
import logging
import sys
import os
import glob
import tqdm
import multiprocessing as mp
from functools import cache, partial

gdal.UseExceptions()

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_ds_bbox(ds: gdal.Dataset) -> tuple[float, float, float, float]:
    """
    Get bounds of a GDAL dataset as (minx, miny, maxx, maxy) in EPSG:4326
    """
    gt = ds.GetGeoTransform()
    width = ds.RasterXSize
    height = ds.RasterYSize
    projection = ds.GetProjection()

    return get_bbox_from_ds_data(gt, width, height, projection)


def bounds_intersect(bounds1: tuple[float, float, float, float], bounds2: tuple[float, float, float, float]) -> bool:
    """
    Check if two bounding boxes intersect.
    """
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2

    # Return True if they intersect, False otherwise
    return not (maxx1 < minx2 or minx1 > maxx2 or maxy1 < miny2 or miny1 > maxy2)


def get_bbox_from_ds_data(gt: tuple[float, ...], width: int, height: int, projection: str = None) -> tuple[
    float, float, float, float]:
    """
    Get bounds of a GDAL dataset as (minx, miny, maxx, maxy) in EPSG:4326
    """
    minx = gt[0]
    maxx = gt[0] + width * gt[1]
    miny = gt[3] + height * gt[5]
    maxy = gt[3]
    if miny > maxy:
        miny, maxy = maxy, miny

    # if projection:
    #     minx, miny, maxx, maxy = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=projection).to_crs(
    #         "EPSG:4326").total_bounds

    return (minx, miny, maxx, maxy)

@cache
def get_raster_bbox(raster_path: str) -> tuple[float, float, float, float]:
    ds: gdal.Dataset = gdal.Open(raster_path)
    return get_ds_bbox(ds)


def get_rasters_in_extent(bounds: list[float], rasters: list[str]) -> list[str]:
    output = []
    for raster in rasters:
        raster_bounds = get_raster_bbox(raster)
        if bounds_intersect(bounds, raster_bounds):
            output.append(raster)

    return output

@cache
def get_raster_res(raster_path: str) -> tuple[float, float]:
    ds: gdal.Dataset = gdal.Open(raster_path)
    gt = ds.GetGeoTransform()
    return (abs(gt[1]), abs(gt[5]))


def buffer_dem(dem: str, output_dem: str, all_dems: list[str], buffer_distance: float = 0.1,
               as_vrt: bool = True) -> str:
    """Expand a DEM tile by ``buffer_distance`` degrees using neighboring rasters."""
    minx, miny, maxx, maxy = get_raster_bbox(dem)

    minx -= buffer_distance
    maxx += buffer_distance
    miny -= buffer_distance
    maxy += buffer_distance

    surrounding_dems = get_rasters_in_extent((minx, miny, maxx, maxy), all_dems)
    if dem not in surrounding_dems:
        raise ValueError("The original DEM is not in the candidates list.")

    if as_vrt and not output_dem.lower().endswith('.vrt'):
        LOG.warning(
            "Output file does not have .vrt extension, but as_vrt is True. Proceeding to create a VRT file regardless.")
    if not as_vrt and output_dem.lower().endswith('.vrt'):
        LOG.warning(
            "Output file has .vrt extension, but as_vrt is False. Proceeding to create a non-VRT file regardless.")

    xres, yres = get_raster_res(dem)
    if as_vrt:
        vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest',
                                           outputBounds=(minx, miny, maxx, maxy),
                                           targetAlignedPixels=True,
                                           xRes=xres,
                                           yRes=yres,
                                           outputSRS='EPSG:4269')
        gdal.BuildVRT(output_dem, surrounding_dems, options=vrt_options)
    else:
        warp_options = gdal.WarpOptions(resampleAlg='nearest',
                                        outputBounds=(minx, miny, maxx, maxy),
                                        targetAlignedPixels=True,
                                        xRes=xres,
                                        yRes=yres)
        gdal.Warp(output_dem, surrounding_dems, options=warp_options)

    return output_dem

def buffer_dem_wrapper(args, buffer_distance, all_dems):
    buffer_dem(*args, buffer_distance=buffer_distance, all_dems=all_dems)

if __name__ == "__main__":
    dems_folder = r"C:\Users\lukemr\PycharmProjects\PythonProject\dem_stuff\DEM_tiles_NA"
    dems = glob.glob(os.path.join(dems_folder, "*.tif"))
    if not dems:
        print("No DEM files found.")
        sys.exit()

    output_folder = "Buffered_VRTs"
    os.makedirs(output_folder, exist_ok=True)

    args = []
    for dem_path in tqdm.tqdm(dems):

        filename = os.path.basename(dem_path)

        output_name = os.path.join(output_folder, filename.replace(".tif", ".vrt"))
        args.append((dem_path, output_name))
        buffer_dem(
            dem=dem_path,
            output_dem=output_name,
            all_dems=dems,
            buffer_distance=0.2,
            as_vrt=True
        )

    # with mp.Pool() as pool:
    #     for _ in tqdm.tqdm(pool.imap_unordered(partial(buffer_dem_wrapper, buffer_distance=0.2, all_dems=dems), args, chunksize=10), total=len(args)):
    #         pass