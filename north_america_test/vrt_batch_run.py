"""
vrt_batch_run.py
Test version: runs TauDEM pipeline on a small sample of VRTs.
Use --n to control sample size (default 5).
"""

import os
import sys
import re
import time
import argparse
import subprocess
import tempfile
import logging
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize, geometry_window
from rasterio.mask import mask as rio_mask
from shapely.geometry import box
import pickle

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# !! FOR GOOGLE CLOUD: update these paths to your GCS bucket mount point
#    e.g. /gcs/your-bucket-name/north_america/...
#    or use gsutil/gcsfs if not using gcsfuse

VRT_DIR    = r"C:\Users\lukemr\PycharmProjects\PythonProject\north_america_test\Buffered_VRTs"
STREAMS_DIR = r"C:\Users\lukemr\PycharmProjects\PythonProject\north_america_test\streams"
BASINS_DIR  = r"C:\Users\lukemr\PycharmProjects\PythonProject\north_america_test\basins"

OUTPUT_CSV = "tc_zonal_stats_test.csv"
BASIN_ID_FIELD = "linkno"

# !! FOR GOOGLE CLOUD: scale this up to match your VM's core count
#    e.g. on an n2-highcpu-32 you'd want 28-30 here
TAUDEM_PROCESSES = 4

TAUDEM_DIR = r"C:\Program Files\TauDEM\TauDEM5Exe"
MPIEXEC    = r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe"

STREAMS_INDEX_CACHE = "streams_index.pkl"
BASINS_INDEX_CACHE = "basins_index.pkl"

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

def save_index(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_index(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------------------
# SAFE PARQUET LOADER
# ---------------------------------------------------------------------------
def safe_read_parquet(path):
    gdf = gpd.read_parquet(path)
    if "geometry" not in gdf.columns:
        raise ValueError(f"No geometry column in {path}")
    return gdf.set_geometry("geometry")

# ---------------------------------------------------------------------------
# TAUDEM RUNNER
# ---------------------------------------------------------------------------
def run_taudem(cmd: list, desc: str):
    log.info(f"    Running TauDEM: {desc}")
    env = os.environ.copy()
    env["PATH"] = TAUDEM_DIR + ";" + env["PATH"]

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)

    if result.returncode != 0:
        log.error(f"TauDEM failed ({desc}):\n{result.stderr}")
        raise RuntimeError(f"TauDEM step failed: {desc}")

    log.info(f"    OK: {desc}")
    return result

def reproject_raster(in_path, out_path, dst_crs="ESRI:54034"):
    cmd = [
        "gdalwarp",
        "-t_srs", dst_crs,
        "-r", "bilinear",
        "-overwrite",
        in_path,
        out_path
    ]
    subprocess.run(cmd, check=True)

# ---------------------------------------------------------------------------
# SPATIAL INDEX
# ---------------------------------------------------------------------------
def build_spatial_index(streams_dir, basins_dir):
    def extract_vpu(path):
        match = re.search(r"(\d+)", path.stem)
        return match.group(1) if match else path.stem

    def bbox_from_gdf(gdf):
        b = gdf.total_bounds
        return box(b[0], b[1], b[2], b[3])

    log.info("Building spatial index for streams...")
    streams_index = {}
    for p in Path(streams_dir).glob("*.gpkg"):
        try:
            gdf = gpd.read_file(p)
            streams_index[extract_vpu(p)] = {
                "path": p,
                "bbox": bbox_from_gdf(gdf),
                "crs": gdf.crs,
            }
        except Exception as e:
            log.warning(f"  Skipping bad stream file {p.name}: {e}")

    log.info(f"  Found {len(streams_index)} stream VPUs")

    log.info("Building spatial index for basins...")
    basins_index = {}
    for p in Path(basins_dir).glob("*.parquet"):
        try:
            gdf = safe_read_parquet(p)
            basins_index[extract_vpu(p)] = {
                "path": p,
                "bbox": bbox_from_gdf(gdf),
                "crs": gdf.crs,
            }
        except Exception as e:
            log.warning(f"  Skipping bad basin file {p.name}: {e}")

    log.info(f"  Found {len(basins_index)} basin VPUs")

    return streams_index, basins_index

def find_overlapping_vpus(vrt_box_wgs84, index):
    matches = []
    for v, info in index.items():
        try:
            bbox_wgs84 = gpd.GeoSeries([info["bbox"]], crs=info["crs"]).to_crs("EPSG:4326").iloc[0]
            if bbox_wgs84.intersects(vrt_box_wgs84):
                matches.append(v)
        except Exception as e:
            log.warning(f"  Could not reproject bbox for VPU {v}: {e}")
    return matches

def load_vpus(vpus, index, loader):
    gdfs = []
    for v in vpus:
        try:
            gdfs.append(loader(index[v]["path"]))
        except Exception as e:
            log.warning(f"  Failed to load VPU {v}: {e}")

    if not gdfs:
        return gpd.GeoDataFrame()

    return pd.concat(gdfs, ignore_index=True)

# ---------------------------------------------------------------------------
# RASTERIZE STREAMS
# ---------------------------------------------------------------------------
def rasterize_streams(streams_gdf, ref_raster, out_path):
    with rasterio.open(ref_raster) as src:
        meta = src.meta.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs

    if streams_gdf.crs != crs:
        streams_gdf = streams_gdf.to_crs(crs)

    burned = rasterize(
        [(g, 1) for g in streams_gdf.geometry if g is not None],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    meta.update(dtype=rasterio.uint8, count=1, nodata=0, driver="GTiff")

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(burned, 1)

# ---------------------------------------------------------------------------
# ZONAL STATS
# ---------------------------------------------------------------------------
def compute_zonal_stats(basins, dist_path, dem_path, vrt_name):
    records = []

    with rasterio.open(dist_path) as dist_src, rasterio.open(dem_path) as dem_src:
        basins_proj = basins.to_crs(dist_src.crs)

        # Clip to raster extent first so we don't iterate thousands of basins outside the tile
        raster_box = box(
            dist_src.bounds.left,
            dist_src.bounds.bottom,
            dist_src.bounds.right,
            dist_src.bounds.top,
        )
        basins_proj = basins_proj[basins_proj.geometry.intersects(raster_box)].copy()
        log.info(f"  Basins after clipping to raster extent: {len(basins_proj)}")

        if basins_proj.empty:
            log.warning("  No basins intersect raster extent — skipping zonal stats")
            return pd.DataFrame()

        for _, basin in basins_proj.iterrows():
            geom = basin.geometry
            if geom is None or geom.is_empty:
                continue

            geom_geojson = [geom.__geo_interface__]

            # --- DISTANCE ---
            try:
                dist_data, _ = rio_mask(dist_src, geom_geojson, crop=True)
                dist_arr = dist_data[0].astype(float)
                dist_arr[dist_arr == dist_src.nodata] = np.nan
                max_dist = float(np.nanmax(dist_arr))
            except:
                max_dist = np.nan

            # --- DEM ---
            try:
                dem_data, _ = rio_mask(dem_src, geom_geojson, crop=True)
                dem_arr = dem_data[0].astype(float)
                dem_arr[dem_arr == dem_src.nodata] = np.nan
                max_elev = float(np.nanmax(dem_arr))
                min_elev = float(np.nanmin(dem_arr))
                mean_elev = float(np.nanmean(dem_arr))
            except:
                max_elev = min_elev = mean_elev = np.nan

            records.append({
                "vrt": vrt_name,
                BASIN_ID_FIELD: basin[BASIN_ID_FIELD],
                "area_km2": basin.get("area_km2", np.nan),
                "max_dist_to_stream": max_dist,
                "max_elev": max_elev,
                "min_elev": min_elev,
                "mean_elev": mean_elev,
            })

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# MAIN PROCESS (single VRT)
# ---------------------------------------------------------------------------
def process_vrt(vrt_path, streams_index, basins_index, output_csv):
    vrt_name = Path(vrt_path).stem
    t0 = time.time()
    log.info(f"{'='*60}")
    log.info(f"Processing: {vrt_name}")

    with rasterio.open(vrt_path) as src:
        bounds = src.bounds
        crs = src.crs

    vrt_box = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    vrt_box_wgs84 = gpd.GeoSeries([vrt_box], crs=crs).to_crs("EPSG:4326").iloc[0]

    stream_vpus = find_overlapping_vpus(vrt_box_wgs84, streams_index)
    basin_vpus  = find_overlapping_vpus(vrt_box_wgs84, basins_index)

    log.info(f"  Matched {len(stream_vpus)} stream VPUs, {len(basin_vpus)} basin VPUs")

    if not stream_vpus or not basin_vpus:
        log.warning(f"  No overlapping VPUs for {vrt_name} — skipping")
        return

    streams = load_vpus(stream_vpus, streams_index, gpd.read_file)
    basins  = load_vpus(basin_vpus, basins_index, safe_read_parquet)

    if streams.empty or basins.empty:
        log.warning(f"  Empty streams or basins for {vrt_name} — skipping")
        return

    with tempfile.TemporaryDirectory() as tmp:
        # -----------------------------
        # 🔁 Reproject DEM FIRST
        # -----------------------------
        dem_54034 = os.path.join(tmp, "dem_54034.tif")

        log.info("    Reprojecting DEM to ESRI:54034...")
        reproject_raster(vrt_path, dem_54034)

        # Get CRS from reprojected DEM
        with rasterio.open(dem_54034) as dem_src:
            target_crs = dem_src.crs

        # -----------------------------
        # 🔁 Reproject vectors
        # -----------------------------
        streams = streams.to_crs(target_crs)
        basins  = basins.to_crs(target_crs)

        # -----------------------------
        # Compute basin area (km^2)
        # -----------------------------
        if "area_km2" not in basins.columns:
            log.info("  Computing basin areas (km^2)...")
            basins["area_km2"] = basins.geometry.area / 1e6

        # -----------------------------
        # TauDEM outputs
        # -----------------------------
        fel   = os.path.join(tmp, "fel.tif")
        p     = os.path.join(tmp, "p.tif")
        sd8   = os.path.join(tmp, "sd8.tif")
        str_r = os.path.join(tmp, "streams.tif")
        dist  = os.path.join(tmp, "dist.tif")

        # -----------------------------
        # TauDEM pipeline
        # -----------------------------
        t_step = time.time()
        run_taudem(
            [MPIEXEC, "-n", str(TAUDEM_PROCESSES),
             os.path.join(TAUDEM_DIR, "pitremove.exe"),
             "-z", dem_54034, "-fel", fel],
            "Pit Remove",
        )
        log.info(f"    Pit Remove took {time.time() - t_step:.1f}s")

        t_step = time.time()
        run_taudem(
            [MPIEXEC, "-n", str(TAUDEM_PROCESSES),
             os.path.join(TAUDEM_DIR, "d8flowdir.exe"),
             "-fel", fel, "-p", p, "-sd8", sd8],
            "Flow Dir",
        )
        log.info(f"    Flow Dir took {time.time() - t_step:.1f}s")

        t_step = time.time()
        rasterize_streams(streams, dem_54034, str_r)
        log.info(f"    Rasterize streams took {time.time() - t_step:.1f}s")

        t_step = time.time()
        run_taudem(
            [MPIEXEC, "-n", str(TAUDEM_PROCESSES),
             os.path.join(TAUDEM_DIR, "d8hdisttostrm.exe"),
             "-p", p, "-src", str_r, "-dist", dist, "-thresh", "1"],
            "Distance to Stream",
        )
        log.info(f"    Distance to Stream took {time.time() - t_step:.1f}s")

        # -----------------------------
        # Zonal stats
        # -----------------------------
        t_step = time.time()
        df = compute_zonal_stats(basins, dist, dem_54034, vrt_name)
        log.info(f"    Zonal stats took {time.time() - t_step:.1f}s — {len(df)} basins")

        write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        df.to_csv(output_csv, mode="a", header=write_header, index=False)
        log.info(f"  CSV written to: {os.path.abspath(output_csv)}")

    elapsed = time.time() - t0
    log.info(f"  Finished {vrt_name} in {elapsed:.1f}s ({elapsed/60:.1f} min)")

# ---------------------------------------------------------------------------
# ENTRY
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="TauDEM batch pipeline (test mode)")
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of VRTs to process (default: 5)"
    )
    parser.add_argument(
        "--vrt", type=str, default=None,
        help="Run a specific VRT by name, e.g. --vrt USGS_1_n15w093"
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of spatial index cache"
    )
    args = parser.parse_args()

    log.info(f"TauDEM Test Run — processing up to {args.n} VRTs")
    log.info(f"  VRT dir:     {VRT_DIR}")
    log.info(f"  Streams dir: {STREAMS_DIR}")
    log.info(f"  Basins dir:  {BASINS_DIR}")
    log.info(f"  Output CSV:  {OUTPUT_CSV}")

    if (
            not args.rebuild_index
            and os.path.exists(STREAMS_INDEX_CACHE)
            and os.path.exists(BASINS_INDEX_CACHE)
    ):
        log.info("Loading cached spatial indexes...")
        streams_index = load_index(STREAMS_INDEX_CACHE)
        basins_index = load_index(BASINS_INDEX_CACHE)
    else:
        log.info("Building spatial indexes...")
        streams_index, basins_index = build_spatial_index(STREAMS_DIR, BASINS_DIR)

        log.info("Saving spatial index cache...")
        save_index(streams_index, STREAMS_INDEX_CACHE)
        save_index(basins_index, BASINS_INDEX_CACHE)

    if args.vrt:
        # allow with or without the .vrt extension
        vrt_name = args.vrt if args.vrt.endswith(".vrt") else args.vrt + ".vrt"
        vrts = [Path(VRT_DIR) / vrt_name]
        if not vrts[0].exists():
            log.error(f"VRT not found: {vrts[0]}")
            sys.exit(1)
    else:
        vrts = sorted(Path(VRT_DIR).glob("*.vrt"))[:args.n]

    if not vrts:
        log.error(f"No VRT files found in {VRT_DIR}")
        sys.exit(1)

    log.info(f"Found {len(vrts)} VRT(s) to process")

    t_total = time.time()
    succeeded, failed = 0, 0

    for v in vrts:
        try:
            process_vrt(str(v), streams_index, basins_index, OUTPUT_CSV)
            succeeded += 1
        except Exception:
            log.exception(f"Failed on {v.name}")
            failed += 1

    total_elapsed = time.time() - t_total
    log.info(f"{'='*60}")
    log.info(f"Done. {succeeded} succeeded, {failed} failed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    if succeeded > 0:
        avg = total_elapsed / succeeded
        log.info(f"Avg per VRT: {avg:.1f}s — estimated full run for 100 VRTs: {avg*100/3600:.1f} hrs")

if __name__ == "__main__":
    main()