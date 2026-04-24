import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------------
# 1️⃣ S3 setup
# --------------------------
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
bucket = "prd-tnm"
prefix = "StagedProducts/Elevation/1/TIFF/current/"

# --------------------------
# 2️⃣ North America bounds
# --------------------------
min_lat = 5
max_lat = 85
min_lon = -170
max_lon = -50

# --------------------------
# 3️⃣ Tile naming function
# --------------------------
def generate_tile_name(lat, lon):
    ns = "n" if lat >= 0 else "s"
    ew = "e" if lon >= 0 else "w"
    return f"{ns}{abs(lat):02d}{ew}{abs(lon):03d}"

# --------------------------
# 4️⃣ Generate tile list (NO shapefile)
# --------------------------
print("🌎 Generating North America tile list...")

tile_names = []
for lat in range(min_lat, max_lat):
    for lon in range(min_lon, max_lon):
        tile_names.append(generate_tile_name(lat, lon))

tile_set = set(tile_names)

print(f"✅ Total tiles in bounding box: {len(tile_set)}")

# --------------------------
# 5️⃣ Prepare local folder
# --------------------------
local_folder = os.path.join(os.path.dirname(__file__), "DEM_tiles_NA")
os.makedirs(local_folder, exist_ok=True)

# --------------------------
# 6️⃣ Fast tile filter
# --------------------------
def is_target_tile(key):
    if not key.endswith(".tif"):
        return False
    try:
        tile = key.split('/')[-2]  # folder like n37w122
        return tile in tile_set
    except:
        return False

# --------------------------
# 7️⃣ Threaded download setup
# --------------------------
MAX_WORKERS = 10  # adjust if needed

def download_tile(key):
    filename = os.path.join(local_folder, os.path.basename(key))

    if os.path.exists(filename):
        return None

    try:
        s3.download_file(bucket, key, filename)
        return f"✅ {filename}"
    except Exception as e:
        return f"❌ Failed: {key} ({e})"

# --------------------------
# 8️⃣ Download tiles
# --------------------------
paginator = s3.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

print("⬇️ Downloading tiles with threading...")

futures = []
count = 0

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']

            if is_target_tile(key):
                futures.append(executor.submit(download_tile, key))

    for future in as_completed(futures):
        result = future.result()
        if result:
            print(result)
            if result.startswith("✅"):
                count += 1

print(f"\n🎉 Total tiles downloaded: {count}")