"""Calculate downstream beneficiaries.

Design:

"""
import argparse
import glob
import logging
import math
import os

from osgeo import gdal
from osgeo import osr
import ecoshard
import pygeoprocessing
import taskgraph

gdal.UseExceptions()

logging.basicConfig(
    level=logging.DEBUG,
#    filename='log.txt',
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.DEBUG)
logging.getLogger('pygeoprocessing').setLevel(logging.DEBUG)


DEM_ZIP_URL = 'https://storage.googleapis.com/global-invest-sdr-data/global_dem_3s_md5_22d0c3809af491fa09d03002bdf09748.zip'

WATERSHED_VECTOR_ZIP_URL = 'https://storage.googleapis.com/ipbes-ndr-ecoshard-data/watersheds_globe_HydroSHEDS_15arcseconds_blake2b_14ac9c77d2076d51b0258fd94d9378d4.zip'

POPULATION_RASTER_URL_MAP = {
    '2000': 'https://storage.googleapis.com/ecoshard-root/population/lspop2000.tif',
    '2017': 'https://storage.googleapis.com/ecoshard-root/population/lspop2017.tif'}

WORKSPACE_DIR = 'workspace'


def process_watershed(
        watershed_vector_path, watershed_fid, dem_path, pop_raster_path_list,
        target_beneficiaries_path_list):
    """Calculate downstream beneficiaries for this watershed.

    Args:
        watershed_vector_path (str): path to watershed vector
        watershed_fid (str): watershed FID to process
        dem_path (str): path to DEM vector
        pop_raster_path_list (list): list of population rasters to route
        target_beneficiaries_path_list (str): list of target downstream
            beneficiary rasters to create, parallel with
            `pop_raster_path_list`.

    Return:
        None.
    """
    job_id = f'''{os.path.basename(
        os.path.splitext(watershed_vector_path)[0])}_{watershed_fid}'''
    LOGGER.debug(f'create working directory for {job_id}')

    working_dir = os.path.join(
        os.path.dirname(target_beneficiaries_path_list[0]), job_id)
    try:
        os.makedirs(working_dir)
    except OSError:
        LOGGER.warning(f'{working_dir} already exists')

    LOGGER.info(f'clip/reproject DEM for {job_id}')
    watershed_info = pygeoprocessing.get_vector_info(watershed_vector_path)
    watershed_vector = gdal.OpenEx(watershed_vector_path, gdal.OF_VECTOR)
    watershed_layer = watershed_vector.GetLayer()
    watershed_feature = watershed_layer.GetFeature(watershed_fid)
    watershed_geom = watershed_feature.GetGeometryRef()
    watershed_centroid = watershed_geom.Centroid()
    utm_code = (
        math.floor((watershed_centroid.GetX() + 180)/6) % 60) + 1
    lat_code = 6 if watershed_centroid.GetY() > 0 else 7
    epsg_code = int('32%d%02d' % (lat_code, utm_code))
    epsg_sr = osr.SpatialReference()
    epsg_sr.ImportFromEPSG(epsg_code)
    LOGGER.debug(f'epsg: {epsg_code} for {job_id}')

    watershed_envelope = watershed_geom.GetEnvelope()
    LOGGER.debug(f'watershed_envelope: {watershed_envelope}')
    target_watershed_bb = pygeoprocessing.transform_bounding_box(
        watershed_envelope, watershed_info['projection_wkt'],
        epsg_sr.ExportToWkt())

    watershed_vector = None
    watershed_layer = None
    watershed_feature = None
    watershed_geom = None
    watershed_centroid = None
    watershed_envelope = None

    dem_info = pygeoprocessing.get_raster_info(dem_path)
    dem_gt = dem_info['geotransform']
    ul = gdal.ApplyGeoTransform(dem_gt, 0, 0)
    lr = gdal.ApplyGeoTransform(dem_gt, 1, 1)
    dem_pixel_bb = [
        min(ul[0], lr[0]),
        min(ul[1], lr[1]),
        max(ul[0], lr[0]),
        max(ul[1], lr[1])]

    target_pixel_bb = pygeoprocessing.transform_bounding_box(
        dem_pixel_bb, dem_info['projection_wkt'], epsg_sr.ExportToWkt())
    # x increases, y decreases
    target_pixel_size = [
        target_pixel_bb[2]-target_pixel_bb[0],
        target_pixel_bb[1]-target_pixel_bb[3],
        ]

    warped_dem_raster_path = os.path.join(working_dir, f'{job_id}_dem.tif')
    pygeoprocessing.warp_raster(
        dem_path, target_pixel_size, warped_dem_raster_path,
        'near', target_bb=target_watershed_bb,
        target_projection_wkt=epsg_sr.ExportToWkt(),
        vector_mask_options={
            'mask_vector_path': watershed_vector_path,
            'mask_vector_where_filter': f'"FID"={watershed_fid}'},
        gdal_warp_options=None, working_dir=working_dir)

    LOGGER.debug('route dem')
    for pop_raster_path, target_beneficiaries_path in zip(
            pop_raster_path_list, target_beneficiaries_path_list):
        LOGGER.debug(
            f'route downstream beneficiaries to '
            f'{target_beneficiaries_path_list}')


def main(watershed_id=None):
    """Entry point.

    Args:
        watershed_id (int?): if present, only run analysis on this watershed.

    Return:
        None.
    """
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 0, 15.0)

    dem_download_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(DEM_ZIP_URL)[0]))
    watershed_download_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(
            WATERSHED_VECTOR_ZIP_URL)[0]))
    population_download_dir = os.path.join(
        WORKSPACE_DIR, 'population_rasters')

    for dir_path in [
            dem_download_dir, watershed_download_dir,
            population_download_dir]:
        os.makedirs(dir_path, exist_ok=True)

    download_dem_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_ZIP_URL, dem_download_dir),
        task_name='download and unzip dem')

    dem_tile_dir = os.path.join(dem_download_dir, 'global_dem_3s')
    dem_vrt_path = os.path.join(
        dem_tile_dir,
        f'{os.path.basename(os.path.splitext(DEM_ZIP_URL)[0])}.vrt')
    LOGGER.debug(f'build vrt to {dem_vrt_path}')

    dem_tile_raster_list = [
        gdal.Open(path, gdal.OF_RASTER)
        for path in glob.glob(os.path.join(dem_tile_dir, '*.tif'))]

    task_graph.add_task(
        func=gdal.BuildVRT,
        args=(os.path.basename(dem_vrt_path), dem_tile_raster_list),
        target_path_list=[dem_vrt_path],
        dependent_task_list=[download_dem_task],
        task_name='build dem vrt')

    download_watershed_vector_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(WATERSHED_VECTOR_ZIP_URL, watershed_download_dir),
        task_name='download and unzip watershed vector')

    pop_raster_path_map = {}
    for pop_id, pop_url in POPULATION_RASTER_URL_MAP.items():
        pop_raster_path = os.path.join(
            population_download_dir, os.path.basename(pop_url))
        download_pop_raster = task_graph.add_task(
            func=ecoshard.download_url,
            args=(pop_url, pop_raster_path),
            target_path_list=[pop_raster_path],
            task_name=f'download {pop_url}')
        pop_raster_path_map[pop_id] = pop_raster_path

    LOGGER.info('wait for downloads to conclude')
    task_graph.join()

    watershed_root_dir = os.path.join(
        watershed_download_dir, 'watersheds_globe_HydroSHEDS_15arcseconds')

    if watershed_id:
        watershed_basename, watershed_fid = watershed_id.split(',')
        watershed_path = os.path.join(
            watershed_root_dir, f'{watershed_basename}.shp')

        process_watershed(
            watershed_path, watershed_fid, dem_vrt_path,
            pop_raster_path_map['2000'],
            f'''downstream_benficiaries_{watershed_basename}_{
                watershed_fid}.tif''')

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream beneficiaries')
    parser.add_argument(
        '--watershed_id', help='if present only run on this watershed id')
    args = parser.parse_args()

    main(watershed_id=args.watershed_id)
