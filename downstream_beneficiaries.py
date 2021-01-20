"""Calculate downstream beneficiaries.

Design:

"""
import argparse
import logging
import math
import os
import subprocess

from osgeo import gdal
from osgeo import osr
import ecoshard
import pygeoprocessing
import pygeoprocessing.routing
import taskgraph

gdal.UseExceptions()

logging.basicConfig(
    level=logging.DEBUG,
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


def _create_outlet_raster(
        outlet_vector_path, base_raster_path, target_outlet_raster_path):
    """Create a raster that has 1s where outlet exists and 0 everywhere else.

    Args:
        outlet_vector_path (str): path to input vector that has 'i', 'j'
            fields indicating which pixels are outlets
        base_raster_path (str): path to base raster used to create
            outlet raster shape/projection.
        target_outlet_raster_path (str): created by this call, contains 0s
            except where pixels intersect with an outlet.

    Return:
        None.
    """
    pygeoprocessing.new_raster_from_base(
        base_raster_path, target_outlet_raster_path, gdal.GDT_Byte,
        [0])

    outlet_raster = gdal.OpenEx(
        target_outlet_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    outlet_band = outlet_raster.GetRasterBand(1)

    outlet_vector = gdal.OpenEx(outlet_vector_path, gdal.OF_VECTOR)
    outlet_layer = outlet_vector.GetLayer()

    for outlet_feature in outlet_layer:
        outlet_band.WriteArray(
            [1],
            outlet_feature.GetField('i'),
            outlet_feature.GetField('j'),
            1, 1)
    outlet_band = None
    outlet_raster = None


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
    # swizzle the envelope order that by default is xmin/xmax/ymin/ymax
    target_watershed_bb = pygeoprocessing.transform_bounding_box(
        [watershed_envelope[i] for i in [0, 2, 1, 3]],
        watershed_info['projection_wkt'],
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
    # make sure we take the smallest side so our dem pixels are square
    target_pixel_side = min(
        abs(target_pixel_bb[2]-target_pixel_bb[0]),
        abs(target_pixel_bb[3]-target_pixel_bb[1]))
    target_pixel_size = (target_pixel_side, -target_pixel_side)

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
    filled_dem_raster_path = os.path.join(
        working_dir, f'{job_id}_filled_dem.tif')
    pygeoprocessing.routing.fill_pits(
        (warped_dem_raster_path, 1), filled_dem_raster_path,
        working_dir=working_dir)
    flow_dir_d8_raster_path = os.path.join(
        working_dir, f'{job_id}_flow_dir_d8.tif')
    pygeoprocessing.routing.flow_dir_d8(
        (filled_dem_raster_path, 1), flow_dir_d8_raster_path,
        working_dir=working_dir)

    outlet_vector_path = os.path.join(working_dir, 'outlet_vector.gpkg')
    pygeoprocessing.routing.detect_outlets(
        (flow_dir_d8_raster_path, 1), outlet_vector_path)

    outlet_raster_path = os.path.join(working_dir, 'outlet_raster.tif')
    _create_outlet_raster(
        outlet_vector_path, flow_dir_d8_raster_path, outlet_raster_path)

    for pop_raster_path, target_beneficiaries_path in zip(
            pop_raster_path_list, target_beneficiaries_path_list):
        LOGGER.debug(
            f'route downstream beneficiaries to '
            f'{target_beneficiaries_path_list}')

        aligned_pop_raster_path = os.path.join(
            working_dir,
            f'{job_id}_{os.path.basename(os.path.splitext(pop_raster_path)[0])}.tif')

        pygeoprocessing.warp_raster(
            pop_raster_path, target_pixel_size, aligned_pop_raster_path,
            'near', target_bb=target_watershed_bb,
            target_projection_wkt=epsg_sr.ExportToWkt(),
            vector_mask_options={
                'mask_vector_path': watershed_vector_path,
                'mask_vector_where_filter': f'"FID"={watershed_fid}'},
            working_dir=working_dir)

        pygeoprocessing.distance_to_channel_d8(
            (flow_dir_d8_raster_path, 1), (outlet_raster_path, 1),
            target_beneficiaries_path,
            weight_raster_path_band=(pop_raster_path, 1))


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

    task_graph.add_task(
        func=subprocess.run,
        args=(f'gdalbuildvrt {dem_vrt_path} {dem_tile_dir}/*.tif',),
        kwargs={'shell': True, 'check': True},
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
            watershed_path, int(watershed_fid), dem_vrt_path,
            [pop_raster_path_map['2000'],
             pop_raster_path_map['2017']],
            [f'''downstream_benficiaries_2000_{watershed_basename}_{
                 watershed_fid}.tif''',
             f'''downstream_benficiaries_2017_{watershed_basename}_{
                 watershed_fid}.tif'''])

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream beneficiaries')
    parser.add_argument(
        '--watershed_id', help='if present only run on this watershed id')
    args = parser.parse_args()

    main(watershed_id=args.watershed_id)

