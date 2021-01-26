"""Calculate downstream beneficiaries.

Design:

"""
import argparse
import glob
import logging
import math
import os
import shutil
import subprocess

from osgeo import gdal
from osgeo import osr
import ecoshard
import numpy
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

    one_array = numpy.ones((1, 1), dtype=numpy.int8)
    for outlet_feature in outlet_layer:
        outlet_band.WriteArray(
            one_array,
            outlet_feature.GetField('i'),
            outlet_feature.GetField('j'))
    outlet_band = None
    outlet_raster = None


def process_watershed(
        watershed_vector_path, watershed_fid, dem_path, pop_raster_path_list,
        target_beneficiaries_path_list, target_stitch_path_list,
        target_stitch_lock_list, completed_job_set, work_db_path, db_lock):
    """Calculate downstream beneficiaries for this watershed.

    Args:
        watershed_vector_path (str): path to watershed vector
        watershed_fid (str): watershed FID to process
        dem_path (str): path to DEM vector
        pop_raster_path_list (list): list of population rasters to route
        target_beneficiaries_path_list (str): list of target downstream
            beneficiary rasters to create, parallel with
            `pop_raster_path_list`.
        target_stitch_path_list (list): list of target stitch rasters to
            stitch the result into
        target_stitch_lock_list (list): list of locks to use when stitching
            to the rasters in the parallel `target_stitch_path_list`.
        completed_job_set (set): jobs that have been completed on previous
            iterations are in this set.
        work_db_path (str): path to an SQLite db that has 'job ids' for
            completed watershed.
        db_lock (lock): lock for the database to write to.

    Return:
        None.
    """
    job_id = f'''{os.path.basename(
        os.path.splitext(watershed_vector_path)[0])}_{watershed_fid}'''
    # check if job_id is in the database as done, if so skip
    if job_id in completed_job_set:
        return

    LOGGER.debug(f'create working directory for {job_id}')

    working_dir = os.path.join(
        os.path.dirname(target_beneficiaries_path_list[0]), job_id)
    try:
        os.makedirs(working_dir)
    except OSError:
        LOGGER.warning(f'{working_dir} already exists')

    task_graph = taskgraph.TaskGraph(working_dir, -1)

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
    warp_dem_task = task_graph.add_task(
        func=pygeoprocessing.warp_raster,
        args=(
            dem_path, target_pixel_size, warped_dem_raster_path, 'near'),
        kwargs={
            'target_bb': target_watershed_bb,
            'target_projection_wkt': epsg_sr.ExportToWkt(),
            'vector_mask_options': {
                'mask_vector_path': watershed_vector_path,
                'mask_vector_where_filter': f'"FID"={watershed_fid}'},
            'gdal_warp_options': None,
            'working_dir': working_dir},
        target_path_list=[warped_dem_raster_path],
        task_name=f'clip and warp dem to {warped_dem_raster_path}')

    LOGGER.debug('route dem')
    filled_dem_raster_path = os.path.join(
        working_dir, f'{job_id}_filled_dem.tif')
    fill_pits_task = task_graph.add_task(
        func=pygeoprocessing.routing.fill_pits,
        args=(
            (warped_dem_raster_path, 1), filled_dem_raster_path),
        kwargs={'working_dir': working_dir},
        dependent_task_list=[warp_dem_task],
        target_path_list=[filled_dem_raster_path],
        task_name=f'fill dem pits to {filled_dem_raster_path}')

    flow_dir_d8_raster_path = os.path.join(
        working_dir, f'{job_id}_flow_dir_d8.tif')
    flow_dir_d8_task = task_graph.add_task(
        func=pygeoprocessing.routing.flow_dir_d8,
        args=(
            (filled_dem_raster_path, 1), flow_dir_d8_raster_path),
        kwargs={'working_dir': working_dir},
        dependent_task_list=[fill_pits_task],
        target_path_list=[flow_dir_d8_raster_path],
        task_name=f'calc flow dir for {flow_dir_d8_raster_path}')

    outlet_vector_path = os.path.join(
        working_dir, f'{job_id}_outlet_vector.gpkg')
    detect_outlets_task = task_graph.add_task(
        func=pygeoprocessing.routing.detect_outlets,
        args=((flow_dir_d8_raster_path, 1), outlet_vector_path),
        dependent_task_list=[flow_dir_d8_task],
        target_path_list=[outlet_vector_path],
        task_name=f'detect outlets {outlet_vector_path}')

    outlet_raster_path = os.path.join(
        working_dir, f'{job_id}_outlet_raster.tif')
    create_outlet_raster_task = task_graph.add_task(
        func=_create_outlet_raster,
        args=(
            outlet_vector_path, flow_dir_d8_raster_path, outlet_raster_path),
        dependent_task_list=[detect_outlets_task],
        target_path_list=[outlet_raster_path],
        task_name=f'create outlet raster {outlet_raster_path}')

    for (pop_raster_path, target_beneficiaries_path,
         target_stitch_path, target_stitch_lock) in zip(
            pop_raster_path_list, target_beneficiaries_path_list,
            target_stitch_path_list, target_stitch_lock_list):
        LOGGER.debug(
            f'route downstream beneficiaries to '
            f'{target_beneficiaries_path_list}')

        aligned_pop_raster_path = os.path.join(
            working_dir,
            f'''{job_id}_{os.path.basename(
                os.path.splitext(pop_raster_path)[0])}.tif''')

        pop_warp_task = task_graph.add_task(
            func=pygeoprocessing.warp_raster,
            args=(
                pop_raster_path, target_pixel_size, aligned_pop_raster_path,
                'near'),
            kwargs={
                'target_bb': target_watershed_bb,
                'target_projection_wkt': epsg_sr.ExportToWkt(),
                'vector_mask_options': {
                    'mask_vector_path': watershed_vector_path,
                    'mask_vector_where_filter': f'"FID"={watershed_fid}'},
                'working_dir': working_dir},
            target_path_list=[aligned_pop_raster_path],
            task_name=f'align {aligned_pop_raster_path}')

        downstream_bene_task = task_graph.add_task(
            func=pygeoprocessing.routing.distance_to_channel_d8,
            args=(
                (flow_dir_d8_raster_path, 1), (outlet_raster_path, 1),
                target_beneficiaries_path),
            kwargs={
                'weight_raster_path_band': (aligned_pop_raster_path, 1)},
            dependent_task_list=[
                pop_warp_task, create_outlet_raster_task, flow_dir_d8_task],
            target_path_list=[target_beneficiaries_path],
            task_name=(
                'calc downstream beneficiaries for '
                f'{target_beneficiaries_path}'))

        downstream_bene_task.join()
        with target_stitch_lock:
            # stitch pop_raster_path into target stitch
            pygeoprocessing.stitch_rasters(
                [(target_beneficiaries_path, 1)], ['near'],
                target_stitch_path)
            # rm the target_beneficiaries_path
            os.remove(target_beneficiaries_path)

    task_graph.close()
    task_graph.join()
    task_graph = None
    shutil.rmtree(working_dir)
    # make entry in database that everything is complete
    with db_lock:
        record_job_id_complete(work_db_path, job_id)


def get_completed_job_id_set(db_path):
    """Return set of completed jobs, or initialize if not set."""
    if not os.path.exists(db_path):
        sql_create_projects_table_script = (
            """
            CREATE TABLE completed_job_ids (
                job_id TEXT NOT NULL,
                PRIMARY KEY (job_id)
            );
            """)
        connection = sqlite3.connect(db_path)
        cursor = connection.execute(
            """
            CREATE TABLE completed_job_ids (
                job_id TEXT NOT NULL,
                PRIMARY KEY (job_id)
            );
            """)
        cursor.close()
        connection.commit()
        connection.close()
        cursor = None
        connection = None

    ro_uri = r'%s?mode=ro' % pathlib.Path(
        os.path.abspath(db_path)).as_uri()
    connection = sqlite3.connect(ro_uri, uri=True)
    cursor = connection.execute('''SELECT * FROM completed_job_ids''')
    result = set(cursor.fetchall())
    cursor.close()
    connection.commit()
    connection.close()
    cursor = None
    connection = None
    return result


def record_job_id_complete(db_path, job_id):
    """Make an entry in the db that the job is complete."""
    connection = sqlite3.connect(db_path)
    cursor = connection.execute(
        f"""
        INSERT INTO completed_job_ids VALUES ("{job_id}")
        """)
    cursor.close()
    connection.commit()
    connection.close()
    cursor = None
    connection = None


def main(watershed_ids=None):
    """Entry point.

    Args:
        watershed_ids (list): if present, only run analysis on the list
            of 'watershed,fid' strings in this list.

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

    work_db_path = os.path.join(WORKSPACE_DIR, 'completed_fids.db')
    completed_job_set = get_completed_job_id_set(work_db_path)

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
    stitch_raster_path_map = {}
    for pop_id, pop_url in POPULATION_RASTER_URL_MAP.items():
        pop_raster_path = os.path.join(
            population_download_dir, os.path.basename(pop_url))
        download_pop_raster = task_graph.add_task(
            func=ecoshard.download_url,
            args=(pop_url, pop_raster_path),
            target_path_list=[pop_raster_path],
            task_name=f'download {pop_url}')
        pop_raster_path_map[pop_id] = pop_raster_path
        stitch_raster_path_map[pop_id] = os.path.join(
            WORKSPACE_DIR, f'global_stitch_{pop_id}.tif')

        if not os.path.exists(stitch_raster_path_map[pop_id]):
            driver = gdal.GetDriverByName('GTiff')
            cell_size = 0.0003
            target_raster = driver.Create(
                stitch_raster_path_map[pop_id],
                int(360/cell_size), int(180/cell_size), 1,
                gdal.GDT_Float32,
                options=(
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                    'SPARSE_OK=TRUE', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            target_raster.SetProjection(wgs84_srs.ExportToWkt())
            target_raster.SetGeoTransform(
                [-180, cell_size, 0, 90, 0, -cell_size])
            target_band = target_raster.GetRasterBand(1)
            target_band.SetNoDataValue(-9999)
            target_raster = None

    LOGGER.info('wait for downloads to conclude')
    task_graph.join()

    watershed_root_dir = os.path.join(
        watershed_download_dir, 'watersheds_globe_HydroSHEDS_15arcseconds')

    manager = multiprocessing.Manager()
    stitch_lock_list = [
        manager.Lock() for _ in range(len(stitch_raster_path_map))]
    db_lock = manager.Lock()

    if watershed_ids:
        for watershed_id in watershed_ids:
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
                     watershed_fid}.tif'''],
                [stitch_raster_path_map['2000'],
                 stitch_raster_path_map['2017']]
                stitch_lock_list,
                completed_job_set,
                work_db_path,
                db_lock)
    else:
        for watershed_path in glob.glob(
                os.path.join(watershed_root_dir, '*.shp')):
            watershed_vector = gdal.OpenEx(watershed_path, gdal.OF_VECTOR)
            watershed_layer = watershed_vector.GetLayer()
            watershed_fid_list = [
                watershed_feature.GetFID()
                for watershed_feature in watershed_layer]
            watershed_layer = None
            watershed_vector = None
            for watershed_fid in watershed_fid_list:
                process_watershed(
                    watershed_path, watershed_fid, dem_vrt_path,
                    [pop_raster_path_map['2000'],
                     pop_raster_path_map['2017']],
                    [f'''downstream_benficiaries_2000_{watershed_basename}_{
                     watershed_fid}.tif''',
                     f'''downstream_benficiaries_2017_{watershed_basename}_{
                         watershed_fid}.tif'''],
                    [stitch_raster_path_map['2000'],
                     stitch_raster_path_map['2017']]
                    stitch_lock_list,
                    completed_job_set,
                    work_db_path,
                    db_lock)
                break
            break

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream beneficiaries')
    parser.add_argument(
        '--watershed_ids', nargs='+',
        help='if present only run on this watershed id')
    args = parser.parse_args()

    main(watershed_ids=args.watershed_ids)
