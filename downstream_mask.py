"""Calculate downstream beneficiaries."""
import argparse
import collections
import glob
import logging
import math
import multiprocessing
import os
import pathlib
import queue
import shutil
import sqlite3
import subprocess
import threading
import time
from inspect import signature
from functools import wraps
from multiprocessing import managers

from osgeo import gdal
from osgeo import osr
import ecoshard
import numpy

import ecoshard.geoprocessing as geoprocessing
import ecoshard.geoprocessing.routing as routing
from ecoshard import taskgraph


gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)
logging.getLogger('taskgraph').setLevel(logging.WARN)
logging.getLogger('ecoshard.geoprocessing').setLevel(logging.WARN)

MASK_PATH_LIST = glob.glob(os.path.join(
    'ecoshards', '*.tif'))

# Backport of https://github.com/python/cpython/pull/4819
# Improvements to the Manager / proxied shared values code
# broke handling of proxied objects without a custom proxy type,
# as the AutoProxy function was not updated.
#
# This code adds a wrapper to AutoProxy if it is missing the
# new argument.
orig_AutoProxy = managers.AutoProxy


@wraps(managers.AutoProxy)
def AutoProxy(*args, incref=True, manager_owned=False, **kwargs):
    """Autoproxy for fixing some manager/proxy shared values.

    Create the autoproxy without the manager_owned flag, then
    update the flag on the generated instance. If the manager_owned flag
    is set, `incref` is disabled, so set it to False here for the same
    result.
    """
    autoproxy_incref = False if manager_owned else incref
    proxy = orig_AutoProxy(*args, incref=autoproxy_incref, **kwargs)
    proxy._owned_by_manager = manager_owned
    return proxy


def apply_manager_autopatch():
    """If not manager owned set it up so it is."""
    if "manager_owned" in signature(managers.AutoProxy).parameters:
        return

    LOGGER.debug(
        "Patching multiprocessing.managers.AutoProxy to add manager_owned")
    managers.AutoProxy = AutoProxy

    # re-register any types already registered to SyncManager without a custom
    # proxy type, as otherwise these would all be using the old unpatched
    # AutoProxy
    SyncManager = managers.SyncManager
    registry = managers.SyncManager._registry
    for typeid, (callable, exposed, method_to_typeid, proxytype) in \
            registry.items():
        if proxytype is not orig_AutoProxy:
            continue
        create_method = hasattr(managers.SyncManager, typeid)
        SyncManager.register(
            typeid,
            callable=callable,
            exposed=exposed,
            method_to_typeid=method_to_typeid,
            create_method=create_method,
        )


DEM_ZIP_URL = (
    'https://storage.googleapis.com/ecoshard-root/key_datasets/'
    'global_dem_3s_md5_22d0c3809af491fa09d03002bdf09748.zip')
WATERSHED_VECTOR_ZIP_URL = (
    'https://storage.googleapis.com/ecoshard-root/key_datasets/'
    'watersheds_globe_HydroSHEDS_15arcseconds_md5_'
    'c6acf2762123bbd5de605358e733a304.zip')

WORKSPACE_DIR = 'workspace'
WATERSHED_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'watershed_workspace')
STITCHED_WORKSPACE_DIR = os.path.join(WORKSPACE_DIR, 'stitched_results')
for dir_path in [
        WORKSPACE_DIR, WATERSHED_WORKSPACE_DIR, STITCHED_WORKSPACE_DIR]:
    os.makedirs(dir_path, exist_ok=True)
N_TO_STITCH = 100


def _mask_op(base_array, nodata, target_nodata):
    """Set > 0 to 1, 0==0 and nodata to nodata."""
    result = numpy.full(base_array.shape, target_nodata, dtype=numpy.int8)
    valid_mask = base_array != nodata
    result[valid_mask] = base_array[valid_mask] > 0
    return result


def threshold_raster(base_raster_path, target_mask_path):
    """Threshold base to 0/1 mask and preserve nodata."""
    raster_info = geoprocessing.get_raster_info(base_raster_path)
    geoprocessing.raster_calculator(
        [(base_raster_path, 1), (raster_info['nodata'], 'raw'), (2, 'raw')],
        _mask_op, target_mask_path, gdal.GDT_Byte, 2)


def process_watershed(
        job_id, watershed_vector_path, watershed_fid_list, epsg_code,
        lat_lng_watershed_bb, dem_path,
        mask_path_list, working_dir,
        target_stitch_work_queue_list):
    """Flow mask downstream like ink running downhill.

    Args:
        job_id (str): unique ID identifying this job, can be used to
            create unique workspaces.
        watershed_vector_path (str): path to watershed vector
        watershed_fid_list (str): list of watershed ids to process
        epsg_code (int): EPSG zone to locally project into
        lat_lng_watershed_bb (list): lat/lng bounding box for the fids
            that are in `watershed_fid_list`.
        dem_path (str): path to DEM raster
        mask_path_list (str): list of raster masks to spill downstream,
            there is one mask per target work queue.
        working_dir (str): a unique path to a workspace that can be used by
            this call
        target_stitch_work_queue_list (list): list of work queue tuples to
            put done signals in when each downstream mask is done. The
            first element is for the standard target, the second for the
            normalized raster.

    Return:
        None.
    """
    try:
        LOGGER.debug(
            f'create working directory for {job_id} at {working_dir} '
            f'with these masks {mask_path_list}')
        os.makedirs(working_dir, exist_ok=True)

        LOGGER.debug(f'create taskgraph at {working_dir}')
        task_graph = taskgraph.TaskGraph(working_dir, -1)
        watershed_info = geoprocessing.get_vector_info(
            watershed_vector_path)

        epsg_sr = osr.SpatialReference()
        epsg_sr.ImportFromEPSG(epsg_code)

        target_watershed_bb = geoprocessing.transform_bounding_box(
            lat_lng_watershed_bb,
            watershed_info['projection_wkt'],
            epsg_sr.ExportToWkt())

        LOGGER.debug(f'target watershed bb {target_watershed_bb}')

        target_pixel_size = (300, -300)

        warped_dem_raster_path = os.path.join(working_dir, f'{job_id}_dem.tif')

        mask_id_list = [
            os.path.basename(os.path.splitext(mask_path)[0])
            for mask_path in mask_path_list]

        warped_mask_raster_path_list = [
            os.path.join(working_dir, f'{job_id}_{mask_id}.tif')
            for mask_id in mask_id_list]

        LOGGER.debug(
            f'align and resize raster stack {job_id} at {working_dir}')
        mask_vector_where_filter = (
            f'"FID" in ('
            f'{", ".join([str(v) for v in watershed_fid_list])})')
        LOGGER.debug(mask_vector_where_filter)
        align_task = task_graph.add_task(
            func=geoprocessing.align_and_resize_raster_stack,
            args=(
                [dem_path] + mask_path_list,
                [warped_dem_raster_path] + warped_mask_raster_path_list,
                ['near']+['mode']*len(mask_path_list),
                target_pixel_size, target_watershed_bb),
            kwargs={
                'target_projection_wkt': epsg_sr.ExportToWkt(),
                'vector_mask_options': {
                    'mask_vector_path': watershed_vector_path,
                    'mask_vector_where_filter': mask_vector_where_filter,
                    },
                },
            target_path_list=[
                warped_dem_raster_path] + warped_mask_raster_path_list,
            task_name=(
                f'align and clip and warp dem/hab to {warped_dem_raster_path} '
                f'{warped_mask_raster_path_list}'))

        # force a drain on the watershed if its large enough
        if len(watershed_fid_list) == 1:
            LOGGER.debug(
                f'detect_lowest_drain_and_sink {job_id} at {working_dir}')
            get_drain_sink_pixel_task = task_graph.add_task(
                func=routing.detect_lowest_drain_and_sink,
                args=((warped_dem_raster_path, 1),),
                store_result=True,
                dependent_task_list=[align_task],
                task_name=f'get drain/sink pixel for {warped_dem_raster_path}')

            edge_pixel, edge_height, pit_pixel, pit_height = (
                get_drain_sink_pixel_task.get())

            if pit_height < edge_height - 20:
                # if the pit is 20 m lower than edge it's probably a big sink
                single_outlet_tuple = pit_pixel
            else:
                single_outlet_tuple = edge_pixel
        else:
            single_outlet_tuple = None

        filled_dem_raster_path = os.path.join(
            working_dir, f'{job_id}_filled_dem.tif')
        LOGGER.debug(f'fill_pits {job_id} at {working_dir}')
        fill_pits_task = task_graph.add_task(
            func=routing.fill_pits,
            args=(
                (warped_dem_raster_path, 1), filled_dem_raster_path),
            kwargs={
                'working_dir': working_dir,
                'max_pixel_fill_count': -1,
                'single_outlet_tuple': single_outlet_tuple},
            dependent_task_list=[align_task],
            target_path_list=[filled_dem_raster_path],
            task_name=f'fill dem pits to {filled_dem_raster_path}')

        LOGGER.debug(f'flow_dir_mfd {job_id} at {working_dir}')
        flow_dir_mfd_raster_path = os.path.join(
            working_dir, f'{job_id}_flow_dir_mfd.tif')
        flow_dir_mfd_task = task_graph.add_task(
            func=routing.flow_dir_mfd,
            args=(
                (filled_dem_raster_path, 1), flow_dir_mfd_raster_path),
            kwargs={'working_dir': working_dir},
            dependent_task_list=[fill_pits_task],
            target_path_list=[flow_dir_mfd_raster_path],
            task_name=f'calc flow dir for {flow_dir_mfd_raster_path}')

        for warped_mask_raster_path, stitch_work_queue in zip(
                warped_mask_raster_path_list, target_stitch_work_queue_list):
            upstream_sum_mask_path = os.path.join(
                working_dir,
                f'upstream_{os.path.basename(warped_mask_raster_path)}')
            LOGGER.debug(
                f'create downstream mask raster {job_id} at {working_dir}')
            flow_accum_mfd_task = task_graph.add_task(
                func=routing.flow_accumulation_mfd,
                args=(
                    (flow_dir_mfd_raster_path, 1),
                    upstream_sum_mask_path),
                dependent_task_list=[flow_dir_mfd_task],
                kwargs={
                    'weight_raster_path_band': (warped_mask_raster_path, 1)},
                target_path_list=[upstream_sum_mask_path],
                task_name=f'calc flow accum for {upstream_sum_mask_path}')

            target_upstream_mask_path = os.path.join(
                working_dir,
                f'upstream_mask_{os.path.basename(warped_mask_raster_path)}')

            threshold_task = task_graph.add_task(
                func=threshold_raster,
                args=(upstream_sum_mask_path, target_upstream_mask_path),
                target_path_list=[target_upstream_mask_path],
                dependent_task_list=[flow_accum_mfd_task],
                task_name=f'threshold {upstream_sum_mask_path}')

            threshold_task.join()
            stitch_start_time = time.time()
            stitch_work_queue.put(
                (target_upstream_mask_path, working_dir, job_id))
            LOGGER.debug(f'took {time.time()-stitch_start_time:.3f}s to put stitch for {job_id}')

        task_graph.close()
        task_graph.join()
        task_graph = None
    except Exception:
        LOGGER.exception('failure in watershed worker')
        raise


def get_completed_job_id_set(db_path):
    """Return set of completed jobs, or initialize if not set."""
    if not os.path.exists(db_path):
        LOGGER.debug(f'dbpath: {db_path}')
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
    result = set([_[0] for _ in cursor.fetchall()])
    cursor.close()
    connection.commit()
    connection.close()
    cursor = None
    connection = None
    return result


def job_complete_worker(
        completed_work_queue, work_db_path, clean_result, n_expected):
    """Update the database with completed work.

    Args:
        completed_work_queue (queue): queue with (working_dir, job_id)
            incoming from each stitched raster
        work_db_path (str): path to the work database
        clean_result (bool): if true, delete the working directory after
            ``n_expected`` results come through.
        n_expected (int): number of expected duplicate jobs to come through
            before marking complete.

    Return:
        ``None``
    """
    try:
        start_time = time.time()
        connection = sqlite3.connect(work_db_path)
        uncommited_count = 0
        processed_so_far = 0
        working_jobs = collections.defaultdict(int)
        global WATERSHEDS_TO_PROCESS_COUNT
        LOGGER.info(
            f'started job complete worker, initial watersheds '
            f'{WATERSHEDS_TO_PROCESS_COUNT}')
        while True:
            payload = completed_work_queue.get()
            if payload is None:
                LOGGER.info('got None in completed work, terminating')
                break
            working_dir, job_id = payload
            working_jobs[job_id] += 1
            if working_jobs[job_id] < n_expected:
                continue
            # we got n_expected, so mark complete
            del working_jobs[job_id]
            WATERSHEDS_TO_PROCESS_COUNT -= 1
            if clean_result:
                shutil.rmtree(working_dir, ignore_errors=True)
            sql_command = (
                f"""
                INSERT INTO completed_job_ids VALUES ("{job_id}")
                """)
            LOGGER.debug(sql_command)
            cursor = connection.execute(sql_command)
            cursor.close()
            LOGGER.info(f'done with {job_id} {working_dir}')
            uncommited_count += 1
            if uncommited_count > N_TO_STITCH:
                connection.commit()
                processed_so_far += uncommited_count
                watersheds_per_sec = (
                    processed_so_far / (time.time() - start_time))
                uncommited_count = 0
                remaining_time_s = (
                    WATERSHEDS_TO_PROCESS_COUNT / watersheds_per_sec)
                remaining_time_h = int(remaining_time_s // 3600)
                remaining_time_s -= remaining_time_h * 3600
                remaining_time_m = int(remaining_time_s // 60)
                remaining_time_s -= remaining_time_m * 60
                LOGGER.info(
                    f'remaining watersheds to process: '
                    f'{WATERSHEDS_TO_PROCESS_COUNT} - '
                    f'processed so far {processed_so_far} - '
                    f'process/sec: {watersheds_per_sec:.1f} - '
                    f'time left: {remaining_time_h}:'
                    f'{remaining_time_m:02d}:{remaining_time_s:04.1f}')

        connection.commit()
        connection.close()
        cursor = None
        connection = None
    except Exception:
        LOGGER.exception('error on job complete worker')
        raise


def general_worker(work_queue, worker_id, max_workers):
    """Invoke func on args coming through work queue."""
    last_time = time.time()
    while True:
        try:
            payload = work_queue.get(True, 5)
        except queue.Empty:
            LOGGER.info(f'worker {worker_id} of {max_workers} waiting {time.time()-last_time:.3}s for work.')
            continue
        if payload is None:
            work_queue.put(None)
            LOGGER.debug('got a none on general worker, quitting')
            break
        func, args = payload
        start_time = time.time()
        LOGGER.info(f'worker {worker_id} of {max_workers} waited {start_time-last_time:.3}s for work and is starting.')
        func(*args)
        LOGGER.info(f'worker {worker_id} of {max_workers} DONE in {time.time()-start_time:.3}s')
        last_time = time.time()


def _sqrt_op(array, nodata):
    result = numpy.full(array.shape, nodata, dtype=numpy.float32)
    valid_array = array >= 0
    result[valid_array] = numpy.sqrt(array[valid_array])
    return result


def stitch_worker(
        stitch_work_queue, target_stitch_raster_path,
        stitch_done_queue, clean_result):
    """Take jobs from stitch work queue and stitch into target."""
    stitch_buffer_list = []
    done_buffer = []
    n_buffered = 0
    while True:
        payload = stitch_work_queue.get()
        if payload is None:
            LOGGER.debug(
                f'stitch worker for {target_stitch_raster_path} '
                f'got DONE signal')
            stitch_work_queue.put(None)
        else:
            raster_path, working_dir, job_id = payload
            done_buffer.append((working_dir, job_id))
            if not os.path.exists(raster_path):
                message = (
                    f'{raster_path} does not exist on disk when stitching '
                    f'into {target_stitch_raster_path} also working dir is '
                    f'{working_dir}')
                LOGGER.error(message)
                raise ValueError(message)
            stitch_buffer_list.append((raster_path, 1))
            n_buffered += 1
        if n_buffered > N_TO_STITCH or payload is None:
            LOGGER.info(
                f'about to stitch {n_buffered} into '
                f'{target_stitch_raster_path}')
            start_time = time.time()
            geoprocessing.stitch_rasters(
                stitch_buffer_list, ['near']*n_buffered,
                (target_stitch_raster_path, 1),
                area_weight_m2_to_wgs84=False,
                overlap_algorithm='replace')
            for working_dir, job_id in done_buffer:
                stitch_done_queue.put((working_dir, job_id))
            stitch_buffer_list = []
            done_buffer = []
            elapsed_time = time.time() - start_time
            LOGGER.info(
                f'took {time.time()-start_time:.2f}s to stitch '
                f'{n_buffered/elapsed_time:.2f} per sec into '
                f'{target_stitch_raster_path}')
            n_buffered = 0
        if payload is None:
            break


def main(watershed_ids=None, n_workers=multiprocessing.cpu_count()):
    """Entry point.

    Args:
        watershed_ids (list): if present, only run analysis on the list
            of 'watershed,fid' strings in this list.

    Return:
        None.
    """
    LOGGER.info('create new taskgraph')
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, -1)

    basename_dem = os.path.basename(os.path.splitext(DEM_ZIP_URL)[0])
    dem_download_dir = os.path.join(WORKSPACE_DIR, basename_dem)
    watershed_download_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(
            WATERSHED_VECTOR_ZIP_URL)[0]))
    population_download_dir = os.path.join(
        WORKSPACE_DIR, 'population_rasters')

    work_db_path = os.path.join(WORKSPACE_DIR, 'completed_fids.db')
    LOGGER.info('fetch completed job set')
    completed_job_set = get_completed_job_id_set(work_db_path)
    LOGGER.info(f'there are {len(completed_job_set)} completed jobs so far')

    for dir_path in [
            dem_download_dir, watershed_download_dir,
            population_download_dir]:
        os.makedirs(dir_path, exist_ok=True)

    LOGGER.info('download dem')
    download_dem_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_ZIP_URL, dem_download_dir),
        target_path_list=[
            os.path.join(
                dem_download_dir, os.path.basename(DEM_ZIP_URL))],
        task_name='download and unzip dem')
    download_dem_task.join()

    dem_tile_dir = os.path.join(dem_download_dir, 'global_dem_3s')
    dem_vrt_path = os.path.join(
        dem_tile_dir,
        f'{os.path.basename(os.path.splitext(DEM_ZIP_URL)[0])}.vrt')
    LOGGER.debug(f'build vrt to {dem_vrt_path}')

    LOGGER.info('build vrt')
    build_vrt_task = task_graph.add_task(
        func=subprocess.run,
        args=(f"gdalbuildvrt {dem_vrt_path} {os.path.join(dem_tile_dir, '*.tif')}",),
        kwargs={'shell': True, 'check': True},
        target_path_list=[dem_vrt_path],
        dependent_task_list=[download_dem_task],
        task_name='build dem vrt')
    build_vrt_task.join()

    download_watershed_vector_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(WATERSHED_VECTOR_ZIP_URL, watershed_download_dir),
        task_name='download and unzip watershed vector')
    download_watershed_vector_task.join()

    stitch_raster_path_map = {}

    LOGGER.info('wait for downloads to conclude')
    task_graph.join()
    task_graph.close()
    task_graph = None

    apply_manager_autopatch()
    manager = multiprocessing.Manager()
    completed_work_queue = manager.Queue()
    LOGGER.info('start complete worker thread')
    global WATERSHEDS_TO_PROCESS_COUNT
    WATERSHEDS_TO_PROCESS_COUNT = 0
    job_complete_worker_thread = threading.Thread(
        target=job_complete_worker,
        args=(
            completed_work_queue, work_db_path, args.clean_result,
            len(MASK_PATH_LIST)))
    job_complete_worker_thread.daemon = True
    job_complete_worker_thread.start()

    # contains work queues for each mask
    stitch_work_queue_list = []
    stitch_worker_process_list = []
    for mask_raster_path in MASK_PATH_LIST:

        mask_id = os.path.basename(os.path.splitext(mask_raster_path)[0])
        stitch_raster_path = os.path.join(
            STITCHED_WORKSPACE_DIR, f'downstream_mask_{mask_id}.tif')
        if not os.path.exists(stitch_raster_path):
            driver = gdal.GetDriverByName('GTiff')
            cell_size = 10./3600. * 2  # do this for Nyquist theorem
            n_cols = int(360./cell_size)
            n_rows = int(180./cell_size)
            LOGGER.info(f'**** creating raster of size {n_cols} by {n_rows}')
            target_raster = driver.Create(
                stitch_raster_path,
                int(360/cell_size), int(180/cell_size), 1,
                gdal.GDT_Byte,
                options=(
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                    'SPARSE_OK=TRUE', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            target_raster.SetProjection(wgs84_srs.ExportToWkt())
            target_raster.SetGeoTransform(
                [-180, cell_size, 0, 90, 0, -cell_size])
            target_band = target_raster.GetRasterBand(1)
            target_band.SetNoDataValue(2)
            target_band.SetStatistics(0, 1, 0, 0)
            target_raster = None

        stitch_work_queue = manager.Queue(N_TO_STITCH*2)
        stitch_work_queue_list.append(stitch_work_queue)
        target_stitch_raster_path = os.path.join(
            STITCHED_WORKSPACE_DIR, os.path.basename(mask_raster_path))
        LOGGER.debug(f'starting a stitcher for {target_stitch_raster_path}')
        stitch_worker_process = multiprocessing.Process(
            target=stitch_worker,
            args=(
                stitch_work_queue, stitch_raster_path,
                completed_work_queue, args.clean_result))
        stitch_worker_process.deamon = True
        stitch_worker_process.start()
        stitch_worker_process_list.append(stitch_worker_process)

    watershed_work_queue = manager.Queue()

    watershed_root_dir = os.path.join(
        watershed_download_dir, 'watersheds_globe_HydroSHEDS_15arcseconds')

    watershed_worker_process_list = []
    for worker_id in range(n_workers):
        watershed_worker_process = multiprocessing.Process(
            target=general_worker,
            args=(watershed_work_queue, worker_id+1, n_workers))
        watershed_worker_process.daemon = True
        watershed_worker_process.start()
        watershed_worker_process_list.append(watershed_worker_process)

    LOGGER.info('building watershed fid list')
    if watershed_ids:
        valid_watershed_basenames = set()
        valid_watersheds = set()
        for watershed_id in watershed_ids:
            watershed_basename, watershed_fid = watershed_id.split(',')
            valid_watershed_basenames.add(watershed_basename)
            valid_watersheds.add((watershed_basename, int(watershed_fid)))
        LOGGER.debug(
            f'valid watershed basenames: {valid_watershed_basenames} '
            f'valid valid_watersheds: {valid_watersheds} ')

    duplicate_job_index_map = collections.defaultdict(int)
    for watershed_path in glob.glob(
            os.path.join(watershed_root_dir, '*.shp')):
        LOGGER.debug(f'processing {watershed_path}')
        watershed_fid_index = collections.defaultdict(
            lambda: [list(), list(), 0])
        watershed_basename = os.path.splitext(
            os.path.basename(watershed_path))[0]
        if watershed_ids:
            if watershed_basename not in valid_watershed_basenames:
                continue
        watershed_vector = gdal.OpenEx(watershed_path, gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()
        for watershed_feature in watershed_layer:
            fid = watershed_feature.GetFID()

            if watershed_ids:
                if not valid_watersheds:
                    # all done, we scheduled them all
                    break
                # skip on immediate mode
                if (watershed_basename, fid) not in valid_watersheds:
                    continue
                valid_watersheds.remove((watershed_basename, fid))
                LOGGER.info(f'scheduling {(watershed_basename, fid)}')
            watershed_geom = watershed_feature.GetGeometryRef()
            watershed_centroid = watershed_geom.Centroid()
            epsg = get_utm_zone(
                watershed_centroid.GetX(), watershed_centroid.GetY())
            if watershed_geom.Area() > 1 or watershed_ids:
                # one degree grids or immediates get special treatment
                job_id = (f'{watershed_basename}_{fid}_{epsg}', epsg)
                watershed_fid_index[job_id][0] = [fid]
            else:
                # clamp into 5 degree square
                x, y = [
                    int(v//5)*5 for v in (
                        watershed_centroid.GetX(), watershed_centroid.GetY())]
                base_job_id = f'{watershed_basename}_{x}_{y}_{epsg}'
                job_id = (
                    f'{base_job_id}_{duplicate_job_index_map[base_job_id]}',
                    epsg)
                if len(watershed_fid_index[job_id][0]) > 1000:
                    duplicate_job_index_map[base_job_id] += 1
                    job_id = (
                        f'''{base_job_id}_{
                            duplicate_job_index_map[base_job_id]}''', epsg)
                watershed_fid_index[job_id][0].append(fid)
            watershed_envelope = watershed_geom.GetEnvelope()
            watershed_bb = [watershed_envelope[i] for i in [0, 2, 1, 3]]
            watershed_fid_index[job_id][1].append(watershed_bb)
            watershed_fid_index[job_id][2] += watershed_geom.Area()

        watershed_geom = None
        watershed_feature = None
        watershed_layer = None
        watershed_vector = None

        processing_list = []
        for (job_id, epsg), (fid_list, watershed_envelope_list, area) in \
                watershed_fid_index.items():
            processing_list.append(
                (area, job_id, epsg, fid_list, watershed_envelope_list))

        for (area, job_id, epsg, fid_list, watershed_envelope_list) in \
                sorted(processing_list, reverse=True):
            if job_id in completed_job_set:
                continue
            if WATERSHEDS_TO_PROCESS_COUNT == args.max_to_run:
                break

            LOGGER.info(
                f'scheduling {job_id} of area {area} and {len(fid_list)} '
                f'watersheds')
            job_bb = geoprocessing.merge_bounding_box_list(
                watershed_envelope_list, 'union')

            workspace_dir = os.path.join(WATERSHED_WORKSPACE_DIR, job_id)
            LOGGER.debug(f'putting {job_id} to work')
            watershed_work_queue.put((
                process_watershed,
                (job_id, watershed_path, fid_list, epsg, job_bb, dem_vrt_path,
                 MASK_PATH_LIST,
                 workspace_dir,
                 stitch_work_queue_list)))

            #process_watershed(
            #    job_id, watershed_path, fid_list, epsg, job_bb, dem_vrt_path,
            #    MASK_PATH_LIST, workspace_dir, stitch_work_queue_list)

            WATERSHEDS_TO_PROCESS_COUNT += 1

    LOGGER.debug('waiting for watershed workers to be done')
    watershed_work_queue.put(None)
    for watershed_worker in watershed_worker_process_list:
        watershed_worker.join()
    LOGGER.debug('watershed workers are done')

    LOGGER.debug('signal stitch workers to be done')

    for stitch_queue in stitch_work_queue_list:
        stitch_queue.put(None)

    for stitch_worker_process in stitch_worker_process_list:
        stitch_worker_process.join()
    LOGGER.debug('stitch worker done')

    completed_work_queue.put(None)
    job_complete_worker_thread.join()
    LOGGER.info('compressing/overview/ecoshard result')

    hash_thread_list = []
    for pop_id in stitch_raster_path_map:
        for raster_path in stitch_raster_path_map[pop_id]:
            hash_thread = threading.Thread(
                target=hash_overview_compress_raster,
                args=(raster_path,))
            hash_thread.daemon = True
            hash_thread.start()
            hash_thread_list.append(hash_thread)

    LOGGER.info('waiting for compress/overview/ecoshard complete')
    for hash_thread in hash_thread_list:
        hash_thread.join()
    LOGGER.info('all done')


def hash_overview_compress_raster(raster_path):
    """Compress, overview, then hash the raster."""
    compressed_path = '%s_compressed_overviews%s' % os.path.splitext(
        raster_path)
    ecoshard.compress_raster(
        raster_path, compressed_path, compression_algorithm='LZW')
    ecoshard.build_overviews(compressed_path)
    compressed_raster = gdal.OpenEx(
        compressed_path, gdal.OF_RASTER | gdal.GA_Update)
    compressed_raster_band = compressed_raster.GetRasterBand(1)
    stats = compressed_raster_band.ComputeStatistics(False)
    LOGGER.debug(stats)
    compressed_raster_band = None
    compressed_raster = None
    ecoshard.hash_file(compressed_path, rename=True)


def get_centroid(vector_path, fid):
    """Return centroid x/y coordinate of given FID in the vector."""
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    feature = layer.GetFeature(fid)
    centroid = feature.GetGeometryRef().Centroid()
    feature = None
    layer = None
    vector = None
    return centroid.GetX(), centroid.GetY()


def get_utm_zone(x, y):
    """Return EPSG code for the utm zone holding lng (x), lat (y) coords."""
    utm_code = (math.floor((x + 180)/6) % 60) + 1
    lat_code = 6 if y > 0 else 7
    epsg_code = int('32%d%02d' % (lat_code, utm_code))
    return epsg_code


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downstream beneficiaries')
    parser.add_argument(
        '--watershed_ids', nargs='+',
        help='if present only run on this watershed id')
    parser.add_argument(
        '--clean_result', action='store_true',
        help='use this flag to delete the workspace after stitching')
    parser.add_argument(
        '--max_to_run', type=int, help='max number of watersheds to process')
    parser.add_argument(
        '--n_workers', type=int, default=multiprocessing.cpu_count(),
        help='max number of watersheds to process')

    args = parser.parse_args()

    main(watershed_ids=args.watershed_ids, n_workers=args.n_workers)
