"""Calculate downstream beneficiaries.

Design:

"""
import logging
import os

import ecoshard
import taskgraph

logging.basicConfig(
    level=logging.DEBUG,
    filename='log.txt',
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


def main():
    """Entry point."""
    task_graph = taskgraph.TaskGraph(WORKSPACE_DIR, 0, 15.0)

    dem_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(DEM_ZIP_URL)[0]))
    watershed_dir = os.path.join(
        WORKSPACE_DIR, os.path.basename(os.path.splitext(
            WATERSHED_VECTOR_ZIP_URL)[0]))
    population_dir = os.path.join(
        WORKSPACE_DIR, 'population_rasters')

    for dir_path in [dem_dir, watershed_dir, population_dir]:
        os.makedirs(dir_path, exist_ok=True)

    download_dem_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(DEM_ZIP_URL, dem_dir),
        task_name='download and unzip dem')

    download_watershed_vector_task = task_graph.add_task(
        func=ecoshard.download_and_unzip,
        args=(WATERSHED_VECTOR_ZIP_URL, watershed_dir),
        task_name='download and unzip watershed vector')

    for pop_id, pop_url in POPULATION_RASTER_URL_MAP.items():
        pop_raster_path = os.path.join(
            population_dir, os.path.basename(pop_url))
        download_pop_raster = task_graph.add_task(
            func=ecoshard.download_url,
            args=(pop_url, pop_raster_path),
            target_path_list=[pop_raster_path],
            task_name=f'download {pop_url}')

    LOGGER.info('wait for downloads to conclude')
    task_graph.join()

    task_graph.join()
    task_graph.close()


if __name__ == '__main__':
    main()
