from pathlib import Path
# noinspection PyShadowingBuiltins
from pprint import pprint as print

import rasterio as rio
from rasterio.io import DatasetReader
from affine import Affine
from joblib import Parallel, delayed
from osgeo import gdal
from rasterio.profiles import default_gtiff_profile
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, transform
from tqdm import tqdm

from errors import WpTillerError, WpTillerWarning
from utils import buildvrt, tile_children

gdal.UseExceptions()


class TileInfo(object):
    """ A class holding info how to tile """

    ## Taken from gdal restile source

    def __init__(self, xsize, ysize, tile_width, tile_height, overlap=0):
        self.width = xsize
        self.height = ysize
        self.tileWidth = tile_width
        self.tileHeight = tile_height
        self.countTilesX = 1
        if xsize > tile_width:
            self.countTilesX += int((xsize - tile_width + (tile_width - overlap) - 1) / (tile_width - overlap))
        self.countTilesY = 1
        if ysize > tile_height:
            self.countTilesY += int((ysize - tile_height + (tile_height - overlap) - 1) / (tile_height - overlap))
        self.overlap = overlap

    def report(self):
        print('tileWidth:   %d' % self.tileWidth)
        print('tileHeight:  %d' % self.tileHeight)
        print('countTilesX: %d' % self.countTilesX)
        print('countTilesY: %d' % self.countTilesY)
        print('overlap:     %d' % self.overlap)


class Dataset(object):
    tileinfos = {}

    def __init__(self, input_raster,
                 width=None, height=None, output_folder='./output',
                 levels=1, cores=1, block_size=512):
        """

        :param input_raster: str or Path
            Path to the input raster.
        :param width: int
            Width of the generated pixel. If None default to 256 px
        :param height: int
            Height of the generated pixel. If None default to 256 px
        :param output_folder: str or Path
            Path where generated tiles will be written.
        :param levels: str
            level or levels to make. a single number (e.g. 4) will construct the tiles for that zoom level.
            if from to format (e.g. 0-5) the program will construct the zoom levels from 0 (source) to 5.
        :param cores: int
            Number of cores to use.
            If cores=1 then the operation is not happening in parrallel mode.
            If cores=-1 all available cores will be used.
            default=1
        :param block_size: int
            internal tilling in pixels. For simplicity square tilling is assumed
        """
        print(input_raster)
        self.file_path = Path(input_raster)
        self.output_folder = Path(output_folder)
        self.blockSize = block_size
        self.cores = cores
        self.name = self.file_path.stem

        # Sanity checks
        # Check if input file exists
        if not self.file_path.is_file():
            raise WpTillerError('%s does not exist' % self.file_path.absolute().as_posix())

        # Check if output folder exists, if not create it
        if not self.output_folder.is_dir():
            WpTillerWarning('Created non existent folder\n%s.' % self.output_folder.as_posix())
            self.output_folder.mkdir(parents=True, exist_ok=True)

        # Level setting
        try:
            self.levels = int(levels)
            self._level_start = self.levels
            self._level_end = self.levels
        except ValueError:
            self._level_start, self._level_end = map(int, str(levels).split('-'))

        if self._level_start < 0 or self._level_end < 0 \
                or not isinstance(self._level_end, int) \
                or not isinstance(self._level_start, int):
            raise ValueError('Levels should be positive integers')

        self.TileWidth = width or 256
        self.TileHeight = height or 256
        if self.TileHeight < 0 or self.TileWidth < 0 \
                or not isinstance(self.TileWidth, int) \
                or not isinstance(self.TileHeight, int):
            raise ValueError('Tile Dimenstion should be positive integers')

        # Build the tileinfo dictinary
        for level in range(self._level_start, self._level_end + 1):
            meta = self.get_metadata(level)
            self.tileinfos[level] = TileInfo(meta['width'],
                                             meta['height'],
                                             self.TileWidth, self.TileHeight)

    def make_gdal_cmds(self, output_folder='./gdal_cmds'):
        raise NotImplemented

    # def sample_from_source(self, geog_x, geog_y):
    #
    #     """
    #
    #     Given a set of coordinates in the datasets coordinate system,
    #     return harvested values from the source raster
    #
    #     :param geog_x: list of float
    #         x values in coordinate reference system
    #     :param geog_y: list of float
    #         x values in coordinate reference system
    #     :return: float
    #     """
    #
    #     ds = self._ds
    #     return sample_gen(ds, zip(geog_x, geog_y))
    #     # raise NotImplemented

    def get_transform(self, level=0):
        ds = self.get_dataset(level)
        return ds.transform

    def get_metadata(self, level):
        ds = self.get_dataset(level)
        return ds.meta

    def get_dataset(self, level) -> DatasetReader:
        """
        Returns a vrt handle based on the source dataset, with resolution = src.resolution * level
        :param level: int
        :return: DatasetReader
        """

        level = int(level)
        if level == 0:
            return rio.open(self.file_path.as_posix())

        lvl0_ds = self.get_dataset(0)
        lvl0_height = lvl0_ds.height
        lvl0_width = lvl0_ds.width
        lvl0_transform = lvl0_ds.transform
        resolution_factor = pow(2, level)

        lvlx_height = lvl0_height / resolution_factor
        lvlx_width = lvl0_width / resolution_factor
        lvlx_tranform = Affine(lvl0_transform.a * resolution_factor,
                               lvl0_transform.b,
                               lvl0_transform.c,
                               lvl0_transform.d,
                               lvl0_transform.e * resolution_factor,
                               lvl0_transform.f)

        vrt = WarpedVRT(src_dataset=lvl0_ds, transform=lvlx_tranform, width=lvlx_width, height=lvlx_height)

        return vrt

    def get_tile(self, level, tile_y, tile_x):
        """
        Returns (id_x, id_y), Window, Transformation
            :param tile_x: int
            :param level: int
            :param tile_y: int
            :return tuple
        """
        level = int(level)
        tile_y = int(tile_y)
        tile_x = int(tile_x)

        ds = self.get_dataset(level)
        width = ds.width
        height = ds.height
        geotransform = self.get_transform(level)

        y_offset = tile_y * self.TileHeight
        x_offset = tile_x * self.TileWidth

        y_size = min(self.TileWidth, height - y_offset)
        x_size = min(self.TileHeight, width - x_offset)

        w = Window(x_offset, y_offset, x_size, y_size)
        t = transform(w, geotransform)

        return (tile_x + 1, tile_y + 1), w, t

    def slice_dataset(self):
        """
        MISSING
        :return: Nonthing
        """
        name_template = "{basename}_{y}_{x}.tif"
        out_folder = self.output_folder
        if not out_folder.exists():
            out_folder.mkdir()

        def make_tile(level, tile):
            """
            MISSING
            :param level:
            :param tile:
            :return:
            """

            # x,y tile indexes
            x = tile[0][0]
            y = tile[0][1]

            def div_by_16(x):
                if divmod(x, 16)[1] == 0:
                    return x
                return div_by_16(x - 1)

            # put tile in its respective dir
            out_dir = out_folder.joinpath(str(level))
            if not out_dir.exists():
                out_dir.mkdir(exist_ok=True)

            size_x = tile[1].width if tile[1].width > 0 else 1
            size_y = tile[1].height if tile[1].height > 0 else 1

            # Out file constructor
            # how many chars to use for representing the tiles.
            name_length = max(len(str(self.tileinfos[level].countTilesX)),
                              len(str(self.tileinfos[level].countTilesY))) + 1
            filename = name_template.format(basename=self.name,
                                            x=str(x).zfill(name_length),
                                            y=str(y).zfill(name_length))
            out_filepath = out_dir.joinpath(filename)
            ## End

            profile = default_gtiff_profile
            profile.update(crs='epsg:4326', driver='GTiff', transform=tile[2],
                           compress='lzw', count=1, width=size_x, height=size_y,
                           blockysize=div_by_16(min(self.blockSize, tile[1].height)),
                           blockxsize=div_by_16(min(self.blockSize, tile[1].width)),
                           )

            if level > 1:
                # except OSError:
                #     # in this level, the amount of pixels that need to be resampled are too many.
                #     # I am choosing to use pixel at the central coordinate of the processing tile
                #     # Sample error:
                #     # ERROR 1: Integer overflow : nSrcXSize=425985, nSrcYSize=163840
                # TODO: don't be lazy, clean write
                try:
                    self.tileinfos[level - 1]
                except KeyError:
                    _meta = self.get_metadata(level - 1)
                    self.tileinfos[level - 1] = TileInfo(_meta['width'], _meta['height'], self.TileWidth,
                                                         self.TileHeight)

                finally:
                    name_length = max(len(str(self.tileinfos[level - 1].countTilesX)),
                                      len(str(self.tileinfos[level - 1].countTilesY))) + 1

                prev_lvl_tiles = tile_children(zoom=level,
                                               src=out_filepath,
                                               ndigits=name_length)
                vrt_handler = buildvrt(prev_lvl_tiles)
                with rio.open(vrt_handler) as src:
                    profile.update(
                            nodata=src.nodata, dtype=src.meta['dtype']
                    )
                    resolution_factor = pow(2, 1)
                    lvlx_height = src.height / 2
                    lvlx_width = src.width / 2
                    lvlx_tranform = Affine(src.transform.a * resolution_factor,
                                           src.transform.b,
                                           src.transform.c,
                                           src.transform.d,
                                           src.transform.e * resolution_factor,
                                           src.transform.f)
                    vrt = WarpedVRT(src, transform=lvlx_tranform, width=lvlx_width, height=lvlx_height)
                    data = vrt.read(1)
            else:
                with self.get_dataset(level) as src:

                    profile.update(
                            nodata=src.nodata, dtype=src.meta['dtype']
                    )
                    data = src.read(1, window=tile[1])

            try:
                with rio.open(out_filepath, 'w', **profile) as dst:
                    window_out = Window(0, 0, size_x, size_y)
                    dst.write(data, window=window_out, indexes=1)

            except:
                print(profile)
                raise Exception

        with Parallel(n_jobs=self.cores) as parallel:
            for level in range(self._level_start, self._level_end + 1):
                x_count = self.tileinfos[level].countTilesX
                y_count = self.tileinfos[level].countTilesY
                self.tileinfos[level].report()
                nTiles = x_count * y_count
                desc = 'Level: %s' % level
                parallel(
                        delayed(make_tile)(level, self.get_tile(level, *divmod(idx, x_count)))
                        for idx in tqdm(range(nTiles), ascii=True, unit='tiles', desc=desc)
                )


if __name__ == '__main__':
    import datetime

    start = datetime.datetime.now()
    out_folder = r'/mainfs/scratch/nv1g17/tiller/L1'
    q = Dataset(r'/mainfs/scratch/nv1g17/tiller/l1mosaicpixelfixed_Max_opt.tif', cores=30, height=2048, width=2048,
                levels='0', output_folder=out_folder)
    q.slice_dataset()
    # w0 = q.windows[0][0]
    # q0 = q.get_tile(0, 0, 0)
    #
    # print(w0 == q0)
    # w0 = q.windows[1][0]
    # q0 = q.get_tile(1, 0, 0)
    # print(w0 == q0)
    end = datetime.datetime.now()
    print(end - start)
