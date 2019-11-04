import os
import sys
import platform
import configparser
from pathlib import Path
from xml.etree import ElementTree

import osgeo

from errors import WpTillerError, WpTillerWarning

config = configparser.ConfigParser()
config.read('configuration.cfg')


def resolve(str):
    raise NotImplemented


def tile_children(zoom, src, ndigits=2):
    """

    :param zoom:
    :param x:
    :param y:
    :param src:
    :param path:
    :return:
    """

    def child_indexes(y_coord, x_coord):
        """
        Given tile x,y tile coord (eg. 1,1)
        return tile coordinates of a lower zoom level.

        :param x_coord: int
        :param y_coord: int
        :return: list of int
        """
        y_coord, x_coord = int(y_coord) * 2, int(x_coord) * 2

        # y,x , y,x+1, ...
        return [(str(y_coord).zfill(ndigits), str(x_coord).zfill(ndigits)),
                (str(y_coord).zfill(ndigits), str(x_coord - 1).zfill(ndigits)),
                (str(y_coord - 1).zfill(ndigits), str(x_coord).zfill(ndigits)),
                (str(y_coord - 1).zfill(ndigits), str(x_coord - 1).zfill(ndigits))]

    if zoom < 2:
        raise WpTillerError('zoom level must be bigger than 2.')

    src = Path(src)
    if src.is_absolute():
        path = src.parent.parent  # directory up
        path = path / str(zoom - 1)
    else:
        path = Path('.') / config['DEFAULT']['OutputSubdirectory'] / str(zoom - 1)

    # TODO: if problems replace this with a regex
    stem = '_'.join(src.stem.split('_')[:-2])
    x, y = src.stem.split('_')[-2:]
    stem = stem + '_{y}_{x}.tif'

    res = [path.joinpath(stem.format(y=y, x=x)) for (x, y) in child_indexes(y, x)]
    res = [x.as_posix() for x in res]

    return res


def buildvrt(src):
    """
    built a vrt file from the input src files. Removes non existing srcs
    Returns None if no file (or no-valid file was given.

    :param src: list of Str or Path
    :return: file handler
    """
    from subprocess import Popen, PIPE
    from tempfile import TemporaryFile
    vrt_exec = GdalFunctions.vrt_exec()

    if not isinstance(src, (list, tuple)):
        src = [src]

    # filter out non-existant src files
    src = [Path(x).as_posix() for x in src if Path(x).exists()]

    if len(src) < 1:
        return None

    cl = Popen(' '.join([vrt_exec,
                "-q",
                "/vsistdout/"] +  # virtual gdal output file, redirects the out to stdout
               src),
               shell=True, stdout=PIPE, stderr=PIPE)

    out, err = cl.communicate()
    # clean output from system newlines chars
    out = out.decode('utf8').replace(os.linesep, '')

    if err:
        print(' '.join([vrt_exec,
        "-q",
        "/vsistdout/"] +  # virtual gdal output file, redirects the out to stdout
       src,))
        raise WpTillerError('GdalBuild VRT encountered an error:\n%s' % err)
    if not out:
        WpTillerWarning('GdalBuild VRT returned an empty xml')
    try:
        ElementTree.fromstring(out)
    except ElementTree.ParseError:
        raise WpTillerError('GdalBuild VRT return a corrupted XML\n%s' % out)

    vrt_handle = TemporaryFile()
    vrt_handle.write(out.encode())
    vrt_handle.flush()
    vrt_handle.seek(0)

    return vrt_handle


class GdalFunctions:

    @staticmethod
    def gdal_root(path=None):
        """

        :param path:
        :return:
        """

        # is gdal installed in an exotic location? if yes overwrite the automatic
        if path:
            path = Path(path)
            return path

        os_name = platform.system()
        paths = []
        res = None
        if os_name == 'Windows':
            # Get from the registry the system paths.
            # Known location of registry based path locations
            sys_path = r'SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment'
            user_path = r'Environment'

            if sys.version_info.major == 3:
                import winreg
            else:  # sys.version_info.major == 2:
                import _winreg as winreg

            # user Path
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, user_path, 0, winreg.KEY_READ)
            query = winreg.QueryValueEx(key, 'Path')[0]
            for _ in query.split(';'):
                p = winreg.ExpandEnvironmentStrings(_)
                p = Path(p)
                paths.append(p)

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, sys_path, 0, winreg.KEY_READ)
            query = winreg.QueryValueEx(key, 'Path')[0]
            for _ in query.split(';'):
                p = winreg.ExpandEnvironmentStrings(_)
                p = Path(p)
                paths.append(p)
        elif os_name == 'Linux':
            env = os.environ['PATH'].split(':')
            for _ in env:
                _ = Path(_)
                paths.append(_)


        # Parse the evniroments for potential gdal roots
        for p in paths:
            # I assume where ever gdalinfo is, the rest of binaries are.
            q = list(p.glob('gdalinfo*'))
            if len(q) > 0:
                if len(q) > 1:
                    raise Exception('too many gdalinfos in %s ' % q)
                res = p
                break
        if res is None:
            raise Exception('gdal root folder not found. Have you installed gdal?')

        res = res.as_posix()
        return res

    @staticmethod
    def vrt_exec():
        gdal_root = GdalFunctions.gdal_root()
        gdal_root = Path(gdal_root)
        res = list(gdal_root.glob('gdalbuildvrt*'))
        if len(res) < 1:
            raise WpTillerError('gdalbuildvrt was not found in %s.' % gdal_root)
        res = res[0]
        res = res.as_posix()

        return res
