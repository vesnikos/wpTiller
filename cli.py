from multiprocessing import cpu_count
import click

from wpTiller import Dataset

DEFAULT_CPUS = cpu_count() // 2


@click.command()
@click.argument('-i', '--input-file',
                type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
                required=True, help='Input raster. Must exist.')
@click.argument('-o', '--output-folder',
                type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
                required=True, help='Output folder. Must exist.')
@click.argument('-j', '--cpus', type=click.INT, default=DEFAULT_CPUS,
                help='How many cpus to use when generating the tiles. Defaults to half the system\'s available cpus')
@click.argument('--blocksize', type=click.INT, default=512,
                'The internal tile size. Default is 512.')
@click.argument('--height', type=click.INT, default=2048,
                help='The height of the generating tile in pixels. Default is 2048')
@click.argument('--width', type=click.INT, default=2048,
                help='The width of the generating tile in pixels. Default is 2048')
@click.argument('--levels', type=click.STRING, default='0-11',
                help='Levels of the raster to generate. 0 represents native resolution.'
                     ' Could be either a single level or range. eg 0-11 (default) or 4.'
                     'The previous level must exist to generating the next one.')
def main(input_file, output_folder, cpus, width, height, blocksize, levels):
    dataset = Dataset(input_raster=input_file, output_folder=output_folder,
                      width=width, height=height, cores=cpus, block_size=blocksize, levels=levels)

    dataset.slice_dataset()
