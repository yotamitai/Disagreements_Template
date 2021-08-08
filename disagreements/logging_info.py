import json
import logging
from datetime import datetime
from os import makedirs
from os.path import abspath, exists, basename, join
from pathlib import Path

from disagreements.utils import make_clean_dirs


def get_logging(args):
    if not exists(abspath('logs')):
        makedirs('logs')
    name = '-'.join([basename(args.a1_name), basename(args.a2_name)])
    file_name = '_'.join([datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '_'), name])
    log_name = join('logs', file_name)
    args.output = join(args.results_dir, file_name)
    make_clean_dirs(args.output)
    with Path(join(args.output, 'metadata.json')).open('w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    logging.basicConfig(filename=log_name + '.log', filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    log(f'Comparing Agents: {name}', args.verbose)
    log(f'Disagreement importance by: {args.importance}', args.verbose)
    return name


def log(msg, verbose=False):
    if verbose: print(msg)
    logging.info(msg)