"""Parse all preserved-capability profiles against the installed backend CLI."""
import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, '/opt/dsv411017/scripts')
import profiles
import runtime

parser = argparse.ArgumentParser()
parser.add_argument('--project', type=Path, default=Path('/project'))
args = parser.parse_args()
import json
catalog = json.loads((args.project / 'configs/profiles.json').read_text())
with tempfile.TemporaryDirectory() as directory:
    for name in catalog['profiles']:
        record = profiles.resolve(args.project, name)
        profiles.validate_record(record, Path(directory))
        runtime.cli_test(record['configuration'])
        print('PROFILE_CLI_PASS=' + name, flush=True)
