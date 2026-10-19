"""Bake reviewed R1.9 SM80 overlays into the exact local runtime image."""
import hashlib
import json
from pathlib import Path
import shutil
import sysconfig

root = Path(__file__).resolve().parent
site = Path(sysconfig.get_paths()['purelib'])
manifest = json.loads((root / 'performance/manifest.json').read_text())
receipt = []
for row in manifest['files']:
    target = site / row['path']
    actual = hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    allowed = {row['base_sha256'], row.get('baseline_sha256'), row['patched_sha256']}
    if actual not in allowed:
        raise RuntimeError('Unexpected base bytes: ' + row['path'])
    source = root / 'performance/patched' / row['path']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['patched_sha256']:
        raise RuntimeError('Overlay payload hash mismatch: ' + row['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    receipt.append({'path': row['path'], 'sha256': row['patched_sha256']})
# The old recorder had a DP2-specific advisory formula. The new launcher
# reports measured capacity and never changes the requested C32 automatically.
capacity = site / 'dsv41_capacity.py'
text = capacity.read_text().replace("'required_concurrency': max(32, 2 * full_windows),",
                                 "'configured_global_concurrency': int(config.scheduler_config.max_num_seqs) * int(config.parallel_config.data_parallel_size),")
if "'required_concurrency'" in text:
    raise RuntimeError('Stale DP2 capacity formula')
capacity.write_text(text)
shutil.copyfile(root / 'dsv41_guard.py', site / 'dsv41_guard.py')
if (root / 'encoding/install.py').exists():
    import runpy
    runpy.run_path(str(root / 'encoding/install.py'), run_name='__main__')
    encoding_manifest = json.loads((root / 'encoding/manifest.json').read_text())
    receipt.extend({'path': item['path'], 'sha256': item['patched_sha256']}
                   for item in encoding_manifest['files'])
receipt.append({'path': 'dsv41_guard.py', 'sha256': hashlib.sha256((site / 'dsv41_guard.py').read_bytes()).hexdigest()})
for item in receipt:
    final_hash = hashlib.sha256((site / item['path']).read_bytes()).hexdigest()
    if final_hash != item['sha256']:
        item['upstream_overlay_sha256'] = item['sha256']
        item['sha256'] = final_hash
(root / 'installed-overlays.json').write_text(json.dumps({
    'source': manifest['base_commit'], 'overlay': manifest['id'],
    'files': receipt, 'gpu_executed': False,
    'capacity_recorder': 'TP/DP-derived global concurrency; no automatic resizing'
}, indent=2) + '\n')
print('Installed', len(receipt), 'reviewed SM80 overlays')
