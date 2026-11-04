"""Run with the installed revision package; no real-data training is started."""
from pathlib import Path
import json, subprocess, sys
import raftup
root = Path(__file__).resolve().parents[1]
print('Installed package:', raftup.__file__, raftup.__version__, flush=True)
count = 0
for path in (root / 'examples/revision').rglob('*.ipynb'):
    if 'protocol_sources' in path.parts:
        continue
    notebook = json.loads(path.read_text())
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), f'{path}:cell{i}', 'exec')
    count += 1
print(f'Validated syntax of {count} portable notebooks', flush=True)
subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(root/'tests'), '-p', 'test_revision.py', '-v'], check=True)
namespace = {'__name__': '__main__'}
nb = json.loads((root/'examples/revision/00_installation_smoke_test.ipynb').read_text())
for cell in nb['cells']:
    if cell['cell_type']=='code':
        exec(compile(''.join(cell['source']), 'smoke_notebook', 'exec'), namespace)
print('Revision checks passed. Real-data training and fresh Conda/GPU setup are separate checks.')
