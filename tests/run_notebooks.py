"""Execute demonstration notebooks locally; no production data is assumed."""
from pathlib import Path
import json
import time
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
report = []
for path in sorted((root / 'notebooks').glob('*.ipynb')):
    start = time.perf_counter()
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=240, kernel_name='python3',
                            resources={'metadata': {'path': str(root)}})
    client.execute()
    nbformat.write(nb, path)
    record = {'notebook': path.name, 'status': 'passed',
              'seconds': time.perf_counter() - start}
    report.append(record)
    print(record, flush=True)
(root / 'benchmarks/notebook_validation.json').write_text(json.dumps(report, indent=2))
