"""Execute from a fresh project kernel and retain the rendered notebook outputs."""
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

root = Path(__file__).parent.resolve()
notebook = nbformat.read(root / 'random_forest_warranty.ipynb', as_version=4)
manager = KernelManager(kernel_name='python3')
# Launch this exact interpreter without requiring a globally installed kernelspec.
manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
client = NotebookClient(notebook, km=manager, timeout=3600, resources={'metadata': {'path': str(root)}})
try:
    client.execute()
finally:
    nbformat.write(notebook, root / 'random_forest_warranty.ipynb')
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
print('Fresh-kernel notebook execution completed.')
