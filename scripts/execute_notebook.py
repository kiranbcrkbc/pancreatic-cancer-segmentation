"""
Notebook Executor for final001 (1).ipynb
Executes the authoritative MSD Task07 notebook top-to-bottom,
logging cell-by-cell progress, capturing stdout/stderr/rich display outputs,
and saving the fully rendered notebook in place.
"""

import sys
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def execute_notebook(notebook_path: str = "final001 (1).ipynb", timeout: int = 1200):
    nb_file = Path(notebook_path)
    if not nb_file.exists():
        raise FileNotFoundError(f"Notebook {notebook_path} not found.")

    print("=" * 80)
    print(f"EXECUTING NOTEBOOK: {notebook_path}")
    print("=" * 80)

    t_start = time.time()
    nb = nbformat.read(str(nb_file), as_version=4)

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=False,
    )

    # Custom execution wrapper to track cell-by-cell progress
    code_cell_idx = 0
    total_cells = len(nb.cells)

    with client.setup_kernel():
        for i, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                code_cell_idx += 1
                source_preview = cell.source.strip().split("\n")[0][:60]
                print(f"[Cell {i+1}/{total_cells} | Code #{code_cell_idx}] Executing: {source_preview}...", flush=True)
                t_cell = time.time()
                client.execute_cell(cell, i)
                dt_cell = time.time() - t_cell

                # Print brief output if any
                out_summary = []
                for out in cell.outputs:
                    if out.output_type == "stream":
                        first_line = out.text.strip().split("\n")[0][:80]
                        out_summary.append(first_line)
                    elif out.output_type == "display_data":
                        out_summary.append("<display_data / image>")
                summary_str = f" -> {out_summary[0]}" if out_summary else ""
                print(f"       -> Done in {dt_cell:.1f}s{summary_str}", flush=True)
            else:
                pass  # Markdown cell

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"NOTEBOOK EXECUTION COMPLETE in {total_time:.1f}s ({total_time/60:.2f} mins)")
    print(f"Saving fully rendered notebook with outputs to: {notebook_path}")
    print("=" * 80)

    nbformat.write(nb, str(nb_file))
    print("Notebook saved successfully.")

if __name__ == "__main__":
    notebook_to_run = sys.argv[1] if len(sys.argv) > 1 else "final001 (1).ipynb"
    execute_notebook(notebook_to_run)
