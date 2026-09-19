# 2D CPU Heat Sink FEM

Python-only quasi-2D finite-element thermal simulation of an air-cooled CPU heat sink.

## Contents

- `project1_2d_cpu_heatsink_fem.ipynb`: validated nine-cell reference workflow.
- `thermal_model.py`: reusable parameterized mesh, convection, FEM, and export engine.
- `simulation_tool.py`: Tkinter desktop tool for changing parameters, running the model, viewing plots, and saving results.
- `project_report.tex`: LaTeX project report.

## Run

```powershell
python -m pip install -r requirements.txt
python simulation_tool.py
```

The default model uses a 150 W die, 24 fins, and 4 m/s channel velocity. The GUI accepts dimensions in millimetres where shown. Press **Mesh** to generate and preview the mesh using independent horizontal and vertical sizes, a minimum layer-element count, and a triangle-diagonal pattern. Press **Run** afterward to solve only that stored mesh. Meshing and solving run in the background with progress, pause, resume, and stop controls. After completion, the Results menu displays one selected plot or the quantitative table at a time. The Save menu exports one selected figure/data product or the complete result package.

## Notebook

Open the notebook in Jupyter or VS Code and use Run All. The reference results are regenerated under `results/project1_heat_sink/`.

## Report

Compile the report with:

```powershell
pdflatex project_report.tex
```

The report describes the formulation, boundary conditions, correlations, validation checks, and baseline results.
# Launching the desktop tool

Use `launch_heatsink_gui.cmd`. It hard-codes the dedicated `heatsink-gui` interpreter, clears `PYTHONPATH`/`PYTHONHOME`, and disables the Python user site so Anaconda base or per-user packages cannot leak into the application. The startup check reports the executable and verifies NumPy, SciPy, Matplotlib, Tkinter, and the compiled optional packages before the GUI is created.

The pinned environment is recorded in `environment.yml`; `explicit-spec.txt` records the Conda package URLs. The GUI environment uses Python 3.9 with NumPy 1.24.3, SciPy 1.11.4, and Matplotlib 3.9.2. Matplotlib 3.11 on Python 3.11 was the native-crash combination on this Windows installation.
