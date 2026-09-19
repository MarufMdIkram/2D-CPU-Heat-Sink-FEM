import csv
import json
import queue
import sys
import threading
import tkinter as tk
import re
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk
from startup_check import run_startup_check

if __name__ == "__main__":
    try:
        run_startup_check()
    except RuntimeError as error:
        print(f"Project 1 startup check failed:\n{error}", file=sys.stderr)
        raise SystemExit(2)

import matplotlib.tri as mtri
import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from material_library import FLUIDS, SOLID_MATERIALS, composite_material, library_material

from thermal_model import (
    SimulationCancelled,
    ThermalParameters,
    create_model,
    save_results,
    solve_model,
    temperature_limit_violations,
)

mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"


def scientific_display(text):
    """Return a compact Unicode rendering for table/status text."""
    subs = str.maketrans("0123456789aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
    def repl(match):
        suffix = match.group(1)
        rendered = suffix.translate(subs)
        rendered = rendered.replace("F", "₍F₎").replace("B", "₍B₎")
        return rendered
    return re.sub(r"_([A-Za-z0-9,∞]+)", repl, str(text))


def subscript_font(base_font):
    """Return a natural-sized subscript font derived from the GUI font."""
    font = base_font.copy()
    size = int(font.cget("size"))
    if size < 0:
        font.configure(size=min(-1, int(size * 0.80)))
    else:
        font.configure(size=max(1, int(round(size * 0.80))))
    return font


def scientific_label(parent, text):
    """Render a scientific label as one same-font canvas text run.

    The suffix is positioned on the same x-coordinate stream as the main
    text, with only its y-coordinate lowered.  This avoids the spacing that
    occurs when separate Tk widgets are packed beside one another.
    """
    value = str(text)
    default_font = tkfont.nametofont("TkDefaultFont")
    suffix_font = subscript_font(default_font)
    style = ttk.Style(parent)
    background = style.lookup("TFrame", "background") or parent.cget("background")
    parts = []
    cursor = 0
    for match in re.finditer(r"_([A-Za-z0-9,∞]+)", value):
        if match.start() > cursor:
            parts.append((value[cursor:match.start()], False))
        parts.append((match.group(1), True))
        cursor = match.end()
    if cursor < len(value):
        parts.append((value[cursor:], False))
    width = max(1, sum((suffix_font if lowered else default_font).measure(part) for part, lowered in parts) + 4)
    height = default_font.metrics("linespace") + 5
    canvas = tk.Canvas(parent, width=width, height=height, borderwidth=0,
                       highlightthickness=0, relief="flat", bg=background)
    x = 1
    baseline = default_font.metrics("ascent") + 1
    for part, lowered in parts:
        font = suffix_font if lowered else default_font
        canvas.create_text(x, baseline + (3 if lowered else 0), text=part,
                           anchor="sw", font=font, fill="#000000")
        x += font.measure(part)
    return canvas


class ScientificResultsTable(ttk.Frame):
    """Three-column results table with true same-size lowered subscripts."""

    def __init__(self, parent):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0,
                                background="white")
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._font = tkfont.nametofont("TkDefaultFont").copy()
        self._subscript_font = subscript_font(self._font)
        self._bold_font = self._font.copy(); self._bold_font.configure(weight="bold")

    def _scientific_text(self, x, baseline, text, bold=False):
        value = str(text)
        font = self._bold_font if bold else self._font
        cursor = 0
        for match in re.finditer(r"_([A-Za-z0-9,∞]+)", value):
            if match.start() > cursor:
                part = value[cursor:match.start()]
                self.canvas.create_text(x, baseline, text=part, anchor="sw", font=font, fill="black")
                x += font.measure(part)
            suffix = match.group(1)
            self.canvas.create_text(x, baseline + 3, text=suffix, anchor="sw", font=self._subscript_font, fill="black")
            x += self._subscript_font.measure(suffix)
            cursor = match.end()
        if cursor < len(value):
            part = value[cursor:]
            self.canvas.create_text(x, baseline, text=part, anchor="sw", font=font, fill="black")

    def set_rows(self, rows):
        self.canvas.delete("all")
        row_height = self._font.metrics("linespace") + 5
        y = row_height
        for quantity, value, unit in rows:
            category = not value and not unit
            self._scientific_text(6, y, quantity, bold=category)
            if not category:
                self.canvas.create_text(350, y, text=str(value), anchor="sw", font=self._font, fill="black")
                self.canvas.create_text(540, y, text=str(unit), anchor="sw", font=self._font, fill="black")
            y += row_height
        self.canvas.configure(scrollregion=(0, 0, 720, max(y + 4, 1)))


class ScientificInfoCanvas(tk.Canvas):
    """Multiline information display using same-size lowered subscripts."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, borderwidth=0, highlightthickness=0,
                         background=kwargs.pop("background", "#d9d9d9"), **kwargs)
        self._font = tkfont.nametofont("TkDefaultFont").copy()
        self._subscript_font = subscript_font(self._font)

    def set_text(self, text):
        self.delete("all")
        line_height = self._font.metrics("linespace") + 3
        max_width = 1
        for row, line in enumerate(str(text).splitlines()):
            baseline = (row + 1) * line_height
            x = 1
            cursor = 0
            for match in re.finditer(r"_([A-Za-z0-9,∞]+)", line):
                if match.start() > cursor:
                    part = line[cursor:match.start()]
                    self.create_text(x, baseline, text=part, anchor="sw", font=self._font, fill="black")
                    x += self._font.measure(part)
                suffix = match.group(1)
                self.create_text(x, baseline + 3, text=suffix, anchor="sw", font=self._subscript_font, fill="black")
                x += self._subscript_font.measure(suffix)
                cursor = match.end()
            if cursor < len(line):
                part = line[cursor:]
                self.create_text(x, baseline, text=part, anchor="sw", font=self._font, fill="black")
                x += self._font.measure(part)
            max_width = max(max_width, x + 2)
        self.configure(width=max_width, height=max(1, len(str(text).splitlines()) * line_height + 3))


class SimulationTool:
    RESULT_OPTIONS = {
        "Temperature contour": "temperature",
        "Heat-flux contour": "heat_flux",
        "Vertical centerline temperature": "centerline",
        "Heat-sink base-top temperature": "base_top",
        "Quantitative analysis": "quantitative",
    }

    SAVE_OPTIONS = ("Temperature contour", "Heat-flux contour", "Vertical centerline temperature", "Heat-sink base-top temperature", "Quantitative analysis", "Mesh NPZ", "Mesh CSV", "Mesh MSH", "Mesh VTK", "Mesh XDMF", "Nodal temperatures", "Element heat flux", "Simulation summary")

    PHYSICS_FIELDS = [
        ("Heat load, Q (W)", "heat_load"),
        ("Ambient temperature, T∞ (°C)", "ambient_temperature"),
        ("Fin count, N_F", "number_of_fins"),
        ("Fin thickness, W_F (mm)", "fin_thickness"),
        ("Fin height, H_F (mm)", "fin_height"),
        ("Base width, W_B (mm)", "base_width"),
        ("Base thickness, H_B (mm)", "base_thickness"),
        ("Fluid velocity, U (m/s)", "air_velocity"),
    ]

    MESH_FIELDS = [
        ("Horizontal size, h_x (mm)", "mesh_size_x"),
        ("Vertical size, h_y (mm)", "mesh_size_y"),
        ("Minimum layer elements, n_min", "minimum_layer_elements"),
    ]

    DIAGONAL_OPTIONS = {
        "Alternating diagonals": "alternating",
        "Rising diagonals (/)": "rising",
        "Falling diagonals (\\)": "falling",
    }

    MATERIAL_MODES = ("", "Library", "Custom")

    LENGTH_FIELDS = {"fin_thickness", "fin_height", "base_width", "base_thickness", "mesh_size_x", "mesh_size_y"}
    MESH_DEPENDENT_FIELDS = {
        "number_of_fins",
        "fin_thickness",
        "fin_height",
        "base_width",
        "base_thickness",
        "mesh_size_x",
        "mesh_size_y",
        "minimum_layer_elements",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("CPU Heat Sink FEM Simulation")
        self.root.geometry("1500x920")
        self.root.minsize(1200, 760)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.entries = {}
        self.material_modes = {"fin": "", "base": ""}
        self.selected_materials = {"fin": None, "base": None}
        self.material_detail_frames = {}
        self.custom_entries = {}
        self.composite_rows = []
        self.result = None
        self.mesh_model = None
        self.mesh_signature = None
        self.worker = None
        self.current_task = None
        self.message_queue = queue.Queue()
        self.run_event = None
        self.stop_event = None
        self.paused = False
        self._poll_after = None

        self._build_layout()
        self.draw_schematic()
        self._poll_after = self.root.after(75, self._poll_messages)

    def _build_layout(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(self.root)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.rowconfigure(0, weight=1)
        left_panel.columnconfigure(0, weight=1)
        controls_canvas = tk.Canvas(left_panel, highlightthickness=0, borderwidth=0)
        v_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=controls_canvas.yview)
        h_scroll = ttk.Scrollbar(left_panel, orient="horizontal", command=controls_canvas.xview)
        controls_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        controls_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        controls = ttk.Frame(controls_canvas, padding=10)
        controls_window = controls_canvas.create_window((0, 0), window=controls, anchor="nw")
        def update_scroll_region(_event=None):
            controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        def resize_controls(event):
            controls_canvas.itemconfigure(controls_window, width=max(event.width, controls.winfo_reqwidth()))
            update_scroll_region()
        controls.bind("<Configure>", update_scroll_region)
        controls_canvas.bind("<Configure>", resize_controls)
        self._left_pointer_inside = False
        controls_canvas.bind("<Enter>", lambda _e: setattr(self, "_left_pointer_inside", True))
        controls_canvas.bind("<Leave>", lambda _e: setattr(self, "_left_pointer_inside", False))
        controls_canvas.bind_all("<MouseWheel>", lambda event: controls_canvas.yview_scroll(int(-event.delta / 120), "units") if self._left_pointer_inside else None)
        controls_canvas.bind_all("<Shift-MouseWheel>", lambda event: controls_canvas.xview_scroll(int(-event.delta / 120), "units") if self._left_pointer_inside else None)
        controls.columnconfigure(0, weight=1)

        input_frame = ttk.LabelFrame(controls, text="Model inputs", padding=10)
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.columnconfigure(1, weight=1)
        defaults = ThermalParameters()
        for row, (label, key) in enumerate(self.PHYSICS_FIELDS):
            scientific_label(input_frame, label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(input_frame, width=14)
            value = getattr(defaults, key) * 1000 if key in self.LENGTH_FIELDS else getattr(defaults, key)
            entry.insert(0, f"{value:g}")
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            entry.bind("<KeyRelease>", lambda event, field=key: self._input_changed(field))
            self.entries[key] = entry

        mesh_frame = ttk.LabelFrame(controls, text="Mesh parameters", padding=10)
        mesh_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        mesh_frame.columnconfigure(1, weight=1)
        mesh_defaults = {
            "mesh_size_x": defaults.mesh_size,
            "mesh_size_y": defaults.mesh_size,
            "minimum_layer_elements": defaults.minimum_layer_elements,
        }
        for row, (label, key) in enumerate(self.MESH_FIELDS):
            scientific_label(mesh_frame, label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(mesh_frame, width=14)
            value = mesh_defaults[key] * 1000 if key in self.LENGTH_FIELDS else mesh_defaults[key]
            entry.insert(0, f"{value:g}")
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            entry.bind("<KeyRelease>", lambda event, field=key: self._input_changed(field))
            self.entries[key] = entry
        ttk.Label(mesh_frame, text="Triangle split").grid(row=len(self.MESH_FIELDS), column=0, sticky="w", padx=(0, 8), pady=3)
        self.diagonal_choice = tk.StringVar(value="Alternating diagonals")
        self.diagonal_selector = ttk.Combobox(
            mesh_frame,
            textvariable=self.diagonal_choice,
            values=list(self.DIAGONAL_OPTIONS),
            state="readonly",
            width=20,
        )
        self.diagonal_selector.grid(row=len(self.MESH_FIELDS), column=1, sticky="ew", pady=3)
        self.diagonal_selector.bind("<<ComboboxSelected>>", lambda event: self._input_changed("diagonal_pattern"))

        material_frame = ttk.LabelFrame(controls, text="Fin and base materials", padding=10)
        material_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        material_frame.columnconfigure(0, weight=1)
        self.material_mode_vars = {}
        self.material_mode_selectors = {}
        self.material_summary_vars = {}
        for row, region in enumerate(("fin", "base")):
            region_frame = ttk.Frame(material_frame)
            region_frame.grid(row=row, column=0, sticky="ew", pady=(0, 8))
            region_frame.columnconfigure(1, weight=1)
            label = "Fin material" if region == "fin" else "Base material"
            ttk.Label(region_frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
            mode_var = tk.StringVar(value="")
            selector = ttk.Combobox(region_frame, textvariable=mode_var, values=self.MATERIAL_MODES, state="readonly")
            selector.grid(row=0, column=1, sticky="ew", pady=3)
            selector.bind("<<ComboboxSelected>>", lambda event, selected_region=region: self._material_mode_changed(selected_region))
            self.material_mode_vars[region] = mode_var
            self.material_mode_selectors[region] = selector
            detail = ttk.Frame(region_frame)
            detail.grid(row=1, column=0, columnspan=2, sticky="ew", padx=(8, 0))
            detail.grid_remove()
            self.material_detail_frames[region] = detail
            self.material_summary_vars[region] = tk.StringVar(value="")

        fluid_frame = ttk.LabelFrame(controls, text="Surrounding fluid", padding=10)
        fluid_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        fluid_frame.columnconfigure(1, weight=1)
        ttk.Label(fluid_frame, text="Fluid type").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        self.fluid_choice = tk.StringVar(value="Air")
        self.fluid_selector = ttk.Combobox(fluid_frame, textvariable=self.fluid_choice, values=list(FLUIDS), state="readonly")
        self.fluid_selector.grid(row=0, column=1, sticky="ew", pady=2)
        self.fluid_selector.bind("<<ComboboxSelected>>", self._fluid_changed)
        self.fluid_properties_text = tk.StringVar()
        ttk.Label(fluid_frame, textvariable=self.fluid_properties_text, justify=tk.LEFT, wraplength=300).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self._update_fluid_summary()

        simulation_frame = ttk.LabelFrame(controls, text="Simulation control", padding=10)
        simulation_frame.grid(row=4, column=0, sticky="ew", pady=(10, 0))
        simulation_frame.columnconfigure(0, weight=1)
        simulation_frame.columnconfigure(1, weight=1)

        self.mesh_button = ttk.Button(simulation_frame, text="Mesh", command=self.mesh)
        self.run_button = ttk.Button(simulation_frame, text="Run", command=self.run, state=tk.DISABLED)
        self.pause_button = ttk.Button(simulation_frame, text="Pause", command=self.pause, state=tk.DISABLED)
        self.resume_button = ttk.Button(simulation_frame, text="Resume", command=self.resume, state=tk.DISABLED)
        self.stop_button = ttk.Button(simulation_frame, text="Stop", command=self.stop, state=tk.DISABLED)
        self.mesh_button.grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.run_button.grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.pause_button.grid(row=1, column=0, sticky="ew", padx=(0, 3), pady=2)
        self.resume_button.grid(row=1, column=1, sticky="ew", padx=(3, 0), pady=2)
        self.stop_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)

        self.progress_value = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(simulation_frame, variable=self.progress_value, maximum=100, mode="determinate")
        self.progress_bar.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(9, 2))
        self.progress_text = tk.StringVar(value="0%")
        ttk.Label(simulation_frame, textvariable=self.progress_text, anchor="center").grid(row=4, column=0, columnspan=2, sticky="ew")
        self.status = tk.StringVar(value="Ready")
        ttk.Label(simulation_frame, textvariable=self.status, wraplength=250, anchor="center").grid(row=5, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.mesh_info = tk.StringVar(value="Mesh information: not generated")
        self.simulation_info = tk.StringVar(value="Simulation information: not run")
        self.mesh_info_canvas = ScientificInfoCanvas(simulation_frame, width=280, height=22)
        self.simulation_info_canvas = ScientificInfoCanvas(simulation_frame, width=280, height=22)
        self.mesh_info_canvas.grid(row=6, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.simulation_info_canvas.grid(row=7, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.mesh_info.trace_add("write", lambda *_: self.mesh_info_canvas.set_text(self.mesh_info.get()))
        self.simulation_info.trace_add("write", lambda *_: self.simulation_info_canvas.set_text(self.simulation_info.get()))
        self.mesh_info_canvas.set_text(self.mesh_info.get())
        self.simulation_info_canvas.set_text(self.simulation_info.get())

        save_frame = ttk.LabelFrame(controls, text="Save selected output", padding=10)
        save_frame.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        save_frame.columnconfigure(0, weight=1)
        self.save_choice = tk.StringVar(value="")
        for index, name in enumerate(self.SAVE_OPTIONS):
            ttk.Radiobutton(save_frame, text=name, value=name, variable=self.save_choice).grid(row=index, column=0, sticky="w")
        self.save_button = ttk.Button(save_frame, text="Save", command=self.save_selected, state=tk.DISABLED)
        self.save_button.grid(row=len(self.SAVE_OPTIONS), column=0, sticky="ew", pady=(6, 2))

        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.schematic_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.schematic_tab, text="Parameter schematic")
        self.notebook.add(self.results_tab, text="Results")

        self.schematic_tab.rowconfigure(0, weight=1)
        self.schematic_tab.columnconfigure(0, weight=1)
        self.schematic_plot_frame = ttk.Frame(self.schematic_tab)
        self.schematic_plot_frame.grid(row=0, column=0, sticky="nsew")
        self.material_panel = ttk.Frame(self.schematic_tab, padding=8, width=310)
        self.material_panel.grid(row=0, column=1, sticky="nsew")
        self.material_panel.grid_remove()
        self.schematic_figure = Figure(figsize=(10, 7), dpi=100)
        self.schematic_canvas = FigureCanvasTkAgg(self.schematic_figure, master=self.schematic_plot_frame)
        self.schematic_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.results_tab.rowconfigure(1, weight=1)
        self.results_tab.columnconfigure(0, weight=1)
        selector_bar = ttk.Frame(self.results_tab, padding=10)
        selector_bar.grid(row=0, column=0, sticky="ew")
        selector_bar.columnconfigure(1, weight=1)
        ttk.Label(selector_bar, text="Result to display:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.result_choice = tk.StringVar(value="")
        self.result_selector = ttk.Combobox(
            selector_bar,
            textvariable=self.result_choice,
            values=list(self.RESULT_OPTIONS),
            state=tk.DISABLED,
        )
        self.result_selector.grid(row=0, column=1, sticky="ew")
        self.result_selector.bind("<<ComboboxSelected>>", self.show_selected_result)

        self.result_display = ttk.Frame(self.results_tab)
        self.result_display.grid(row=1, column=0, sticky="nsew")
        self.result_display.rowconfigure(1, weight=1)
        self.result_display.columnconfigure(0, weight=1)
        self.warning_banner = ttk.Label(self.result_display, text="", foreground="#a00000", wraplength=950, justify=tk.LEFT)
        self.warning_banner.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))
        self.warning_banner.grid_remove()

        self.result_figure = Figure(figsize=(10, 7), dpi=100)
        self.result_canvas = FigureCanvasTkAgg(self.result_figure, master=self.result_display)
        self.result_widget = self.result_canvas.get_tk_widget()

        self.empty_result_label = ttk.Label(
            self.result_display,
            text="Press Mesh to build the model, then press Run to solve it.",
            anchor="center",
            font=("Segoe UI", 12),
        )
        self.empty_result_label.grid(row=1, column=0, sticky="nsew")

        self.table_frame = ttk.Frame(self.result_display, padding=12)
        self.table_frame.rowconfigure(0, weight=1)
        self.table_frame.columnconfigure(0, weight=1)
        self.quantitative_table = ScientificResultsTable(self.table_frame)
        self.quantitative_table.grid(row=0, column=0, sticky="nsew")

    def _input_changed(self, field):
        if self.worker is not None and self.worker.is_alive():
            return
        if field in self.MESH_DEPENDENT_FIELDS or field in {"diagonal_pattern", "materials", "fluid"}:
            self.mesh_model = None
            self.mesh_signature = None
            self.run_button.configure(state=tk.DISABLED)
            self.status.set("Mesh inputs changed. Press Mesh to rebuild it.")
        try:
            self.draw_schematic()
        except (TypeError, ValueError):
            pass

    def _fluid_changed(self, _event=None):
        self._update_fluid_summary()
        self._input_changed("fluid")

    def _update_fluid_summary(self):
        fluid = FLUIDS[self.fluid_choice.get()]
        self.fluid_properties_text.set(
            f"ρ = {fluid['density']:g} kg/m³   μ = {fluid['viscosity']:g} Pa·s\n"
            f"k = {fluid['conductivity']:g} W/(m·K)   cₚ = {fluid['specific_heat']:g} J/(kg·K)"
        )

    def _material_mode_changed(self, region):
        self.material_modes[region] = self.material_mode_vars[region].get()
        self.selected_materials[region] = None
        self._input_changed("materials")
        self._refresh_material_inline(region)

    def _refresh_material_panel(self):
        for region in ("fin", "base"):
            self._refresh_material_inline(region)
        self.draw_schematic()

    def _apply_material(self, material):
        region = getattr(self, "_editing_region", "fin")
        self.selected_materials[region] = material
        self._input_changed("materials")
        self._refresh_material_inline(region)
        self.draw_schematic()

    def _refresh_material_inline(self, region):
        frame = self.material_detail_frames[region]
        for child in frame.winfo_children():
            child.destroy()
        mode = self.material_modes[region]
        if not mode:
            frame.grid_remove()
            return
        frame.grid()
        self._editing_region = region
        if mode == "Library":
            var = tk.StringVar(value=self.selected_materials[region]["name"] if self.selected_materials[region] else "")
            box = ttk.Combobox(frame, textvariable=var, values=list(SOLID_MATERIALS), state="readonly")
            box.pack(fill=tk.X, pady=(0, 3))
            box.bind("<<ComboboxSelected>>", lambda _e, r=region, v=var: self._apply_material_for_region(r, library_material(v.get())))
        elif mode == "Custom":
            fields = (("Custom material name", "name"), ("Thermal conductivity k", "conductivity"), ("Density ρ", "density"), ("Specific heat cp", "specific_heat"), ("Temperature limit", "critical_temperature_C"))
            entries = {}
            for label, key in fields:
                ttk.Label(frame, text=label).pack(anchor="w")
                entry = ttk.Entry(frame)
                entry.pack(fill=tk.X, pady=(0, 2))
                entries[key] = entry
            self.custom_entries[region] = entries
            ttk.Button(frame, text="Apply custom material", command=lambda r=region: self._apply_custom_inline(r)).pack(fill=tk.X, pady=(2, 3))
        current = self.selected_materials[region]
        if current:
            ttk.Label(frame, text=(f"Selected: {current['name']}\nρ = {current['density']:g} kg/m³\nk = {current['conductivity']:g} W/(m·K)\ncp = {current['specific_heat']:g} J/(kg·K)\nTemperature limit = {current['critical_temperature_C']:g} °C"), wraplength=300).pack(anchor="w")

    def _apply_material_for_region(self, region, material):
        self._editing_region = region
        self._apply_material(material)

    def _apply_custom_inline(self, region):
        try:
            entries = self.custom_entries[region]
            material = {"name": entries["name"].get().strip(), "conductivity": float(entries["conductivity"].get()), "density": float(entries["density"].get()), "specific_heat": float(entries["specific_heat"].get()), "critical_temperature_C": float(entries["critical_temperature_C"].get()), "limit_kind": "user-defined material limit"}
            if not material["name"] or any(not np.isfinite(material[k]) or material[k] <= 0 for k in material if k not in {"name", "limit_kind"}):
                raise ValueError("Enter a valid name and positive finite material properties")
            self._editing_region = region
            self._apply_material(material)
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid custom material", str(error), parent=self.root)

    def _build_library_picker(self):
        ttk.Label(self.material_panel, text="Select a material from the library:").pack(anchor="w")
        picker_frame = ttk.Frame(self.material_panel)
        picker_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        listbox = tk.Listbox(picker_frame, height=13, exportselection=False, font=("Times New Roman", 10))
        scrollbar = ttk.Scrollbar(picker_frame, orient="vertical", command=listbox.yview)
        listbox.configure(yscrollcommand=scrollbar.set)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        for name in SOLID_MATERIALS:
            listbox.insert(tk.END, name)

        def choose(_event=None):
            selection = listbox.curselection()
            if selection:
                self._apply_material(library_material(listbox.get(selection[0])))

        listbox.bind("<Double-Button-1>", choose)
        ttk.Button(self.material_panel, text="Use selected material", command=choose).pack(fill=tk.X, pady=(2, 0))

    def _build_custom_editor(self):
        fields = [
            ("Custom material name", "name", ""),
            ("Conductivity, k (W/m·K)", "conductivity", ""),
            ("Density, ρ (kg/m³)", "density", ""),
            ("Specific heat, cₚ (J/kg·K)", "specific_heat", ""),
            ("Temperature limit (°C)", "critical_temperature_C", ""),
        ]
        self.custom_entries = {}
        for label, key, initial in fields:
            ttk.Label(self.material_panel, text=label).pack(anchor="w", pady=(3, 0))
            entry = ttk.Entry(self.material_panel)
            entry.insert(0, initial)
            entry.pack(fill=tk.X)
            self.custom_entries[key] = entry
        ttk.Label(self.material_panel, text="Enter a melting or degradation limit for warning checks.", wraplength=275).pack(anchor="w", pady=6)
        ttk.Button(self.material_panel, text="Apply custom material", command=self._apply_custom_material).pack(fill=tk.X)

    def _apply_custom_material(self):
        try:
            material = {
                "name": self.custom_entries["name"].get().strip(),
                "conductivity": float(self.custom_entries["conductivity"].get()),
                "density": float(self.custom_entries["density"].get()),
                "specific_heat": float(self.custom_entries["specific_heat"].get()),
                "critical_temperature_C": float(self.custom_entries["critical_temperature_C"].get()),
                "limit_kind": "user-defined material limit",
            }
            if not material["name"]:
                raise ValueError("Enter a custom material name")
            for key in ("conductivity", "density", "specific_heat", "critical_temperature_C"):
                if not np.isfinite(material[key]) or material[key] <= 0:
                    raise ValueError(f"{key.replace('_', ' ').capitalize()} must be finite and positive")
            self._apply_material(material)
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid custom material", str(error), parent=self.root)

    def _build_composite_editor(self):
        ttk.Label(self.material_panel, text="Number of constituents:").pack(anchor="w")
        self.composite_count_var = tk.IntVar(value=2)
        count = ttk.Spinbox(self.material_panel, from_=2, to=6, textvariable=self.composite_count_var, width=8, command=self._rebuild_composite_rows)
        count.pack(anchor="w", pady=(0, 5))
        count.bind("<KeyRelease>", lambda _event: self._rebuild_composite_rows())
        self.composite_rows_frame = ttk.Frame(self.material_panel)
        self.composite_rows_frame.pack(fill=tk.BOTH, expand=True)
        self._rebuild_composite_rows()
        ttk.Label(self.material_panel, text="Fractions are volume fractions and must sum to 1.0.", wraplength=275).pack(anchor="w", pady=5)
        ttk.Button(self.material_panel, text="Calculate and apply composite", command=self._apply_composite).pack(fill=tk.X)

    def _rebuild_composite_rows(self):
        if not hasattr(self, "composite_rows_frame") or not self.composite_rows_frame.winfo_exists():
            return
        for child in self.composite_rows_frame.winfo_children():
            child.destroy()
        self.composite_rows = []
        try:
            count = max(2, min(6, int(self.composite_count_var.get())))
        except (ValueError, tk.TclError):
            return
        for index in range(count):
            row = ttk.Frame(self.composite_rows_frame)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{index + 1}").pack(side=tk.LEFT, padx=(0, 4))
            material_var = tk.StringVar(value="")
            material_box = ttk.Combobox(row, textvariable=material_var, values=("", *SOLID_MATERIALS), state="readonly", width=20)
            material_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
            fraction_entry = ttk.Entry(row, width=7)
            fraction_entry.pack(side=tk.RIGHT, padx=(4, 0))
            fraction_entry.insert(0, "")
            ttk.Label(row, text="φ").pack(side=tk.RIGHT)
            self.composite_rows.append((material_var, fraction_entry))

    def _apply_composite(self):
        try:
            components = []
            for material_var, fraction_entry in self.composite_rows:
                name = material_var.get()
                if not name:
                    raise ValueError("Choose a library material for every composite constituent")
                material = library_material(name)
                material["volume_fraction"] = float(fraction_entry.get())
                components.append(material)
            self._apply_material(composite_material(components))
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid composite", str(error), parent=self.root)

    def parameters(self):
        values = {}
        for key, entry in self.entries.items():
            text = entry.get().strip()
            if not text:
                raise ValueError(f"A value is required for {key.replace('_', ' ')}")
            values[key] = float(text)
        if not values["number_of_fins"].is_integer():
            raise ValueError("Fin count, N_F, must be a whole number")
        if not values["minimum_layer_elements"].is_integer():
            raise ValueError("Minimum layer elements, n_min, must be a whole number")
        values["number_of_fins"] = int(values["number_of_fins"])
        values["minimum_layer_elements"] = int(values["minimum_layer_elements"])
        for key in self.LENGTH_FIELDS:
            values[key] /= 1000
        values["mesh_size"] = values["mesh_size_x"]
        values["diagonal_pattern"] = self.DIAGONAL_OPTIONS[self.diagonal_choice.get()]
        if self.selected_materials["fin"] is None or self.selected_materials["base"] is None:
            raise ValueError("Choose and apply a material for both the fins and the base")
        values["fin_material"] = dict(self.selected_materials["fin"])
        values["base_material"] = dict(self.selected_materials["base"])
        values["fluid_name"] = self.fluid_choice.get()
        values["fluid"] = {"name": self.fluid_choice.get(), **FLUIDS[self.fluid_choice.get()]}
        if values["number_of_fins"] < 2 or values["number_of_fins"] * values["fin_thickness"] >= values["base_width"]:
            raise ValueError("Invalid geometry: require N_F ≥ 2 and N_F·W_F < W_B")
        if any(values[key] <= 0 for key in ("heat_load", "fin_thickness", "fin_height", "base_width", "base_thickness", "mesh_size_x", "mesh_size_y")):
            raise ValueError("Physical dimensions, heat load, and mesh sizes must be positive")
        return ThermalParameters(**values)

    @staticmethod
    def _mesh_signature(parameters):
        return (
            parameters.number_of_fins,
            parameters.fin_thickness,
            parameters.fin_height,
            parameters.base_width,
            parameters.base_thickness,
            parameters.mesh_size_x,
            parameters.mesh_size_y,
            parameters.minimum_layer_elements,
            parameters.diagonal_pattern,
            json.dumps(parameters.fin_material, sort_keys=True),
            json.dumps(parameters.base_material, sort_keys=True),
            parameters.fluid_name,
        )

    @staticmethod
    def _dimension(axis, start, end, label, text_position, color="#333333"):
        if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
            gap = 0.42
            left = text_position[0] - gap
            right = text_position[0] + gap
            axis.plot([start[0], left], [start[1], start[1]], color=color, lw=1.4, marker="<", markevery=[0], ms=5)
            axis.plot([right, end[0]], [end[1], end[1]], color=color, lw=1.4, marker=">", markevery=[1], ms=5)
        else:
            gap = 0.18
            lower = text_position[1] - gap
            upper = text_position[1] + gap
            axis.plot([start[0], start[0]], [start[1], lower], color=color, lw=1.4, marker="<", markevery=[0], ms=5)
            axis.plot([end[0], end[0]], [upper, end[1]], color=color, lw=1.4, marker=">", markevery=[1], ms=5)
        axis.text(text_position[0], text_position[1], label, color=color, ha="center", va="center", fontsize=10)

    def draw_schematic(self, figure=None, parameters=None):
        figure = figure or self.schematic_figure
        figure.clear(); axis = figure.add_subplot(111); axis.set_aspect("equal"); axis.axis("off")
        if parameters is None:
            try:
                values = {key: float(entry.get()) for key, entry in self.entries.items() if entry.get().strip()}
                for key in self.LENGTH_FIELDS:
                    if key in values: values[key] /= 1000
                defaults = ThermalParameters()
                parameters = ThermalParameters(
                    heat_load=values.get("heat_load", defaults.heat_load), ambient_temperature=values.get("ambient_temperature", defaults.ambient_temperature),
                    number_of_fins=int(values.get("number_of_fins", defaults.number_of_fins)), fin_thickness=values.get("fin_thickness", defaults.fin_thickness), fin_height=values.get("fin_height", defaults.fin_height), base_width=values.get("base_width", defaults.base_width), base_thickness=values.get("base_thickness", defaults.base_thickness), air_velocity=values.get("air_velocity", defaults.air_velocity),
                    fin_material=self.selected_materials.get("fin") or defaults.fin_material, base_material=self.selected_materials.get("base") or defaults.base_material,
                )
            except Exception: parameters = ThermalParameters()
        wb, tf, hf, hb = parameters.base_width * 1000, parameters.fin_thickness * 1000, parameters.fin_height * 1000, parameters.base_thickness * 1000
        nf, u, tamb, q = parameters.number_of_fins, parameters.air_velocity, parameters.ambient_temperature, parameters.heat_load
        spacing = (wb - nf * tf) / (nf - 1)
        if nf < 2 or tf <= 0 or hf <= 0 or wb <= 0 or hb <= 0 or spacing <= 0: raise ValueError("Invalid schematic geometry: require N_F ≥ 2, N_F·W_F < W_B, and positive dimensions")
        L, R, B, BT, FT = 2.0, 10.0, 3.0, 4.0, 8.15; scale = (R-L) / wb
        fw, sw = tf * scale, spacing * scale
        fin_positions = [L + i * (fw + sw) for i in range(nf)]
        def block(x0, x1, y0, y1, color):
            rgb = mpl.colors.to_rgb(color)
            axis.imshow(np.ones((2, 2, 3)) * rgb, extent=(x0, x1, y0, y1), origin="lower", interpolation="nearest", zorder=1, aspect="auto")
            axis.plot([x0,x1,x1,x0,x0], [y0,y0,y1,y1,y0], color="black", lw=.7, zorder=2)
        for x in fin_positions: block(x, x + fw, BT, FT, "#4472c4")
        base_record = getattr(self, "selected_materials", {}).get("base")
        base_label = (base_record or {}).get("name", "Aluminum base")
        layers = [(L, R, B, BT, "#ffc000", base_label), (3.15, 8.85, 2.62, 3.00, "#70ad47", "TIM 2"), (2.65, 9.35, 1.86, 2.62, "#a5a5a5", "Copper IHS"), (4.00, 8.00, 1.50, 1.86, "#5b9bd5", "TIM 1"), (4.55, 7.45, .72, 1.50, "#f4b183", "Silicon die")]
        for x0,x1,y0,y1,color,label in layers:
            block(x0, x1, y0, y1, color); axis.text((x0+x1)/2,(y0+y1)/2,label,ha="center",va="center",fontsize=9,weight="bold",color="black",zorder=3)
        axis.text(6, 9.25, rf"$\odot$ Fluid Flow: U = {u:g} m/s, T∞ = {tamb:g} °C", ha="center", va="center", fontsize=10, weight="bold", fontfamily="Times New Roman")
        for x in np.linspace(4.85, 7.15, 7):
            axis.plot([x, x], [.03, .62], color="#b00000", lw=1.4, marker="^", markevery=[1])
        axis.text(6,-.18,"Heat Load",ha="center",va="top",fontsize=10,weight="bold",color="black",fontfamily="Times New Roman")
        axis.set_xlim(.6,11.4); axis.set_ylim(-.95,9.75)
        figure.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.91)
        if figure is self.schematic_figure:
            self.schematic_canvas.draw_idle()

    def mesh(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            parameters = self.parameters()
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error))
            return
        self.draw_schematic(parameters=parameters)

        self.mesh_model = None
        self.mesh_signature = None
        self.result = None
        self.result_choice.set("")
        self.save_choice.set("")
        self._show_empty_result("Generate the mesh, then press Run to solve the model.")
        self.result_selector.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self._start_task("mesh", "Generating mesh…")
        self.worker = threading.Thread(target=self._mesh_worker, args=(parameters,), daemon=True)
        self.worker.start()

    def run(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            parameters = self.parameters()
        except (TypeError, ValueError) as error:
            messagebox.showerror("Invalid input", str(error))
            return
        self.draw_schematic(parameters=parameters)
        if self.mesh_model is None or self.mesh_signature != self._mesh_signature(parameters):
            messagebox.showinfo("Mesh required", "Press Mesh to build the current geometry before running the simulation.")
            self.run_button.configure(state=tk.DISABLED)
            return

        self.result = None
        self.result_choice.set("")
        self.save_choice.set("")
        self._show_empty_result("Simulation is running. Results will remain blank until it finishes.")
        self.result_selector.configure(state=tk.DISABLED)
        self.save_button.configure(state=tk.DISABLED)
        self._start_task("solve", "Starting simulation…")
        self.worker = threading.Thread(target=self._simulation_worker, args=(parameters,), daemon=True)
        self.worker.start()

    def _start_task(self, task, status):
        self.current_task = task
        self.progress_value.set(0)
        self.progress_text.set("0%")
        self.status.set(status)
        self.run_event = threading.Event()
        self.run_event.set()
        self.stop_event = threading.Event()
        self.paused = False
        self._set_running_controls(True)
        self._set_entries_state(tk.DISABLED)

    def _mesh_worker(self, parameters):
        def report(percent, message):
            scaled_percent = min(100.0, percent * 100.0 / 30.0)
            self.message_queue.put(("progress", scaled_percent, message))

        try:
            model = create_model(
                parameters,
                progress_callback=report,
                run_event=self.run_event,
                stop_event=self.stop_event,
            )
            self.message_queue.put(("mesh_complete", model, self._mesh_signature(parameters)))
        except SimulationCancelled as error:
            self.message_queue.put(("cancelled", str(error)))
        except Exception as error:
            self.message_queue.put(("error", str(error)))

    def _simulation_worker(self, parameters):
        def report(percent, message):
            scaled_percent = min(100.0, max(0.0, (percent - 30.0) * 100.0 / 70.0))
            self.message_queue.put(("progress", scaled_percent, message))

        try:
            result = solve_model(
                parameters,
                self.mesh_model,
                progress_callback=report,
                run_event=self.run_event,
                stop_event=self.stop_event,
            )
            violations = temperature_limit_violations(result)
            if violations:
                self.message_queue.put(("temperature_limit", result, violations))
            else:
                self.message_queue.put(("complete", result))
        except SimulationCancelled as error:
            self.message_queue.put(("cancelled", str(error)))
        except Exception as error:
            self.message_queue.put(("error", str(error)))

    def pause(self):
        if self.worker is not None and self.worker.is_alive() and not self.paused:
            self.paused = True
            self.run_event.clear()
            self.pause_button.configure(state=tk.DISABLED)
            self.resume_button.configure(state=tk.NORMAL)
            self.status.set("Meshing paused" if self.current_task == "mesh" else "Simulation paused")

    def resume(self):
        if self.worker is not None and self.worker.is_alive() and self.paused:
            self.paused = False
            self.run_event.set()
            self.pause_button.configure(state=tk.NORMAL)
            self.resume_button.configure(state=tk.DISABLED)
            self.status.set("Meshing resumed" if self.current_task == "mesh" else "Simulation resumed")

    def stop(self):
        if self.worker is not None and self.worker.is_alive():
            self.stop_event.set()
            self.run_event.set()
            self.pause_button.configure(state=tk.DISABLED)
            self.resume_button.configure(state=tk.DISABLED)
            self.stop_button.configure(state=tk.DISABLED)
            self.status.set("Stopping meshing…" if self.current_task == "mesh" else "Stopping simulation…")

    def _poll_messages(self):
        try:
            while True:
                message = self.message_queue.get_nowait()
                kind = message[0]
                if kind == "progress":
                    _, percent, text = message
                    self.progress_value.set(percent)
                    self.progress_text.set(f"{percent:.0f}%")
                    if not self.paused and not (self.stop_event and self.stop_event.is_set()):
                        self.status.set(text)
                elif kind == "mesh_complete":
                    self._mesh_complete(message[1], message[2])
                elif kind == "complete":
                    self._simulation_complete(message[1])
                elif kind == "temperature_limit":
                    self._temperature_limit_prompt(message[1], message[2])
                elif kind == "cancelled":
                    self._simulation_cancelled(message[1])
                elif kind == "error":
                    self._simulation_failed(message[1])
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self._poll_after = self.root.after(75, self._poll_messages)

    def _mesh_complete(self, model, signature):
        self.mesh_model = model
        self.mesh_signature = signature
        self.current_task = None
        self.progress_value.set(100)
        self.progress_text.set("100%")
        self.status.set(f"Mesh ready: {len(model['nodes']):,} nodes, {len(model['elements']):,} elements")
        self.mesh_info.set(f"Mesh information\nNumber of nodes: {len(model['nodes']):,}\nNumber of triangular elements: {len(model['elements']):,}\nNumber of convective boundary edges: {len(model['convective_edges']):,}\nMesh status: valid")
        self._set_running_controls(False)
        self._set_entries_state(tk.NORMAL)
        self.show_mesh_popup()

    def _simulation_complete(self, result):
        self.result = result
        self.current_task = None
        self.progress_value.set(100)
        self.progress_text.set("100%")
        self.status.set("Simulation complete with temperature-limit warning" if result.get("continued_past_limit") else "Simulation complete")
        self.simulation_info.set(f"Simulation information\nStatus: complete\nT_j,max: {result['junction_temperature']:.3f} °C\nR_ja: {result['R_ja']:.6g} K/W\nEnergy-balance error: {result['energy_balance_error']:.3e}\nThermal target: {'PASS' if result['target_pass'] else 'FAIL'}")
        self._set_running_controls(False)
        self._set_entries_state(tk.NORMAL)
        self.result_selector.configure(state="readonly")
        self.save_button.configure(state=tk.NORMAL)
        violations = result.get("temperature_limit_violations", [])
        if violations and result.get("continued_past_limit"):
            lines = ["WARNING: Local temperature exceeded a recorded material or fluid limit. Results use the requested single-phase/non-degraded continuation assumption."]
            lines.extend(
                f"{item['name']}: {item['temperature_C']:.2f} °C > {item['limit_C']:.2f} °C ({item['limit_kind']}) at x={item['location_mm'][0]:.2f} mm, y={item['location_mm'][1]:.2f} mm."
                for item in violations
            )
            self.warning_banner.configure(text="\n".join(lines))
            self.warning_banner.grid()
        else:
            self.warning_banner.configure(text="")
            self.warning_banner.grid_remove()
        self._show_empty_result("Choose one result from the menu above.")

    def _temperature_limit_prompt(self, result, violations):
        self.current_task = None
        self._set_running_controls(False)
        self._set_entries_state(tk.NORMAL)
        self.progress_value.set(100)
        self.progress_text.set("100%")
        self.status.set("Temperature limit exceeded; waiting for your choice")
        summary = "\n".join(
            f"• {item['name']}: {item['temperature_C']:.2f} °C exceeds {item['limit_C']:.2f} °C ({item['limit_kind']})."
            for item in violations
        )
        dialog = tk.Toplevel(self.root)
        dialog.title("Temperature limit exceeded")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.protocol("WM_DELETE_WINDOW", lambda: choose_stop())
        ttk.Label(
            dialog,
            text="A local solved temperature is above a recorded material or fluid limit:\n\n" + summary + "\n\nStop this run, or force continuation using the current single-phase and non-degraded model.",
            wraplength=520,
            justify=tk.LEFT,
            padding=16,
        ).pack(fill=tk.BOTH, expand=True)
        buttons = ttk.Frame(dialog, padding=(12, 0, 12, 12))
        buttons.pack(fill=tk.X)
        choice = {"continue": False}

        def choose_stop():
            choice["continue"] = False
            dialog.destroy()

        def choose_continue():
            choice["continue"] = True
            dialog.destroy()

        ttk.Button(buttons, text="Stop and discard results", command=choose_stop).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(buttons, text="Force continue", command=choose_continue).pack(side=tk.RIGHT)
        dialog.grab_set()
        self.root.wait_window(dialog)
        if choice["continue"]:
            result["temperature_limit_violations"] = violations
            result["continued_past_limit"] = True
            self._simulation_complete(result)
        else:
            self.result = None
            self.status.set("Stopped because a temperature limit was exceeded")
            self._show_empty_result("Results discarded after the material/fluid temperature warning.")

    def _simulation_cancelled(self, message):
        stopped_task = self.current_task
        self.current_task = None
        self.status.set("Meshing stopped" if stopped_task == "mesh" else "Simulation stopped")
        self._set_running_controls(False)
        self._set_entries_state(tk.NORMAL)
        if stopped_task == "mesh":
            self.mesh_model = None
            self.mesh_signature = None
            self._show_empty_result("Meshing stopped. Press Mesh to start again.")
        else:
            self._show_empty_result("Simulation stopped. Press Run to start again.")

    def _simulation_failed(self, message):
        failed_task = self.current_task
        self.current_task = None
        self.status.set("Meshing failed" if failed_task == "mesh" else "Simulation failed")
        self._set_running_controls(False)
        self._set_entries_state(tk.NORMAL)
        if failed_task == "mesh":
            self.mesh_model = None
            self.mesh_signature = None
            self._show_empty_result("The mesh could not be generated.")
        else:
            self._show_empty_result("The simulation failed; no results are available.")
        messagebox.showerror("Mesh error" if failed_task == "mesh" else "Simulation error", message)

    def _set_running_controls(self, running):
        self.mesh_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.run_button.configure(state=tk.DISABLED if running or self.mesh_model is None else tk.NORMAL)
        self.pause_button.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.resume_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _set_entries_state(self, state):
        for entry in self.entries.values():
            entry.configure(state=state)
        self.diagonal_selector.configure(state=tk.DISABLED if state == tk.DISABLED else "readonly")

    def show_mesh_popup(self):
        if self.mesh_model is None:
            return
        popup = tk.Toplevel(self.root)
        popup.title("Generated finite-element mesh")
        popup.geometry("1000x720")
        popup.minsize(760, 540)
        popup.rowconfigure(0, weight=1)
        popup.columnconfigure(0, weight=1)

        figure = Figure(figsize=(10, 6.5), dpi=100)
        axis = figure.add_subplot(111)
        nodes = self.mesh_model["nodes"]
        axis.triplot(nodes[:, 0] * 1000, nodes[:, 1] * 1000, self.mesh_model["elements"], color="#1f4e79", linewidth=0.25)
        axis.set_aspect("equal")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_title("Generated conforming triangular mesh")
        figure.tight_layout()
        canvas = FigureCanvasTkAgg(figure, master=popup)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.draw_idle()

        footer = ttk.Frame(popup, padding=(10, 5, 10, 10))
        footer.grid(row=1, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(
            footer,
            text=f"{len(nodes):,} nodes | {len(self.mesh_model['elements']):,} triangular elements",
        ).grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Close", command=popup.destroy).grid(row=0, column=1, sticky="e")

    def _show_empty_result(self, text):
        self.result_widget.grid_remove()
        self.table_frame.grid_remove()
        self.empty_result_label.configure(text=text)
        self.empty_result_label.grid(row=1, column=0, sticky="nsew")

    def show_selected_result(self, _event=None):
        if self.result is None:
            return
        selected = self.RESULT_OPTIONS.get(self.result_choice.get())
        if not selected:
            self._show_empty_result("Choose one result from the menu above.")
            return
        self.empty_result_label.grid_remove()
        if selected == "quantitative":
            self.result_widget.grid_remove()
            self.table_frame.grid(row=1, column=0, sticky="nsew")
            self._populate_quantitative_table()
        else:
            self.table_frame.grid_remove()
            self.result_widget.grid(row=1, column=0, sticky="nsew")
            self.render_result_figure(self.result_figure, selected)
            self.result_canvas.draw_idle()

    def _triangulation(self):
        model = self.result["model"]
        return mtri.Triangulation(model["nodes"][:, 0] * 1000, model["nodes"][:, 1] * 1000, model["elements"])

    def render_result_figure(self, figure, result_type):
        figure.clear()
        model = self.result["model"]
        nodes = model["nodes"]
        if result_type == "temperature":
            axis = figure.add_subplot(111)
            field = axis.tripcolor(self._triangulation(), self.result["temperatures"], shading="gouraud", cmap="inferno")
            axis.set_title("Temperature field")
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            axis.set_aspect("equal")
            figure.colorbar(field, ax=axis, label="Temperature (°C)")
        elif result_type == "heat_flux":
            axis = figure.add_subplot(111)
            field = axis.tripcolor(self._triangulation(), facecolors=self.result["element_flux"][:, 4], shading="flat", cmap="viridis")
            axis.set_title("Element heat-flux magnitude")
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            axis.set_aspect("equal")
            figure.colorbar(field, ax=axis, label="Heat flux magnitude (W/m²)")
        elif result_type == "centerline":
            axis = figure.add_subplot(111)
            mask = np.isclose(nodes[:, 0], 0.0, atol=1e-12)
            selected_nodes = nodes[mask]
            temperatures = self.result["temperatures"][mask]
            order = np.argsort(selected_nodes[:, 1])
            axis.plot(selected_nodes[order, 1] * 1000, temperatures[order], color="#c00000", lw=2)
            axis.set_title("Vertical centerline temperature")
            axis.set_xlabel("y (mm)")
            axis.set_ylabel("Temperature (°C)")
            axis.grid(alpha=0.25)
        elif result_type == "base_top":
            axis = figure.add_subplot(111)
            mask = np.isclose(nodes[:, 1], model["y_base_top"], atol=1e-12)
            selected_nodes = nodes[mask]
            temperatures = self.result["temperatures"][mask]
            order = np.argsort(selected_nodes[:, 0])
            axis.plot(selected_nodes[order, 0] * 1000, temperatures[order], color="#1f4e79", lw=2)
            axis.set_title("Temperature along heat-sink base top")
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("Temperature (°C)")
            axis.grid(alpha=0.25)
        else:
            raise ValueError(f"Unknown result type: {result_type}")
        figure.tight_layout()

    def quantitative_rows(self):
        result = self.result
        parameters = result["parameters"]
        model = result["model"]
        convection = result["convection"]
        violations = result.get("temperature_limit_violations", [])
        rows = [
            ("THERMAL PERFORMANCE", "", ""),
            ("Maximum junction temperature, T_j,max", result["junction_temperature"], "°C"),
            ("Mean silicon temperature", result["mean_die_temperature"], "°C"),
            ("Minimum model temperature", float(np.min(result["temperatures"])), "°C"),
            ("Junction-to-ambient resistance, R_ja", result["R_ja"], "K/W"),
            ("Convective heat removal", result["total_removed_heat"], "W"),
            ("Thermal target", "PASS" if result["target_pass"] else "FAIL", ""),
            ("FLUID / CONVECTION", "", ""),
            ("Convection coefficient, h", convection["h"], "W/(m²·K)"),
            ("Reynolds number, Re", convection["Re"], ""),
            ("Nusselt number, Nu", convection["Nu"], ""),
            ("Estimated pressure drop, Δp", convection["pressure_drop"], "Pa"),
            ("Fin-channel spacing, s", convection["fin_spacing"] * 1000, "mm"),
            ("Hydraulic diameter, D_h", convection["hydraulic_diameter"] * 1000, "mm"),
            ("MESH SUMMARY", "", ""),
            ("Number of nodes", len(model["nodes"]), ""),
            ("Number of triangular elements", len(model["elements"]), ""),
            ("Number of convective boundary edges", len(model["convective_edges"]), ""),
            ("Mesh status", "valid", ""),
            ("NUMERICAL VALIDATION", "", ""),
            ("Integrated FEM heat generation", result["total_generated_heat"], "W"),
            ("Convective heat removal", result["total_removed_heat"], "W"),
            ("Energy-balance error", result["energy_balance_error"], "fraction"),
            ("Normalized linear-system residual", result["residual"], "fraction"),
        ]
        for violation in violations:
            rows.append((f"Temperature-limit warning: {violation['name']}", f"{violation['temperature_C']:.2f} > {violation['limit_C']:.2f} °C", ""))
        return rows

    @staticmethod
    def _format_value(value):
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if isinstance(value, (float, np.floating)):
            if value == 0:
                return "0"
            if abs(value) < 1e-4 or abs(value) >= 1e5:
                return f"{value:.6e}"
            return f"{value:.6g}"
        return str(value)

    def _populate_quantitative_table(self):
        rows = []
        for quantity, value, unit in self.quantitative_rows():
            rows.append((quantity, "" if value == "" else self._format_value(value), scientific_display(unit)))
        self.quantitative_table.set_rows(rows)

    def save_selected(self):
        if self.result is None:
            messagebox.showinfo("No result", "Run a simulation first.")
            return
        selected = self.save_choice.get()
        if not selected:
            messagebox.showinfo("Choose an output", "Select one output from the save menu first.")
            return
        try:
            figure_types = {"Temperature contour": "temperature", "Heat-flux contour": "heat_flux", "Vertical centerline temperature": "centerline", "Heat-sink base-top temperature": "base_top"}
            extensions = {**{name: ".png" for name in figure_types}, "Quantitative analysis": ".csv", "Mesh NPZ": ".npz", "Mesh CSV": ".csv", "Mesh MSH": ".msh", "Mesh VTK": ".vtk", "Mesh XDMF": ".xdmf", "Nodal temperatures": ".csv", "Element heat flux": ".csv", "Simulation summary": ".json"}
            path = filedialog.asksaveasfilename(title=f"Save {selected}", defaultextension=extensions[selected], filetypes=[("All files", "*.*")], initialfile="")
            if not path:
                return
            path = Path(path)
            if selected in figure_types:
                fig = Figure(figsize=(10, 7), dpi=120); self.render_result_figure(fig, figure_types[selected]); fig.savefig(path, dpi=300, bbox_inches="tight")
            elif selected == "Quantitative analysis": self._write_quantitative_csv(path)
            elif selected == "Nodal temperatures": self._write_nodes_csv(path)
            elif selected == "Element heat flux": self._write_flux_csv(path)
            elif selected == "Simulation summary": path.write_text(json.dumps({k: v for k, v in self.result.items() if k not in {"model", "temperatures", "element_flux"}}, default=str, indent=2, ensure_ascii=False), encoding="utf-8")
            elif selected.startswith("Mesh "): self._write_single_mesh(path, selected[5:])
            self._saved_message(path)
        except Exception as error:
            messagebox.showerror("Save error", str(error))

    def _ask_save_path(self, title, extension, filetypes):
        selected = filedialog.asksaveasfilename(title=f"Save {title}", defaultextension=extension, filetypes=filetypes)
        return Path(selected) if selected else None

    def _save_figure(self, figure_type):
        path = self._ask_save_path("figure", ".png", [("PNG images", "*.png"), ("JPEG images", "*.jpg;*.jpeg"), ("PDF files", "*.pdf")])
        if not path:
            return
        figure = Figure(figsize=(10, 7), dpi=120)
        if figure_type == "schematic":
            self.draw_schematic(figure, ThermalParameters(**self.result["parameters"]))
        else:
            self.render_result_figure(figure, figure_type)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        self._saved_message(path)

    def _write_single_mesh(self, path, fmt):
        model = self.result["model"]
        path = Path(path)
        if fmt == "NPZ":
            np.savez_compressed(path, nodes=model["nodes"], elements=model["elements"], materials=np.asarray(model["element_materials"], dtype=str), units="m")
        elif fmt == "CSV":
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle); writer.writerow(["node_id", "x_m", "y_m"])
                for i, (x, y) in enumerate(model["nodes"]): writer.writerow([i, x, y])
                writer.writerow([]); writer.writerow(["element_id", "n1", "n2", "n3", "material_region"])
                for i, (element, material) in enumerate(zip(model["elements"], model["element_materials"])): writer.writerow([i, *element, material])
        elif fmt == "MSH":
            with path.open("w", encoding="utf-8") as handle:
                handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{}\n".format(len(model["nodes"])))
                for i, (x, y) in enumerate(model["nodes"], 1): handle.write(f"{i} {x} {y} 0\n")
                handle.write("$EndNodes\n$Elements\n{}\n".format(len(model["elements"])))
                for i, element in enumerate(model["elements"], 1): handle.write(f"{i} 2 0 {element[0]+1} {element[1]+1} {element[2]+1}\n")
                handle.write("$EndElements\n")
        elif fmt == "VTK":
            n, e = len(model["nodes"]), len(model["elements"])
            with path.open("w", encoding="utf-8") as handle:
                handle.write("# vtk DataFile Version 3.0\nProject 1 mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
                handle.write(f"POINTS {n} float\n" + "".join(f"{x} {y} 0\n" for x, y in model["nodes"]))
                handle.write(f"CELLS {e} {4*e}\n" + "".join(f"3 {a} {b} {c}\n" for a, b, c in model["elements"]))
                handle.write(f"CELL_TYPES {e}\n" + "".join("5\n" for _ in range(e)))
        elif fmt == "XDMF":
            n, e = len(model["nodes"]), len(model["elements"])
            topology = " ".join(" ".join(map(str, tri)) for tri in model["elements"])
            geometry = " ".join(f"{x} {y}" for x, y in model["nodes"])
            content = f'<?xml version="1.0"?><Xdmf Version="3.0"><Domain><Grid Name="thermal_mesh" GridType="Uniform"><Topology TopologyType="Triangle" NumberOfElements="{e}"><DataItem Dimensions="{e} 3" NumberType="Int" Format="XML">{topology}</DataItem></Topology><Geometry GeometryType="XY"><DataItem Dimensions="{n} 2" NumberType="Float" Format="XML">{geometry}</DataItem></Geometry></Grid></Domain></Xdmf>'
            path.write_text(content, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported mesh format: {fmt}")

    def _write_mesh_exports(self, output):
        model = self.result["model"]
        mesh_dir = Path(output) / "mesh"
        mesh_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(mesh_dir / "mesh_data.npz", nodes=model["nodes"], elements=model["elements"], materials=np.asarray(model["element_materials"], dtype=str), units="m")
        with (mesh_dir / "mesh_nodes_elements.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle); writer.writerow(["node_id","x_m","y_m"])
            for i,(x,y) in enumerate(model["nodes"]): writer.writerow([i,x,y])
            writer.writerow([]); writer.writerow(["element_id","n1","n2","n3","material_region"])
            for i,(element, material) in enumerate(zip(model["elements"], model["element_materials"])): writer.writerow([i,*element,material])
        with (mesh_dir / "mesh_boundaries.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle); writer.writerow(["boundary_id", "node_1", "node_2", "boundary_type", "units"])
            convective = {tuple(e) for e in model.get("convective_edges", [])}
            for i, record in enumerate(model.get("boundary_edges", [])):
                edge = record[0] if isinstance(record, tuple) and len(record) == 2 and isinstance(record[1], str) else record
                kind = "convective" if tuple(edge) in convective else "external_or_internal"
                writer.writerow([i, int(edge[0]), int(edge[1]), kind, "m"])
        with (mesh_dir / "mesh.msh").open("w", encoding="utf-8") as handle:
            handle.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n{}\n".format(len(model["nodes"])))
            for i,(x,y) in enumerate(model["nodes"],1): handle.write(f"{i} {x} {y} 0\n")
            handle.write("$EndNodes\n$Elements\n{}\n".format(len(model["elements"])))
            for i,element in enumerate(model["elements"],1): handle.write(f"{i} 2 0 {element[0]+1} {element[1]+1} {element[2]+1}\n")
            handle.write("$EndElements\n")
        n = len(model["nodes"]); e = len(model["elements"])
        with (mesh_dir / "mesh.vtk").open("w", encoding="utf-8") as handle:
            handle.write("# vtk DataFile Version 3.0\nProject 1 mesh\nASCII\nDATASET UNSTRUCTURED_GRID\n")
            handle.write(f"POINTS {n} float\n" + "".join(f"{x} {y} 0\n" for x,y in model["nodes"]))
            handle.write(f"CELLS {e} {4*e}\n" + "".join(f"3 {a} {b} {c}\n" for a,b,c in model["elements"]))
            handle.write(f"CELL_TYPES {e}\n" + "".join("5\n" for _ in range(e)))
        with (mesh_dir / "mesh.xdmf").open("w", encoding="utf-8") as handle:
            handle.write(f"<?xml version=\"1.0\"?><Xdmf Version=\"3.0\"><Domain><Grid Name=\"thermal_mesh\" GridType=\"Uniform\"><Topology TopologyType=\"Triangle\" NumberOfElements=\"{e}\"><DataItem Dimensions=\"{e} 3\" NumberType=\"Int\" Format=\"XML\">{' '.join(' '.join(map(str, tri)) for tri in model['elements'])}</DataItem></Topology><Geometry GeometryType=\"XY\"><DataItem Dimensions=\"{n} 2\" NumberType=\"Float\" Format=\"XML\">{' '.join(f'{x} {y}' for x,y in model['nodes'])}</DataItem></Geometry></Grid></Domain></Xdmf>")

    def _write_quantitative_csv(self, path):
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["quantity", "value", "unit"])
            writer.writeheader()
            for quantity, value, unit in self.quantitative_rows():
                writer.writerow({"quantity": quantity, "value": value, "unit": unit})

    def _write_nodes_csv(self, path):
        nodes = self.result["model"]["nodes"]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["node ID", "x_m", "y_m", "x_mm", "y_mm", "temperature_C"])
            for index, ((x_value, y_value), temperature) in enumerate(zip(nodes, self.result["temperatures"])):
                writer.writerow([index, x_value, y_value, x_value * 1000, y_value * 1000, temperature])

    def _write_flux_csv(self, path):
        flux = self.result["element_flux"]
        materials = self.result["model"]["element_materials"]
        with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["element ID", "centroid x", "centroid y", "q_x", "q_y", "heat_flux_magnitude", "material"])
            for index, (row, material) in enumerate(zip(flux, materials)):
                writer.writerow([index, row[0], row[1], row[2], row[3], row[4], material])

    def _saved_message(self, path):
        self.status.set(f"Saved: {path}")
        messagebox.showinfo("Saved", f"Saved to:\n{path}")

    def close(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.run_event is not None:
            self.run_event.set()
        if self._poll_after is not None:
            try:
                self.root.after_cancel(self._poll_after)
            except tk.TclError:
                pass
        for canvas in (getattr(self, "schematic_canvas", None), getattr(self, "result_canvas", None)):
            callback_id = getattr(canvas, "_idle_draw_id", None)
            if callback_id is not None:
                try:
                    self.root.after_cancel(callback_id)
                except tk.TclError:
                    pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


if __name__ == "__main__":
    app = tk.Tk()
    SimulationTool(app)
    app.mainloop()
