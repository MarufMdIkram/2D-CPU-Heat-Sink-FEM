from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
import csv
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from material_library import FLUIDS, library_material, validate_solid_material


class SimulationCancelled(RuntimeError):
    """Raised when a running simulation is stopped by the caller."""


def _checkpoint(run_event=None, stop_event=None):
    """Cooperatively pause or cancel work without coupling the model to Tkinter."""
    if stop_event is not None and stop_event.is_set():
        raise SimulationCancelled("Simulation stopped by the user")
    if run_event is not None:
        while not run_event.wait(0.1):
            if stop_event is not None and stop_event.is_set():
                raise SimulationCancelled("Simulation stopped by the user")


def _report(progress_callback, percent, message):
    if progress_callback is not None:
        progress_callback(float(percent), message)


@dataclass
class ThermalParameters:
    heat_load: float = 150.0
    ambient_temperature: float = 35.0
    maximum_junction_temperature: float = 95.0
    number_of_fins: int = 24
    fin_thickness: float = 0.001
    fin_height: float = 0.035
    base_width: float = 0.092
    base_thickness: float = 0.005
    air_velocity: float = 4.0
    mesh_size: float = 0.0005
    mesh_size_x: float | None = None
    mesh_size_y: float | None = None
    minimum_layer_elements: int = 1
    diagonal_pattern: str = "alternating"
    fin_material: dict = field(default_factory=lambda: library_material("Aluminum 6063-T5"))
    base_material: dict = field(default_factory=lambda: library_material("Aluminum 6063-T5"))
    fluid_name: str = "Air"
    fluid: dict = field(default_factory=lambda: {"name": "Air", **FLUIDS["Air"]})


def convection_properties(parameters):
    if parameters.number_of_fins < 2:
        raise ValueError("At least two fins are required")
    if parameters.air_velocity <= 0:
        raise ValueError("Fluid velocity must be positive")
    spacing = (parameters.base_width - parameters.number_of_fins * parameters.fin_thickness) / (parameters.number_of_fins - 1)
    if spacing <= 0 or parameters.fin_height <= 0:
        raise ValueError("Fin geometry must leave a positive channel spacing and height")
    hydraulic_diameter = 2 * spacing * parameters.fin_height / (spacing + parameters.fin_height)
    fluid = parameters.fluid or {"name": parameters.fluid_name, **FLUIDS[parameters.fluid_name]}
    rho = float(fluid["density"])
    mu = float(fluid["viscosity"])
    k_air = float(fluid["conductivity"])
    pr = float(fluid.get("prandtl", mu * float(fluid["specific_heat"]) / k_air))
    reynolds = rho * parameters.air_velocity * hydraulic_diameter / mu
    alpha = min(spacing, parameters.fin_height) / max(spacing, parameters.fin_height)
    nu_laminar = 7.541 * (1 - 2.610 * alpha + 4.970 * alpha**2 - 5.119 * alpha**3 + 2.702 * alpha**4 - 0.548 * alpha**5)
    f_re_laminar = 96 * (1 - 1.3553 * alpha + 1.9467 * alpha**2 - 1.7012 * alpha**3 + 0.9564 * alpha**4 - 0.2537 * alpha**5)
    f_laminar = f_re_laminar / reynolds
    f_turbulent = (0.79 * np.log(max(reynolds, 1.0001)) - 1.64) ** -2
    nu_turbulent = ((f_turbulent / 8) * (reynolds - 1000) * pr) / (1 + 12.7 * np.sqrt(f_turbulent / 8) * (pr ** (2 / 3) - 1))
    if reynolds <= 2300:
        nusselt, friction = nu_laminar, f_laminar
    elif reynolds >= 3000:
        nusselt, friction = nu_turbulent, f_turbulent
    else:
        blend = (reynolds - 2300) / 700
        nusselt = nu_laminar + blend * (nu_turbulent - nu_laminar)
        friction = f_laminar + blend * (f_turbulent - f_laminar)
    h = nusselt * k_air / hydraulic_diameter
    pressure_drop = friction * (parameters.base_width / hydraulic_diameter) * rho * parameters.air_velocity**2 / 2
    return {"fluid_name": fluid.get("name", parameters.fluid_name), "fluid_density": rho, "fluid_viscosity": mu, "fluid_conductivity": k_air, "fluid_specific_heat": float(fluid["specific_heat"]), "Pr": pr, "fin_spacing": spacing, "hydraulic_diameter": hydraulic_diameter, "Re": reynolds, "Nu": nusselt, "h": h, "pressure_drop": pressure_drop}


def create_model(parameters, progress_callback=None, run_event=None, stop_event=None):
    _report(progress_callback, 0, "Validating inputs")
    _checkpoint(run_event, stop_event)
    if parameters.number_of_fins < 2 or parameters.base_width <= parameters.number_of_fins * parameters.fin_thickness:
        raise ValueError("Fin count and thickness leave no positive channel spacing")
    mesh_size_x = parameters.mesh_size_x if parameters.mesh_size_x is not None else parameters.mesh_size
    mesh_size_y = parameters.mesh_size_y if parameters.mesh_size_y is not None else parameters.mesh_size
    positive_values = {
        "fin thickness": parameters.fin_thickness,
        "fin height": parameters.fin_height,
        "base width": parameters.base_width,
        "base thickness": parameters.base_thickness,
        "horizontal mesh size": mesh_size_x,
        "vertical mesh size": mesh_size_y,
    }
    if any(not np.isfinite(value) or value <= 0 for value in positive_values.values()):
        raise ValueError("Fin, base, and mesh dimensions must be finite and positive")
    if not np.isfinite(parameters.heat_load) or parameters.heat_load <= 0:
        raise ValueError("Heat load must be finite and positive")
    if not np.isfinite(parameters.ambient_temperature):
        raise ValueError("Ambient temperature must be finite")
    if not np.isfinite(parameters.air_velocity) or parameters.air_velocity <= 0:
        raise ValueError("Fluid velocity must be finite and positive")
    if not isinstance(parameters.minimum_layer_elements, (int, np.integer)) or parameters.minimum_layer_elements < 1:
        raise ValueError("Minimum layer elements must be a positive whole number")
    if parameters.diagonal_pattern not in {"alternating", "rising", "falling"}:
        raise ValueError("Triangle diagonal pattern must be alternating, rising, or falling")
    fin_material = validate_solid_material(parameters.fin_material)
    base_material = validate_solid_material(parameters.base_material)
    fluid = parameters.fluid or {"name": parameters.fluid_name, **FLUIDS.get(parameters.fluid_name, {})}
    if not fluid or any(key not in fluid for key in ("density", "viscosity", "conductivity", "specific_heat")):
        raise ValueError("Choose a valid surrounding-fluid record before meshing")
    if any(not np.isfinite(float(fluid[key])) or float(fluid[key]) <= 0 for key in ("density", "viscosity", "conductivity", "specific_heat")):
        raise ValueError("Surrounding-fluid properties must be finite and positive")
    die_width, die_thickness, die_depth = 0.030, 0.00050, 0.030
    tim1_thickness, tim1_conductivity = 0.00005, 8.0
    ihs_width, ihs_thickness, ihs_depth, ihs_conductivity = 0.040, 0.0020, 0.040, 400.0
    tim2_thickness, tim2_conductivity = 0.00010, 5.0
    k_silicon, k_aluminum = 130.0, 201.0
    y0 = 0.0
    y1 = y0 + die_thickness
    y2 = y1 + tim1_thickness
    y3 = y2 + ihs_thickness
    y4 = y3 + tim2_thickness
    y5 = y4 + parameters.base_thickness
    y6 = y5 + parameters.fin_height
    spacing = (parameters.base_width - parameters.number_of_fins * parameters.fin_thickness) / (parameters.number_of_fins - 1)
    fins = [(-parameters.base_width / 2 + i * (parameters.fin_thickness + spacing), -parameters.base_width / 2 + i * (parameters.fin_thickness + spacing) + parameters.fin_thickness) for i in range(parameters.number_of_fins)]
    layers = [("silicon", (-die_width / 2, die_width / 2), (y0, y1)), ("TIM1", (-die_width / 2, die_width / 2), (y1, y2)), ("copper", (-ihs_width / 2, ihs_width / 2), (y2, y3)), ("TIM2", (-ihs_width / 2, ihs_width / 2), (y3, y4)), ("base", (-parameters.base_width / 2, parameters.base_width / 2), (y4, y5))]
    critical_x = [-parameters.base_width / 2, -die_width / 2, -ihs_width / 2, 0.0, ihs_width / 2, die_width / 2, parameters.base_width / 2] + [v for fin in fins for v in fin]
    critical_y = [y0, y1, y2, y3, y4, y5, y6]

    def material_at(x, y):
        for name, x_range, y_range in layers:
            if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
                return name
        if y5 <= y <= y6 and any(a <= x <= b for a, b in fins):
            return "fin"
        return None

    x_critical = sorted(set(round(value, 14) for value in critical_x))
    y_critical = sorted(set(round(value, 14) for value in critical_y))
    x_points = np.unique(np.concatenate([np.linspace(a, b, max(2, int(np.ceil((b - a) / mesh_size_x)) + 1)) for a, b in zip(x_critical[:-1], x_critical[1:])]))
    y_intervals = []
    for a, b in zip(y_critical[:-1], y_critical[1:]):
        minimum_count = parameters.minimum_layer_elements if b <= y5 else 1
        interval_count = max(minimum_count, int(np.ceil((b - a) / mesh_size_y)))
        y_intervals.append(np.linspace(a, b, interval_count + 1))
    y_points = np.unique(np.concatenate(y_intervals))
    node_map, nodes, elements, element_materials = {}, [], [], []
    _report(progress_callback, 5, "Building conforming mesh")

    def get_node(x, y):
        key = (round(float(x), 14), round(float(y), 14))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append(key)
        return node_map[key]

    total_cells = (len(y_points) - 1) * (len(x_points) - 1)
    report_stride = max(1, total_cells // 150)
    completed_cells = 0
    for iy in range(len(y_points) - 1):
        for ix in range(len(x_points) - 1):
            if completed_cells % report_stride == 0:
                _checkpoint(run_event, stop_event)
                _report(progress_callback, 5 + 20 * completed_cells / max(total_cells, 1), "Building conforming mesh")
            x_left, x_right = x_points[ix:ix + 2]
            y_bottom, y_top = y_points[iy:iy + 2]
            material = material_at((x_left + x_right) / 2, (y_bottom + y_top) / 2)
            completed_cells += 1
            if material is None:
                continue
            corners = [get_node(x_left, y_bottom), get_node(x_right, y_bottom), get_node(x_right, y_top), get_node(x_left, y_top)]
            falling = [(corners[0], corners[1], corners[3]), (corners[1], corners[2], corners[3])]
            rising = [(corners[0], corners[1], corners[2]), (corners[0], corners[2], corners[3])]
            if parameters.diagonal_pattern == "falling":
                triangles = falling
            elif parameters.diagonal_pattern == "rising":
                triangles = rising
            else:
                triangles = falling if (ix + iy) % 2 == 0 else rising
            elements.extend(triangles)
            element_materials.extend([material, material])
    nodes = np.asarray(nodes, dtype=float)
    elements = np.asarray(elements, dtype=int)
    edge_map = {}
    _report(progress_callback, 25, "Finding exposed boundaries")
    edge_stride = max(1, len(elements) // 100)
    for index, element in enumerate(elements):
        if index % edge_stride == 0:
            _checkpoint(run_event, stop_event)
            _report(progress_callback, 25 + 5 * index / max(len(elements), 1), "Finding exposed boundaries")
        for pair in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(element[pair[0]]), int(element[pair[1]]))))
            edge_map.setdefault(edge, []).append(index)
    boundary_edges = [(edge, element_materials[indices[0]]) for edge, indices in edge_map.items() if len(indices) == 1]
    convective_edges = [edge for edge, material in boundary_edges if material in {"base", "fin"} and not all(np.isclose(nodes[index, 1], y4) for index in edge)]
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 30, "Mesh complete")
    material_properties = {
        "silicon": (k_silicon, die_depth),
        "TIM1": (tim1_conductivity, die_depth),
        "copper": (ihs_conductivity, ihs_depth),
        "TIM2": (tim2_conductivity, ihs_depth),
        "base": (float(base_material["conductivity"]), parameters.base_width),
        "fin": (float(fin_material["conductivity"]), parameters.base_width),
    }
    return {"nodes": nodes, "elements": elements, "element_materials": element_materials, "boundary_edges": boundary_edges, "convective_edges": convective_edges, "fin_edges": fins, "interfaces": [y0, y1, y2, y3, y4, y5, y6], "material_properties": material_properties, "solid_materials": {"base": base_material, "fin": fin_material}, "fluid": fluid, "y_base_top": y5}


def solve_model(parameters, model=None, progress_callback=None, run_event=None, stop_event=None):
    model = model or create_model(parameters, progress_callback, run_event, stop_event)
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 30, "Assembling conduction equations")
    nodes, elements = model["nodes"], model["elements"]
    properties = model["material_properties"]
    source = parameters.heat_load / (0.030 * 0.030 * 0.00050)
    rows, cols, values, force, gradients, generated = [], [], [], np.zeros(len(nodes)), [], 0.0
    assembly_stride = max(1, len(elements) // 150)
    for element_index, (element, material) in enumerate(zip(elements, model["element_materials"])):
        if element_index % assembly_stride == 0:
            _checkpoint(run_event, stop_event)
            _report(progress_callback, 30 + 35 * element_index / max(len(elements), 1), "Assembling conduction equations")
        coordinates = nodes[element]
        twice_area = ((coordinates[1, 0] - coordinates[0, 0]) * (coordinates[2, 1] - coordinates[0, 1]) - (coordinates[2, 0] - coordinates[0, 0]) * (coordinates[1, 1] - coordinates[0, 1]))
        area = abs(twice_area) / 2
        if area <= 0:
            raise ValueError("Mesh contains a non-positive-area triangle")
        b_matrix = np.array([[coordinates[1, 1] - coordinates[2, 1], coordinates[2, 1] - coordinates[0, 1], coordinates[0, 1] - coordinates[1, 1]], [coordinates[2, 0] - coordinates[1, 0], coordinates[0, 0] - coordinates[2, 0], coordinates[1, 0] - coordinates[0, 0]]]) / twice_area
        conductivity, depth = properties[material]
        element_matrix = conductivity * depth * area * (b_matrix.T @ b_matrix)
        element_force = source * depth * area / 3 * np.ones(3) if material == "silicon" else np.zeros(3)
        generated += element_force.sum()
        force[element] += element_force
        gradients.append(b_matrix)
        for local_row in range(3):
            for local_col in range(3):
                rows.append(element[local_row]); cols.append(element[local_col]); values.append(element_matrix[local_row, local_col])
    convection = convection_properties(parameters)
    _report(progress_callback, 65, "Applying convection boundaries")
    convection_stride = max(1, len(model["convective_edges"]) // 100)
    for edge_index, edge in enumerate(model["convective_edges"]):
        if edge_index % convection_stride == 0:
            _checkpoint(run_event, stop_event)
            _report(progress_callback, 65 + 7 * edge_index / max(len(model["convective_edges"]), 1), "Applying convection boundaries")
        length = np.linalg.norm(nodes[edge[0]] - nodes[edge[1]])
        matrix = convection["h"] * parameters.base_width * length / 6 * np.array([[2, 1], [1, 2]])
        edge_force = convection["h"] * parameters.base_width * length * parameters.ambient_temperature / 2 * np.ones(2)
        force[list(edge)] += edge_force
        for local_row in range(2):
            for local_col in range(2):
                rows.append(edge[local_row]); cols.append(edge[local_col]); values.append(matrix[local_row, local_col])
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 74, "Finalizing sparse system")
    stiffness = coo_matrix((values, (rows, cols)), shape=(len(nodes), len(nodes))).tocsr()
    difference = stiffness - stiffness.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-10:
        raise ValueError("Global matrix is not symmetric")
    if not np.isfinite(stiffness.data).all() or not np.isfinite(force).all():
        raise ValueError("Linear system contains NaN or infinite values")
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 78, "Solving temperature field")
    temperatures = spsolve(stiffness, force)
    if not np.isfinite(temperatures).all():
        raise ValueError("Temperature solution contains NaN or infinite values")
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 87, "Calculating heat flux")
    fluxes, silicon_nodes = [], []
    flux_stride = max(1, len(elements) // 150)
    for element_index, (element, material, b_matrix) in enumerate(zip(elements, model["element_materials"], gradients)):
        if element_index % flux_stride == 0:
            _checkpoint(run_event, stop_event)
            _report(progress_callback, 87 + 9 * element_index / max(len(elements), 1), "Calculating heat flux")
        flux = -properties[material][0] * (b_matrix @ temperatures[element])
        center = nodes[element].mean(axis=0)
        fluxes.append((center[0], center[1], flux[0], flux[1], np.linalg.norm(flux)))
        if material == "silicon":
            silicon_nodes.extend(element.tolist())
    removed = 0.0
    _report(progress_callback, 96, "Checking energy balance")
    removal_stride = max(1, len(model["convective_edges"]) // 100)
    for edge_index, edge in enumerate(model["convective_edges"]):
        if edge_index % removal_stride == 0:
            _checkpoint(run_event, stop_event)
            _report(progress_callback, 96 + 3 * edge_index / max(len(model["convective_edges"]), 1), "Checking energy balance")
        length = np.linalg.norm(nodes[edge[0]] - nodes[edge[1]])
        removed += convection["h"] * parameters.base_width * length * (temperatures[list(edge)].mean() - parameters.ambient_temperature)
    silicon_nodes = np.unique(silicon_nodes)
    residual = np.linalg.norm(stiffness @ temperatures - force) / max(np.linalg.norm(force), 1e-30)
    _checkpoint(run_event, stop_event)
    _report(progress_callback, 100, "Simulation complete")
    return {"parameters": asdict(parameters), "model": model, "temperatures": temperatures, "element_flux": np.asarray(fluxes), "junction_temperature": float(temperatures[silicon_nodes].max()), "mean_die_temperature": float(temperatures[silicon_nodes].mean()), "R_ja": float((temperatures[silicon_nodes].max() - parameters.ambient_temperature) / parameters.heat_load), "total_generated_heat": generated, "total_removed_heat": removed, "energy_balance_error": abs(removed - generated) / parameters.heat_load, "residual": residual, "convection": convection, "target_pass": float(temperatures[silicon_nodes].max()) <= parameters.maximum_junction_temperature}


def temperature_limit_violations(result):
    """Find solid-material limits and fluid boiling limits exceeded by the solution."""
    model = result["model"]
    nodes = model["nodes"]
    temperatures = result["temperatures"]
    materials = np.asarray(model["element_materials"])
    violations = []
    for region, record in model["solid_materials"].items():
        threshold = record.get("critical_temperature_C")
        if threshold is None:
            continue
        region_elements = model["elements"][materials == region]
        if len(region_elements) == 0:
            continue
        region_nodes = np.unique(region_elements)
        local_index = int(np.argmax(temperatures[region_nodes]))
        node_index = int(region_nodes[local_index])
        local_temperature = float(temperatures[node_index])
        if local_temperature > float(threshold):
            violations.append({
                "kind": "solid material",
                "name": record["name"],
                "region": region,
                "temperature_C": local_temperature,
                "limit_C": float(threshold),
                "limit_kind": record.get("limit_kind", "material temperature limit"),
                "location_mm": (float(nodes[node_index, 0] * 1000), float(nodes[node_index, 1] * 1000)),
            })

    fluid = model.get("fluid", {})
    fluid_limit = fluid.get("boiling_temperature_C")
    if fluid_limit is not None and model["convective_edges"]:
        wall_nodes = np.unique(np.asarray(model["convective_edges"], dtype=int))
        local_index = int(np.argmax(temperatures[wall_nodes]))
        node_index = int(wall_nodes[local_index])
        local_temperature = float(temperatures[node_index])
        if local_temperature > float(fluid_limit):
            violations.append({
                "kind": "surrounding fluid",
                "name": fluid.get("name", "selected fluid"),
                "region": "convective wall",
                "temperature_C": local_temperature,
                "limit_C": float(fluid_limit),
                "limit_kind": fluid.get("limit_kind", "boiling temperature"),
                "location_mm": (float(nodes[node_index, 0] * 1000), float(nodes[node_index, 1] * 1000)),
            })
    return violations


def save_results(result, output_directory):
    output = Path(output_directory)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    model = result["model"]
    nodes = model["nodes"]
    node_rows = [{"node ID": index, "x_m": nodes[index, 0], "y_m": nodes[index, 1], "x_mm": nodes[index, 0] * 1000, "y_mm": nodes[index, 1] * 1000, "temperature_C": result["temperatures"][index]} for index in range(len(nodes))]
    write_csv(output / "nodal_temperatures.csv", node_rows)
    flux = result["element_flux"]
    flux_rows = [{"element ID": index, "centroid x": flux[index, 0], "centroid y": flux[index, 1], "q_x": flux[index, 2], "q_y": flux[index, 3], "heat_flux_magnitude": flux[index, 4], "material": model["element_materials"][index]} for index in range(len(flux))]
    write_csv(output / "element_heat_flux.csv", flux_rows)
    summary = {"heat_load_W": result["parameters"]["heat_load"], "ambient_temperature_C": result["parameters"]["ambient_temperature"], "T_j_max_C": result["junction_temperature"], "R_ja_K_per_W": result["R_ja"], "energy_balance_error": result["energy_balance_error"], "residual": result["residual"], "thermal_target": "PASS" if result["target_pass"] else "FAIL", "fin_material": result["parameters"]["fin_material"]["name"], "base_material": result["parameters"]["base_material"]["name"], "surrounding_fluid": result["parameters"]["fluid_name"], "temperature_limit_warning": "YES" if result.get("temperature_limit_violations") else "NO", "forced_continuation": bool(result.get("continued_past_limit", False)), "temperature_limit_violations": json.dumps(result.get("temperature_limit_violations", [])), **{key: value for key, value in result["convection"].items()}}
    write_csv(output / "summary_results.csv", [summary])
    (output / "parameters.json").write_text(json.dumps(result["parameters"], indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(output / "mesh_data.npz", nodes=model["nodes"], elements=model["elements"], materials=np.asarray(model["element_materials"], dtype=str))
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
