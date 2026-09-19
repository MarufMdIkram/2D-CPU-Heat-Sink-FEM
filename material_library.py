"""Reference solid-material and working-fluid data used by the desktop model.

Values are representative near room temperature, not certified design data.
The solid model is isotropic; composite conductivity uses the arithmetic
volume-fraction rule (parallel/Voigt estimate).
"""

SOLID_MATERIALS = {
    "Aluminum 6061-T6": {"conductivity": 167.0, "density": 2700.0, "specific_heat": 896.0, "critical_temperature_C": 582.0, "limit_kind": "solidus/melting"},
    "Aluminum 6063-T5": {"conductivity": 201.0, "density": 2700.0, "specific_heat": 900.0, "critical_temperature_C": 615.0, "limit_kind": "solidus/melting"},
    "Aluminum 1050": {"conductivity": 222.0, "density": 2710.0, "specific_heat": 900.0, "critical_temperature_C": 646.0, "limit_kind": "solidus/melting"},
    "Copper C110": {"conductivity": 390.0, "density": 8960.0, "specific_heat": 385.0, "critical_temperature_C": 1083.0, "limit_kind": "melting"},
    "Brass C260": {"conductivity": 120.0, "density": 8530.0, "specific_heat": 380.0, "critical_temperature_C": 915.0, "limit_kind": "solidus/melting"},
    "Carbon steel": {"conductivity": 54.0, "density": 7850.0, "specific_heat": 486.0, "critical_temperature_C": 1425.0, "limit_kind": "solidus/melting"},
    "Stainless steel 304": {"conductivity": 16.2, "density": 8000.0, "specific_heat": 500.0, "critical_temperature_C": 1400.0, "limit_kind": "solidus/melting"},
    "Stainless steel 316": {"conductivity": 16.3, "density": 8000.0, "specific_heat": 500.0, "critical_temperature_C": 1375.0, "limit_kind": "solidus/melting"},
    "Titanium Grade 2": {"conductivity": 21.9, "density": 4510.0, "specific_heat": 523.0, "critical_temperature_C": 1668.0, "limit_kind": "melting"},
    "Nickel 200": {"conductivity": 70.0, "density": 8890.0, "specific_heat": 456.0, "critical_temperature_C": 1455.0, "limit_kind": "melting"},
    "Aluminum 3003-H14": {"conductivity": 159.0, "density": 2730.0, "specific_heat": 893.0, "critical_temperature_C": 643.0, "limit_kind": "solidus/melting"},
    "Aluminum 5052-H32": {"conductivity": 138.0, "density": 2680.0, "specific_heat": 880.0, "critical_temperature_C": 607.2, "limit_kind": "solidus/melting"},
    "Aluminum 7075-T6": {"conductivity": 130.0, "density": 2810.0, "specific_heat": 960.0, "critical_temperature_C": 477.0, "limit_kind": "solidus/melting"},
    "Copper C10100 OFE": {"conductivity": 391.0, "density": 8940.0, "specific_heat": 385.0, "critical_temperature_C": 1083.0, "limit_kind": "melting"},
    "Magnesium AZ31B-H24": {"conductivity": 96.0, "density": 1770.0, "specific_heat": 1000.0, "critical_temperature_C": 605.0, "limit_kind": "solidus/melting"},
    "Magnesium AZ91D-F": {"conductivity": 72.7, "density": 1810.0, "specific_heat": 1047.0, "critical_temperature_C": 421.0, "limit_kind": "solidus/melting"},
    "Titanium Ti-6Al-4V Grade 5": {"conductivity": 6.70, "density": 4430.0, "specific_heat": 526.3, "critical_temperature_C": 1604.0, "limit_kind": "melting"},
    "Silver 99.95%": {"conductivity": 425.0, "density": 10490.0, "specific_heat": 234.0, "critical_temperature_C": 961.8, "limit_kind": "melting"},
}

FLUIDS = {
    "Air": {"density": 1.092, "viscosity": 1.96e-5, "conductivity": 0.028, "specific_heat": 1007.0, "prandtl": 0.707, "boiling_temperature_C": None, "limit_kind": "no boiling check for a gas", "reference_temperature_C": 35.0},
    "Water": {"density": 997.0, "viscosity": 8.90e-4, "conductivity": 0.607, "specific_heat": 4182.0, "boiling_temperature_C": 100.0, "limit_kind": "boiling at 1 atm", "reference_temperature_C": 25.0},
    "Glycerin (99%)": {"density": 1260.0, "viscosity": 0.95, "conductivity": 0.285, "specific_heat": 2410.0, "boiling_temperature_C": 290.0, "limit_kind": "decomposition / boiling limit", "reference_temperature_C": 25.0},
    "Ethylene glycol": {"density": 1113.0, "viscosity": 1.61e-2, "conductivity": 0.252, "specific_heat": 2415.0, "boiling_temperature_C": 197.0, "limit_kind": "boiling at 1 atm", "reference_temperature_C": 25.0},
    "Propylene glycol": {"density": 1036.0, "viscosity": 4.20e-2, "conductivity": 0.200, "specific_heat": 2480.0, "boiling_temperature_C": 188.0, "limit_kind": "boiling at 1 atm", "reference_temperature_C": 25.0},
    "Mineral oil": {"density": 850.0, "viscosity": 3.00e-2, "conductivity": 0.130, "specific_heat": 1900.0, "boiling_temperature_C": 300.0, "limit_kind": "approx. thermal decomposition", "reference_temperature_C": 25.0},
    "Silicone oil": {"density": 950.0, "viscosity": 5.00e-2, "conductivity": 0.150, "specific_heat": 1500.0, "boiling_temperature_C": 300.0, "limit_kind": "approx. thermal decomposition", "reference_temperature_C": 25.0},
}


def library_material(name):
    if name not in SOLID_MATERIALS:
        raise ValueError(f"Unknown solid material: {name}")
    return {"name": name, **SOLID_MATERIALS[name]}


def composite_material(components):
    """Mix component records using volume fraction and the Voigt rule for k."""
    if len(components) < 2:
        raise ValueError("A composite requires at least two components")
    fractions = [float(component["volume_fraction"]) for component in components]
    if any(fraction <= 0 for fraction in fractions) or abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("Composite volume fractions must be positive and sum to 1.0")
    conductivity = sum(f * float(item["conductivity"]) for f, item in zip(fractions, components))
    density = sum(f * float(item["density"]) for f, item in zip(fractions, components))
    volumetric_heat_capacity = sum(f * float(item["density"]) * float(item["specific_heat"]) for f, item in zip(fractions, components))
    specific_heat = volumetric_heat_capacity / density
    critical = min(float(item["critical_temperature_C"]) for item in components)
    description = " + ".join(f"{item['name']} {fraction:.1%}" for item, fraction in zip(components, fractions))
    return {
        "name": f"Composite ({description})",
        "conductivity": conductivity,
        "density": density,
        "specific_heat": specific_heat,
        "critical_temperature_C": critical,
        "limit_kind": "lowest constituent limit",
        "components": [dict(item, volume_fraction=fraction) for item, fraction in zip(components, fractions)],
        "mixing_rule": "volume-weighted conductivity; density-weighted specific heat",
    }


def validate_solid_material(material):
    required = ("name", "conductivity", "density", "specific_heat", "critical_temperature_C")
    if not isinstance(material, dict) or any(key not in material for key in required):
        raise ValueError("A solid material requires a name, conductivity, density, specific heat, and temperature limit")
    for key in required[1:]:
        value = float(material[key])
        if not np_is_finite_positive(value):
            raise ValueError(f"Material property {key} must be finite and positive")
    return dict(material)


def np_is_finite_positive(value):
    try:
        return value > 0 and value < float("inf")
    except (TypeError, ValueError):
        return False
