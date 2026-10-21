"""A cell_average block for numerics.yml, for tests: every key at its production value unless overridden.

Tests of the configuration need many near-identical blocks that differ in one key. Writing each in full
repeats every key in every test, so adding a key meant editing all of them. Here a test names only what
it changes: ``cell_average_yaml(route="quadrature")``, or ``cell_average_yaml(far_tol=None)`` to leave a
key out.
"""

#: the production values (cubic_scattering/numerics.yml), as YAML text
PRODUCTION = {
    "backend": "fortran",
    "route": "kseries",
    "kseries_tol": "1.0e-15",
    "kseries_max_reach": "3.0",
    "kseries_store": "store.npy",
    "far_expansion": "true",
    "far_tol": "1.0e-15",
    "far_max_order": "24",
    "fallback_n_gauss": "24",
}


def cell_average_yaml(**values: object) -> str:
    """The block with these keys overridden (a value of None leaves the key out)."""
    merged = {**PRODUCTION, **{k: v for k, v in values.items()}}
    unknown = set(values) - set(PRODUCTION)
    if unknown:
        raise KeyError(f"cell_average_yaml: no key {sorted(unknown)}; the keys are {sorted(PRODUCTION)}")
    lines = [f"  {k}: {v}" for k, v in merged.items() if v is not None]
    return "cell_average:\n" + "\n".join(lines) + "\n"
