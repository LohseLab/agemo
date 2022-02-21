import numpy as np
import sage.all

import gf.gflib as gflib
import gf.inverse as gfinverse
import gf.legacy.inverse as linverse


def get_parameter_dict(coalescence_rates, global_info, sim_config, gf_vars):
    parameter_dict = {}
    reference_pop = global_info["reference_pop"]
    if gf_vars.get("migration_rate"):
        migration_string = (
            "me_A_B"
            if gf_vars["migration_direction"] == [(1, 2)]
            else "me_B_A"
        )
        parameter_dict[gf_vars["migration_rate"]] = sage.all.Rational(
            2
            * sim_config[migration_string]
            * sim_config[f"Ne_{reference_pop}"]
        )
    if gf_vars.get("exodus_rate"):
        parameter_dict[sage.all.SR.var("T")] = sage.all.Rational(
            sim_config["T"] / (2 * sim_config[f"Ne_{reference_pop}"])
        )
    for c, Ne in zip(coalescence_rates, ("Ne_A_B", "Ne_A", "Ne_B")):
        if Ne in sim_config:
            parameter_dict[c] = sage.all.Rational(
                sim_config[f"Ne_{reference_pop}"] / sim_config[Ne]
            )
        else:
            parameter_dict[c] = 0.0
    return parameter_dict


def get_theta(global_info, sim_config, **kwargs):
    reference_pop = global_info["reference_pop"]
    Ne_ref = sim_config[f"Ne_{reference_pop}"]
    mu = global_info["mu"]
    block_length = global_info["blocklength"]
    return 2 * sage.all.Rational(Ne_ref * mu) * block_length


def equations_from_matrix_with_inverse(
    multiplier_array, paths, var_array, time, delta_idx
):
    split_paths = gflib.split_paths_laplace(paths, multiplier_array, delta_idx)
    delta_in_nom_all = multiplier_array[:, 0, delta_idx] == 1
    results = np.zeros(len(split_paths), dtype=object)
    subset_no_delta = np.arange(multiplier_array.shape[-1]) != delta_idx
    multiplier_array_no_delta = multiplier_array[:, :, subset_no_delta]
    for idx, (no_delta, with_delta) in enumerate(split_paths):
        delta_in_nom_list = delta_in_nom_all[with_delta]
        inverse = gfinverse.inverse_laplace_single_event(
            multiplier_array_no_delta[with_delta],
            var_array,
            time,
            delta_in_nom_list,
        )
        if isinstance(inverse, np.ndarray):
            inverse = np.sum(inverse)
        no_inverse = np.prod(
            gflib.equations_from_matrix(
                multiplier_array_no_delta[no_delta], var_array
            )
        )
        results[idx] = np.prod((inverse, no_inverse))
    return results


def equations_with_sage(multiplier_array, paths, var_array, time, delta_idx):
    delta = var_array[delta_idx] if delta_idx is not None else None
    eqs = np.zeros(len(paths), dtype=object)
    for i, path in enumerate(paths):
        ma = multiplier_array[np.array(path, dtype=int)]
        temp = np.prod(gflib.equations_from_matrix(ma, var_array))
        eqs[i] = linverse.return_inverse_laplace(temp, delta).subs(
            {sage.all.SR.var("T"): time}
        )
    return eqs


def make_branchtype_dict_idxs_gimble():
    return {"a": 1, "abb": 1, "b": 0, "aab": 0, "ab": 2, "aa": 3, "bb": 3}
