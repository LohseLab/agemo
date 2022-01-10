import sage.all

import gf.gflib as gflib

def get_parameter_dict(coalescence_rates, global_info, sim_config, gf_vars):
	parameter_dict = {}
	reference_pop = global_info['reference_pop']
	if gf_vars.get('migration_rate'):
		migration_string = 'me_A_B' if gf_vars['migration_direction'] == [(1,2)] else 'me_B_A'
		parameter_dict[gf_vars['migration_rate']] = sage.all.Rational(2 * sim_config[migration_string] * sim_config[f'Ne_{reference_pop}'])
	if gf_vars.get('exodus_rate'):
		parameter_dict[sage.all.SR.var('T')] = sage.all.Rational(sim_config['T']/(2*sim_config[f'Ne_{reference_pop}']))
	for c, Ne in zip(coalescence_rates,('Ne_A_B', 'Ne_A', 'Ne_B')):
		if Ne in sim_config:
			parameter_dict[c] = sage.all.Rational(sim_config[f'Ne_{reference_pop}']/sim_config[Ne])
		else:
			parameter_dict[c] = 0.0
	return parameter_dict

def get_theta(global_info, sim_config, **kwargs):
	reference_pop = global_info['reference_pop']
	Ne_ref = sim_config[f'Ne_{reference_pop}']
	mu=global_info['mu']
	block_length = global_info['blocklength']
	return 2*sage.all.Rational(Ne_ref*mu)*block_length