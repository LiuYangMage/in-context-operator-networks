
def fill_caption_number(caption_template, identifier, params, format_str = '{:.3g}'):
  if "ode_auto_const" in identifier:
    caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
                              .replace('[0.002]', format_str.format(params[1]))
  elif "ode_auto_linear1" in identifier:
    caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
                              .replace('[0.002]', format_str.format(params[1]))
  elif "ode_auto_linear2" in identifier:
    caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
                              .replace('[0.002]', format_str.format(params[1])) \
                              .replace('[0.003]', format_str.format(params[2]))
  elif "series_damped_oscillator" in identifier:
    caption = caption_template.replace('[0.001]', format_str.format(params[0]))
  elif "pde_poisson_spatial" in identifier:
    caption = caption_template.replace('[0.001]', format_str.format(params[0])) \
                              .replace('[0.002]', format_str.format(params[1]))
  elif "pde_porous_spatial" in identifier:
    # all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_c, coeff_a))
    # coeff_l = lamda_prod_coeff_a
    # coeff_a = coeff_c
    caption = caption_template.replace('[0.003]', format_str.format(params[0])) \
                              .replace('[0.004]', format_str.format(params[1])) \
                              .replace('[0.002]', format_str.format(params[2])) \
                              .replace('[0.001]', format_str.format(params[3] * 0.05))
  elif "pde_cubic_spatial" in identifier:
    # all_params.append("{:.8f}_{:.8f}_{:.8f}_{:.8f}".format(coeff_ul, coeff_ur, coeff_a, coeff_k))
    # coeff_l = lamda_prod_coeff_a
    # coeff_a = coeff_k
    caption = caption_template.replace('[0.003]', format_str.format(params[0])) \
                              .replace('[0.004]', format_str.format(params[1])) \
                              .replace('[0.001]', format_str.format(params[2] * 0.1)) \
                              .replace('[0.002]', format_str.format(params[3]))
  elif "mfc_gparam_hj" in identifier:
    str_gvals = ', '.join([format_str.format(g) for g in params])
    caption = caption_template.replace('[g]', str_gvals)
  elif "mfc_rhoparam_hj" in identifier:
    str_gvals = ', '.join([format_str.format(g) for g in params])
    caption = caption_template.replace('[\\rho_0]', str_gvals)
  else:
    raise NotImplementedError
  return caption