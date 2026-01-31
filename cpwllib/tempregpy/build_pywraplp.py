from .functions import *
from .plotting import *
from .utils import *
from ortools.linear_solver import pywraplp

HULL_SURF = 'Hull constraint: Surface'
HULL_RD = 'Hull constraint: Ramping down for '
HULL_RU = 'Hull constraint: Ramping up for '


def build_and_run(inputs, milp=False):

    # Sets
    time = ["t{:02}".format(i) for i in range(1, 25)]
    units = ["UnitA"]
    plants = ["PlantA"]

    if inputs['cnstr_temp_convex_hull'] is True:
        hulls = {}
        ### Compute hull coefficients
        data_gen = DataGenerator(inputs['Q_tur_min_values'], inputs['Q_tur_max_values'],
                                 inputs['Q_bp_min_values'], inputs['Q_bp_max_values'],
                                 inputs['Q_tot_min_values'], inputs['Q_tot_max_values'])
        hulls['hull_X'] = HullComputation(data_gen.points_X, "X = Q_bp * Q_tur1")
        hulls['hull_Y'] = HullComputation(data_gen.points_Y, "Y = Q_tur * Q_bp1")
        hulls['hull_Z'] = HullComputation(data_gen.points_Z, "Z = Q_tot * Q_tot1")

        hulls['hull_X'].print_plane_equations()
        hulls['hull_Y'].print_plane_equations()
        hulls['hull_Z'].print_plane_equations()

        if inputs['plt_show'] is True:
            Plotter(data_gen, hulls['hull_X'], hulls['hull_Y'], hulls['hull_Z'])

    ### Assign parameters
    # Example of logging parameter assignment with summarization
    logger.info('Assigning parameters for optimization:')
    parameters = {}
    parameters['LMP'] = {(t): inputs['LMP_values'][i] for i, t in enumerate(time)}
    parameters['T_RM61_hourly_max'] = {(t): inputs['T_RM61_hourly_max_values'][i] for i, t in enumerate(time)}
    parameters['T_RM61_hourly_min'] = {(t): inputs['T_RM61_hourly_min_values'][i] for i, t in enumerate(time)}
    parameters['dT_hourly'] = {(t): inputs['dT_hourly_values'][i] for i, t in enumerate(time)}
    parameters['T_bp'] = {(t, u): inputs['T_bp_values'][i] for i, t in enumerate(time) for u in units}
    parameters['T_tur'] = {(t, u): inputs['T_tur_values'][i] for i, t in enumerate(time) for u in units}
    parameters['Q_bp_min'] = {(t, u): inputs['Q_bp_min_values'][i] for i, t in enumerate(time) for u in units}
    parameters['Q_bp_max'] = {(t, u): inputs['Q_bp_max_values'][i] for i, t in enumerate(time) for u in units}
    parameters['Q_tur_min'] = {(t, u): inputs['Q_tur_min_values'][i] for i, t in enumerate(time) for u in units}
    parameters['Q_tur_max'] = {(t, u): inputs['Q_tur_max_values'][i] for i, t in enumerate(time) for u in units}
    parameters['P_tur_min'] = {(t, u): inputs['P_tur_min_values'][i] for i, t in enumerate(time) for u in units}
    parameters['P_tur_max'] = {(t, u): inputs['P_tur_max_values'][i] for i, t in enumerate(time) for u in units}
    parameters['Q_tot_min'] = {t: inputs['Q_tot_min_values'][i] for i, t in enumerate(time)}
    parameters['Q_tot_max'] = {t: inputs['Q_tot_max_values'][i] for i, t in enumerate(time)}
    parameters['Demand'] = {(t): inputs['Demand_values'][i] for i, t in enumerate(time)}

    def log_sample_summary(logger, name, parameter_dict, sample_size=5):
        logger.debug(f"{name} assigned (sample): {list(parameter_dict.values())[:sample_size]} ... (total: {len(parameter_dict)})")

    # Log summary of LMP values
    log_sample_summary(logger, "LMP values", parameters['LMP'])

    # Log summary of temperature and demand parameters
    log_sample_summary(logger, "T_RM61_hourly_max values", parameters['T_RM61_hourly_max'])
    log_sample_summary(logger, "T_RM61_hourly_min values", parameters['T_RM61_hourly_min'])
    log_sample_summary(logger, "dT_hourly values", parameters['dT_hourly'])
    log_sample_summary(logger, "Demand values", parameters['Demand'])

    # Log total counts for other parameters
    log_sample_summary(logger, "T_bp", parameters['T_bp'])
    log_sample_summary(logger, "T_tur", parameters['T_tur'])
    log_sample_summary(logger, "Q_bp_min", parameters['Q_bp_min'])
    log_sample_summary(logger, "Q_bp_max", parameters['Q_bp_max'])
    log_sample_summary(logger, "Q_tur_min", parameters['Q_tur_min'])
    log_sample_summary(logger, "Q_tur_max", parameters['Q_tur_max'])
    log_sample_summary(logger, "P_tur_min", parameters['P_tur_min'])
    log_sample_summary(logger, "P_tur_max", parameters['P_tur_max'])
    log_sample_summary(logger, "Q_tot_min", parameters['Q_tot_min'])
    log_sample_summary(logger, "Q_tot_max", parameters['Q_tot_max'])

    # Create solver
    if milp is True:
        solver_name = 'SCIP'
    else:
        solver_name = 'GLOP'  # Example solver name
    solver = pywraplp.Solver.CreateSolver(solver_name)

    solver.EnableOutput()
    solver.iterations()
    # Log solver information
    logger.info(f"Created optimization solver '{solver_name}'.")

    variables = {}
    # Variables
    variables['q_bp'] = {(t, u): solver.NumVar(parameters['Q_bp_min'][t, u],
                                               parameters['Q_bp_max'][t, u],
                                               'q_bp[%s,%s]' % (t, u)) for t in time for u in units
                         }
    variables['q_tur'] = {(t, u): solver.NumVar(parameters['Q_tur_min'][t, u],
                                                parameters['Q_tur_max'][t, u],
                                                'q_tur[%s,%s]' % (t, u)) for t in time for u in units
                          }
    variables['q_tot'] = {(t, u): solver.NumVar(parameters['Q_tot_min'][t],
                                                parameters['Q_tot_max'][t],
                                                'q_tot[%s,%s]' % (t, u)) for t in time for u in units
                          }
    variables['q_bp_t_tur_t1'] = {(t, u): solver.NumVar(parameters['Q_bp_min'][t, u] * parameters['Q_tur_min'][t, u],
                                                        parameters['Q_bp_max'][t, u] * parameters['Q_tur_max'][t, u],
                                                        'q_bp_t_tur_t1[%s,%s]' % (t, u)) for t in time for u in units
                                  }
    variables['q_tur_t_bp_t1'] = {(t, u): solver.NumVar(parameters['Q_bp_min'][t, u] * parameters['Q_tur_min'][t, u],
                                                        parameters['Q_bp_max'][t, u] * parameters['Q_tur_max'][t, u],
                                                        'q_tur_t_bp_t1[%s,%s]' % (t, u)) for t in time for u in units
                                  }
    variables['q_tot_t_tot_t1'] = {(t, u): solver.NumVar(parameters['Q_tot_min'][t] * parameters['Q_tot_min'][t],
                                                         parameters['Q_tot_max'][t] * parameters['Q_tot_max'][t],
                                                         'q_tot_t_tot_t1[%s,%s]' % (t, u)) for t in time for u in units
                                   }
    variables['q_bp_ramp_up'] = {(t, u): solver.NumVar(0,
                                                       inputs['Q_bp_ramp_up_max'],
                                                       'q_bp_ramp_up[%s,%s]' % (t, u)) for t in time for u in units
                                 }
    variables['q_bp_ramp_down'] = {(t, u): solver.NumVar(0,
                                                         inputs['Q_bp_ramp_down_max'],
                                                         'q_bp_ramp_down[%s,%s]' % (t, u)) for t in time for u in units
                                   }
    variables['q_tur_ramp_up'] = {(t, u): solver.NumVar(0,
                                                        inputs['Q_tur_ramp_up_max'],
                                                        'q_tur_ramp_up[%s,%s]' % (t, u)) for t in time for u in units
                                  }
    variables['q_tur_ramp_down'] = {(t, u): solver.NumVar(0,
                                                          inputs['Q_tur_ramp_down_max'],
                                                          'q_tur_ramp_down[%s,%s]' % (t, u)) for t in time for u in units
                                    }
    variables['q_tot_ramp_up'] = {(t, u): solver.NumVar(0,
                                                        inputs['Q_tot_ramp_up_max'],
                                                        'q_tot_ramp_up[%s,%s]' % (t, u)) for t in time for u in units
                                  }
    variables['q_tot_ramp_down'] = {(t, u): solver.NumVar(0,
                                                          inputs['Q_tot_ramp_down_max'],
                                                          'q_tot_ramp_down[%s,%s]' % (t, u)) for t in time for u in units
                                    }
    variables['q_tot_min'] = {(u): solver.NumVar(min(parameters['Q_tot_min'].values()),
                                                 max(parameters['Q_tot_max'].values()),
                                                 'q_tot_min[%s]' % (u)) for u in units
                              }
    variables['p_tur'] = {(t, u): solver.NumVar(parameters['P_tur_min'][t, u],
                                                parameters['P_tur_max'][t, u],
                                                'p_tur[%s,%s]' % (t, u)) for t in time for u in units
                          }

    logger.info("Optimization variables sucesfully created.")

    # Constraints
    constraints = []
    constraint_id_main = []
    constraint_id_unit = []
    constraint_id_time = []

    # Add constraints for total hourly release
    for u in units:
        for t in time:
            constraint_id_main.append('Total hourly release')
            constraint_id_unit.append(f'{u}')
            constraint_id_time.append(f'{t}')
            constraints.append(
                solver.Add(variables['q_bp'][t, u] + variables['q_tur'][t, u] == variables['q_tot'][t, u]))

    # Add constraint for total daily release
    constraint_id_main.append('Total daily release')
    constraint_id_unit.append('')
    constraint_id_time.append('')
    constraints.append(
        solver.Add(sum(variables['q_tot'][t, u] for u in units for t in time) == inputs['Q_tot_daily']))

    # Add constraints for maximum average daily temperature
    constraint_id_main.append('Maximum average daily temperature')
    constraint_id_unit.append('')
    constraint_id_time.append('')
    constraints.append(
        solver.Add(sum(
            variables['q_bp'][t, u] * parameters['T_bp'][t, u] + variables['q_tur'][t, u] * parameters['T_tur'][t, u]
            for u in units for t in time) <=
                   (inputs['T_RM61_daily'] - inputs['dT_daily']) * sum(
            variables['q_tot'][t, u] for u in units for t in time)))

    if inputs['cnstr_temp_t01t24']:
        logger.info('Temperature constraints at H01 == H24 have been included in the optimization.')
        if (inputs['cnstr_tur_t01t24'] or inputs['cnstr_bp_t01t24'] or inputs['cnstr_tot_t01t24']):
            logger.warning('Temperature constraints at H01 == H24 may conflict with cyclic turbine, bypass, or total release constraints.')
        constraint_id_main.append('Temperature t01 = t24')
        constraint_id_unit.append('')
        constraint_id_time.append('')
        constraints.append(solver.Add(sum(
            variables['q_bp'][time[0], u] * parameters['T_bp'][time[0], u] + variables['q_tur'][time[0], u] *
            parameters['T_tur'][time[0], u] - (
                        parameters['T_RM61_hourly_max'][time[0]] - parameters['dT_hourly'][time[0]]) *
            variables['q_tot'][time[0], u] for u in units) ==
                                      sum(variables['q_bp'][time[-1], u] * parameters['T_bp'][time[-1], u] +
                                          variables['q_tur'][time[-1], u] * parameters['T_tur'][time[-1], u] - (
                                                      parameters['T_RM61_hourly_max'][time[-1]] -
                                                      parameters['dT_hourly'][time[-1]]) * variables['q_tot'][
                                              time[-1], u] for u in units)))

    if inputs['cnstr_bp_t01t24']:
        logger.info('Bypass release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Bypass release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')

        constraint_id_main.append('q_bp t01 = t24')
        constraint_id_unit.append('')
        constraint_id_time.append('')
        constraints.append(solver.Add(
            sum(variables['q_bp'][time[0], u] for u in units) == sum(variables['q_bp'][time[-1], u] for u in units)))

    if inputs['cnstr_tur_t01t24']:
        logger.info('Turbine release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Turbine release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        constraint_id_main.append('q_tur t01 = t24')
        constraint_id_unit.append('')
        constraint_id_time.append('')
        constraints.append(solver.Add(
            sum(variables['q_tur'][time[0], u] for u in units) == sum(variables['q_tur'][time[-1], u] for u in units)))

    if inputs['cnstr_tot_t01t24']:
        logger.info('Total release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Total release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        constraint_id_main.append('q_tot t01 = t24')
        constraint_id_unit.append('')
        constraint_id_time.append('')
        constraints.append(solver.Add(
            sum(variables['q_tot'][time[0], u] for u in units) == sum(variables['q_tot'][time[-1], u] for u in units)))

    if inputs['cnstr_oper_range'] is True:
        logger.info('Maximum daily operating range constraints have been included in the optimization.')
        for t in time:
            # Add constraints for maximum total daily ramping rate 8000CFS
            constraint_id_main.append('Maximum daily operating range - min')
            constraint_id_unit.append('')
            constraint_id_time.append(f'{t}')
            constraints.append(
                solver.Add(
                    sum(variables['q_tot_min'][u] for u in units) <= sum(variables['q_tot'][t, u] for u in units)))

        for t in time:
            # Add constraints for maximum total daily ramping rate 8000CFS
            constraint_id_main.append('Maximum daily operating range - max')
            constraint_id_unit.append('')
            constraint_id_time.append(f'{t}')
            constraints.append(
                solver.Add(sum(variables['q_tot_min'][u] for u in units) + inputs['Q_max_daily_change'] >= sum(
                    variables['q_tot'][t, u] for u in units)))

    # Add constraints for maximum hourly temperature
    for t in time:
        constraint_id_main.append('Maximum hourly temperature')
        constraint_id_unit.append('')
        constraint_id_time.append(f'{t}')
        constraints.append(solver.Add(sum(
            variables['q_bp'][t, u] * parameters['T_bp'][t, u] + variables['q_tur'][t, u] * parameters['T_tur'][t, u]
            for u in units) <=
                                      (parameters['T_RM61_hourly_max'][t] - parameters['dT_hourly'][t]) * sum(
            variables['q_tot'][t, u] for u in units)))

    # Add constraints for minimum hourly temperature
    for t in time:
        constraint_id_main.append('Minimum hourly temperature')
        constraint_id_unit.append('')
        constraint_id_time.append(f'{t}')
        constraints.append(solver.Add(sum(
            variables['q_bp'][t, u] * parameters['T_bp'][t, u] + variables['q_tur'][t, u] * parameters['T_tur'][t, u]
            for u in units) >=
                                      (parameters['T_RM61_hourly_min'][t] - parameters['dT_hourly'][t]) * sum(
            variables['q_tot'][t, u] for u in units)))

    # Add constraints for power balance
    for u in units:
        for t in time:
            constraint_id_main.append('Power Output')
            constraint_id_unit.append(f'{u}')
            constraint_id_time.append(f'{t}')
            constraints.append(
                solver.Add(variables['p_tur'][t, u] == variables['q_tur'][t, u] * inputs['WaterToPowerConversion']))

    # Add constraints for power balance
    for t in time:
        constraint_id_main.append('Power Demand Balance')
        constraint_id_unit.append('')
        constraint_id_time.append(f'{t}')
        constraints.append(solver.Add(parameters['Demand'][t] - sum(variables['p_tur'][t, u] for u in units) >= 0))

    if inputs['cnstr_bp_ramp'] is True:
        logger.info('Bypass release ramping constraints have been included in the optimization.')
        # Add constraint for bypass ramping up rate
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Bypass ramping')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        variables['q_bp'][time[0], u] - variables['q_bp'][time[-1], u] <= variables['q_bp_ramp_up'][
                            t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_bp'][t, u] - variables['q_bp'][prev_t, u] <= variables['q_bp_ramp_up'][t, u]))

        # Add constraint for bypass ramping down rate
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Bypass ramping')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        variables['q_bp'][time[-1], u] - variables['q_bp'][time[0], u] <= variables['q_bp_ramp_down'][
                            t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_bp'][prev_t, u] - variables['q_bp'][t, u] <= variables['q_bp_ramp_down'][t, u]))

    if inputs['cnstr_tur_ramp'] is True:
        logger.info('Turbine release ramping constraints have been included in the optimization.')
        # Add constraint for turbine ramping up rate
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Turbine ramping')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        variables['q_tur'][time[0], u] - variables['q_tur'][time[-1], u] <= variables['q_tur_ramp_up'][
                            t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_tur'][t, u] - variables['q_tur'][prev_t, u] <= variables['q_tur_ramp_up'][t, u]))

        # Add constraint for turbine ramping down rate
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Turbine ramping')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(variables['q_tur'][time[-1], u] - variables['q_tur'][time[0], u] <=
                                                  variables['q_tur_ramp_down'][t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_tur'][prev_t, u] - variables['q_tur'][t, u] <= variables['q_tur_ramp_down'][t, u]))

    if inputs['cnstr_tot_ramp']:
        logger.info('Total release ramping constraints have been included in the optimization.')
        # Add constraint for total ramping up constraint
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Ramp up constraint')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        variables['q_tot'][time[0], u] - variables['q_tot'][time[-1], u] <= variables['q_tot_ramp_up'][
                            t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_tot'][t, u] - variables['q_tot'][prev_t, u] <= variables['q_tot_ramp_up'][t, u]))

        # Add constraint for total ramping down constraint
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append('Ramp down constraint')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(variables['q_tot'][time[-1], u] - variables['q_tot'][time[0], u] <=
                                                  variables['q_tot_ramp_down'][t, u]))
                else:
                    constraints.append(solver.Add(
                        variables['q_tot'][prev_t, u] - variables['q_tot'][t, u] <= variables['q_tot_ramp_down'][t, u]))

    #####################################################################
    ##### Hull surface area constraints
    if inputs['cnstr_temp_convex_hull'] is True:
        logger.info('Convex hull temperature ramping approximation constraints have been included in the optimization.')
        for u in units:
            for t in time:
                constraint_id_main.append('Temperature ramp up constraint')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                constraints.append(solver.Add((variables['q_bp_t_tur_t1'][t, u] - variables['q_tur_t_bp_t1'][t, u]) * (
                            parameters['T_bp'][t, u] - parameters['T_tur'][t, u]) <=
                                              inputs['T_ramp_up_delta'] * variables['q_tot_t_tot_t1'][t, u]))

        for u in units:
            for t in time:
                constraint_id_main.append('Temperature ramp down constraint')
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                constraints.append(solver.Add((variables['q_bp_t_tur_t1'][t, u] - variables['q_tur_t_bp_t1'][t, u]) * (
                            parameters['T_bp'][t, u] - parameters['T_tur'][t, u]) >=
                                              -inputs['T_ramp_down_delta'] * variables['q_tot_t_tot_t1'][t, u]))

        for j in range(0, 4):
            for u in units:
                for i in range(len(time)):
                    t = time[i]
                    prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                    constraint_id_main.append(HULL_SURF + str(j) + ' for ' + hulls['hull_X'].label)
                    constraint_id_unit.append(f'{u}')
                    constraint_id_time.append(f'{t}')
                    if i == 0:
                        if hulls['hull_X'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_X'].plane_equations[j][0] * variables['q_bp'][time[0], u] +
                                           hulls['hull_X'].plane_equations[j][1] * variables['q_tur'][time[-1], u] +
                                           hulls['hull_X'].plane_equations[j][2] * variables['q_bp_t_tur_t1'][
                                               time[0], u] <=
                                           hulls['hull_X'].plane_equations[j][3]))
                        if hulls['hull_X'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_X'].plane_equations[j][0] * variables['q_bp'][time[0], u] +
                                           hulls['hull_X'].plane_equations[j][1] * variables['q_tur'][time[-1], u] +
                                           hulls['hull_X'].plane_equations[j][2] * variables['q_bp_t_tur_t1'][
                                               time[0], u] >=
                                           hulls['hull_X'].plane_equations[j][3]))
                    else:
                        if hulls['hull_X'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_X'].plane_equations[j][0] * variables['q_bp'][t, u] +
                                           hulls['hull_X'].plane_equations[j][1] * variables['q_tur'][prev_t, u] +
                                           hulls['hull_X'].plane_equations[j][2] * variables['q_bp_t_tur_t1'][t, u] <=
                                           hulls['hull_X'].plane_equations[j][3]))
                        if hulls['hull_X'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_X'].plane_equations[j][0] * variables['q_bp'][t, u] +
                                           hulls['hull_X'].plane_equations[j][1] * variables['q_tur'][prev_t, u] +
                                           hulls['hull_X'].plane_equations[j][2] * variables['q_bp_t_tur_t1'][t, u] >=
                                           hulls['hull_X'].plane_equations[j][3]))

        for j in range(0, 4):
            for u in units:
                for i in range(len(time)):
                    t = time[i]
                    prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                    constraint_id_main.append(HULL_SURF + str(j) + ' for ' + hulls['hull_Y'].label)
                    constraint_id_unit.append(f'{u}')
                    constraint_id_time.append(f'{t}')
                    if i == 0:
                        if hulls['hull_Y'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_Y'].plane_equations[j][0] * variables['q_tur'][time[0], u] +
                                           hulls['hull_Y'].plane_equations[j][1] * variables['q_bp'][time[-1], u] +
                                           hulls['hull_Y'].plane_equations[j][2] * variables['q_tur_t_bp_t1'][
                                               time[0], u] <=
                                           hulls['hull_Y'].plane_equations[j][3]))
                        if hulls['hull_Y'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_Y'].plane_equations[j][0] * variables['q_tur'][time[0], u] +
                                           hulls['hull_Y'].plane_equations[j][1] * variables['q_bp'][time[-1], u] +
                                           hulls['hull_Y'].plane_equations[j][2] * variables['q_tur_t_bp_t1'][
                                               time[0], u] >=
                                           hulls['hull_Y'].plane_equations[j][3]))
                    else:
                        if hulls['hull_Y'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_Y'].plane_equations[j][0] * variables['q_tur'][t, u] +
                                           hulls['hull_Y'].plane_equations[j][1] * variables['q_bp'][prev_t, u] +
                                           hulls['hull_Y'].plane_equations[j][2] * variables['q_tur_t_bp_t1'][t, u] <=
                                           hulls['hull_Y'].plane_equations[j][3]))
                        if hulls['hull_Y'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_Y'].plane_equations[j][0] * variables['q_tur'][t, u] +
                                           hulls['hull_Y'].plane_equations[j][1] * variables['q_bp'][prev_t, u] +
                                           hulls['hull_Y'].plane_equations[j][2] * variables['q_tur_t_bp_t1'][t, u] >=
                                           hulls['hull_Y'].plane_equations[j][3]))

        for j in range(0, 4):
            for u in units:
                for i in range(len(time)):
                    t = time[i]
                    prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                    constraint_id_main.append(HULL_SURF + str(j) + ' for ' + hulls['hull_Z'].label)
                    constraint_id_unit.append(f'{u}')
                    constraint_id_time.append(f'{t}')
                    if i == 0:
                        if hulls['hull_Z'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_Z'].plane_equations[j][0] * variables['q_tot'][time[0], u] +
                                           hulls['hull_Z'].plane_equations[j][1] * variables['q_tot'][time[-1], u] +
                                           hulls['hull_Z'].plane_equations[j][2] * variables['q_tot_t_tot_t1'][
                                               time[0], u] <=
                                           hulls['hull_Z'].plane_equations[j][3]))
                        if hulls['hull_Z'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_Z'].plane_equations[j][0] * variables['q_tot'][time[0], u] +
                                           hulls['hull_Z'].plane_equations[j][1] * variables['q_tot'][time[-1], u] +
                                           hulls['hull_Z'].plane_equations[j][2] * variables['q_tot_t_tot_t1'][
                                               time[0], u] >=
                                           hulls['hull_Z'].plane_equations[j][3]))
                    else:
                        if hulls['hull_Z'].inequality_signs[j] == '<=':
                            constraints.append(
                                solver.Add(hulls['hull_Z'].plane_equations[j][0] * variables['q_tot'][t, u] +
                                           hulls['hull_Z'].plane_equations[j][1] * variables['q_tot'][prev_t, u] +
                                           hulls['hull_Z'].plane_equations[j][2] * variables['q_tot_t_tot_t1'][t, u] <=
                                           hulls['hull_Z'].plane_equations[j][3]))
                        if hulls['hull_Z'].inequality_signs[j] == '>=':
                            constraints.append(
                                solver.Add(hulls['hull_Z'].plane_equations[j][0] * variables['q_tot'][t, u] +
                                           hulls['hull_Z'].plane_equations[j][1] * variables['q_tot'][prev_t, u] +
                                           hulls['hull_Z'].plane_equations[j][2] * variables['q_tot_t_tot_t1'][t, u] >=
                                           hulls['hull_Z'].plane_equations[j][3]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RD + hulls['hull_X'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_bp_min'][time[0], u] - inputs['Q_bp_ramp_down_max']) * variables['q_tur'][
                            time[-1], u] <= variables['q_bp_t_tur_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_bp_min'][t, u] - inputs['Q_bp_ramp_down_max']) * variables['q_tur'][
                            prev_t, u] <= variables['q_bp_t_tur_t1'][t, u]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RU + hulls['hull_X'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_bp_max'][time[0], u] + inputs['Q_bp_ramp_up_max']) * variables['q_tur'][
                            time[-1], u] >= variables['q_bp_t_tur_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_bp_max'][t, u] + inputs['Q_bp_ramp_up_max']) * variables['q_tur'][
                            prev_t, u] >= variables['q_bp_t_tur_t1'][t, u]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RD + hulls['hull_Y'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_tur_min'][time[0], u] - inputs['Q_tur_ramp_down_max']) * variables['q_bp'][
                            time[-1], u] <= variables['q_tur_t_bp_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_tur_min'][t, u] - inputs['Q_tur_ramp_down_max']) * variables['q_bp'][
                            prev_t, u] <= variables['q_tur_t_bp_t1'][t, u]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RU + hulls['hull_Y'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_tur_max'][time[0], u] + inputs['Q_tur_ramp_up_max']) * variables['q_bp'][
                            time[-1], u] >= variables['q_tur_t_bp_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_tur_max'][t, u] + inputs['Q_tur_ramp_up_max']) * variables['q_bp'][
                            prev_t, u] >= variables['q_tur_t_bp_t1'][t, u]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RD + hulls['hull_Z'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_tot_min'][time[0]] - inputs['Q_tot_ramp_down_max']) * variables['q_tot'][
                            time[-1], u] <= variables['q_tot_t_tot_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_tot_min'][t] - inputs['Q_tot_ramp_down_max']) * variables['q_tot'][
                            prev_t, u] <= variables['q_tot_t_tot_t1'][t, u]))

        # Additional hull constraints
        for u in units:
            for i in range(len(time)):
                t = time[i]
                prev_t = time[i - 1] if i > 0 else time[-1]  # Handling wrap-around
                constraint_id_main.append(HULL_RU + hulls['hull_Z'].label)
                constraint_id_unit.append(f'{u}')
                constraint_id_time.append(f'{t}')
                if i == 0:
                    constraints.append(solver.Add(
                        (parameters['Q_tot_max'][time[0]] + inputs['Q_tot_ramp_up_max']) * variables['q_tot'][
                            time[-1], u] >= variables['q_tot_t_tot_t1'][time[0], u]))
                else:
                    constraints.append(solver.Add(
                        (parameters['Q_tot_max'][t] + inputs['Q_tot_ramp_up_max']) * variables['q_tot'][
                            prev_t, u] >= variables['q_tot_t_tot_t1'][t, u]))

    # Objective
    objective = solver.Objective()
    for u in units:
        for t in time:
            objective.SetCoefficient(variables['p_tur'][t, u], parameters['LMP'][t])
            objective.SetCoefficient(variables['q_bp_ramp_down'][t, u], -inputs['C_bp_ramp_down'])
            objective.SetCoefficient(variables['q_bp_ramp_up'][t, u], -inputs['C_bp_ramp_up'])
            objective.SetCoefficient(variables['q_tur_ramp_down'][t, u], -inputs['C_tur_ramp_down'])
            objective.SetCoefficient(variables['q_tur_ramp_up'][t, u], -inputs['C_tur_ramp_up'])
            objective.SetCoefficient(variables['q_tot_ramp_down'][t, u], -inputs['C_tot_ramp_down'])
            objective.SetCoefficient(variables['q_tot_ramp_up'][t, u], -inputs['C_tot_ramp_up'])

    objective.SetMaximization()

    # Determine the directory where the Excel file is located
    base_directory = os.path.dirname(inputs['file_path'])

    # Create the "results" folder in the same directory as the Excel file
    results_folder_path = os.path.join(base_directory, 'Results')
    os.makedirs(results_folder_path, exist_ok=True)

    # Create the simulation folder inside the "results" folder
    simulation_folder_path = os.path.join(results_folder_path, inputs['sim_folder'])

    # Check if the folder already exists
    if os.path.exists(simulation_folder_path):
        logger.warning("The existing simulation will be overwritten.")
    else:
        # Create the folder
        try:
            os.makedirs(simulation_folder_path)
            logger.info(f"The folder '{simulation_folder_path}' was successfully created.")
        except Exception as e:
            logger.error(f"Unable to create the folder '{simulation_folder_path}'. Encountered error: {e}. "
                         f"Please check your permissions and ensure the path is correct.")
            sys.exit(1)

    with open(simulation_folder_path + '\\model.mps', 'w') as file:
        mps_string = solver.ExportModelAsMpsFormat(fixed_format=False, obfuscated=False)
        file.write(mps_string)

    # Solve the problem
    status = solver.Solve()

    # Log solver status
    if status == pywraplp.Solver.OPTIMAL:
        logger.info("Optimization successful: Found optimal solution.")
    elif status == pywraplp.Solver.FEASIBLE:
        logger.info("Optimization terminated: Found feasible solution.")
    elif status == pywraplp.Solver.INFEASIBLE:
        logger.warning("Optimization terminated: Problem is infeasible.")
    elif status == pywraplp.Solver.NOT_SOLVED:
        logger.warning("Optimization not solved: No solution found.")
    else:
        logger.error("Optimization failed: Solver encountered an error.")

    # Check if the problem has an optimal solution
    try:
        if status == pywraplp.Solver.OPTIMAL:
            logger.info('Objective value = ' + str(objective.Value()))
            results = {}
            results['Objective Value'] = objective.Value()
            results['q_tot_min'] = process_single_index(variables['q_tot_min'], units)/CFS_TO_AFH #CFS->AF
            results['q_tur'] = process_double_index(variables['q_tur'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_bp'] = process_double_index(variables['q_bp'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_tot'] = process_double_index(variables['q_tot'], time, units)/CFS_TO_AFH #CFS->AF
            results['p_tur'] = process_double_index(variables['p_tur'], time, units)
            results['q_bp_t_tur_t1'] = process_double_index(variables['q_bp_t_tur_t1'], time, units)
            results['q_tur_t_bp_t1'] = process_double_index(variables['q_tur_t_bp_t1'], time, units)
            results['q_tot_t_tot_t1'] = process_double_index(variables['q_tot_t_tot_t1'], time, units)
            results['q_bp_ramp_up'] = process_double_index(variables['q_bp_ramp_up'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_bp_ramp_down'] = process_double_index(variables['q_bp_ramp_down'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_tur_ramp_up'] = process_double_index(variables['q_tur_ramp_up'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_tur_ramp_down'] = process_double_index(variables['q_tur_ramp_down'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_tot_ramp_up'] = process_double_index(variables['q_tot_ramp_up'], time, units)/CFS_TO_AFH #CFS->AF
            results['q_tot_ramp_down'] = process_double_index(variables['q_tot_ramp_down'], time, units)/CFS_TO_AFH #CFS->AF
            results['T_dam'] = (results['q_tur'] * process_double_index(parameters['T_tur'], time, units, input=True) +
                                results['q_bp'] * process_double_index(parameters['T_bp'], time, units, input=True)) / \
                               results['q_tot']
            results['T_RM61'] = results['T_dam'].T + process_single_index(parameters['dT_hourly'], time, input=True,
                                                                          col_name='UnitA')
            results['T_RM61'].columns = ['T_RM61']
            results['Dual Values'] = export_dual_values(constraints, constraint_id_main, constraint_id_unit,
                                                        constraint_id_time)
            results['T_dam'] = results['T_dam'].T
            results['T_dam'].columns = ['T_dam']

            def concat_and_rename(results, q_tur_value, q_bp_value, unit_name='UnitA'):
                # Selecting desired rows from each dataframe
                q_tur_selected = results[q_tur_value].loc[unit_name, :]
                q_bp_selected = results[q_bp_value].loc[unit_name, :]

                # Concatenating them
                concatenated_df = pd.concat([q_bp_selected, q_tur_selected], axis=1)

                # Assigning index names
                concatenated_df.columns = [q_bp_value, q_tur_value]

                return concatenated_df

            # Usage example
            df_dispatch = concat_and_rename(results, 'q_tur', 'q_bp')
            df_lmp = process_single_index(parameters['LMP'], time, input=True, col_name='LMP')
            # tmp = pd.DataFrame(parameters['Demand'])
            # Concatenate initial dataframes
            results['Temperature_summary'] = pd.concat([
                results['T_RM61'],
                process_single_index(parameters['T_RM61_hourly_max'], time, input=True, col_name='T_RM61_max'),
                process_single_index(parameters['T_RM61_hourly_min'], time, input=True, col_name='T_RM61_min')
            ], axis=1)

            # Calculate 'T_min' and 'T_max'
            results['Temperature_summary']['T_min'] = (
                    results['Temperature_summary']['T_RM61'] - inputs['T_ramp_down_delta']).shift(1)
            results['Temperature_summary'].loc['t01', 'T_min'] = \
            (results['Temperature_summary']['T_RM61'] - inputs['T_ramp_down_delta']).iloc[-1]

            results['Temperature_summary']['T_max'] = (
                    results['Temperature_summary']['T_RM61'] + inputs['T_ramp_up_delta']).shift(1)
            results['Temperature_summary'].loc['t01', 'T_max'] = \
            (results['Temperature_summary']['T_RM61'] + inputs['T_ramp_up_delta']).iloc[-1]

            # Cap 'T_max' and 'T_min' values
            results['Temperature_summary']['T_max'] = results['Temperature_summary'][['T_max', 'T_RM61_max']].min(axis=1)
            results['Temperature_summary']['T_min'] = results['Temperature_summary'][['T_min', 'T_RM61_min']].max(axis=1)

            # Concatenate additional results and add 'd_T' column
            results['Temperature_summary'] = pd.concat([results['Temperature_summary'], results['T_dam']], axis=1)
            results['Temperature_summary']['d_T'] = process_single_index(parameters['dT_hourly'], time, input=True,
                                                                         col_name='UnitA')
            # Export results to a file
            results_file_path = os.path.join(simulation_folder_path, 'optimization_results.txt')
            try:
                with open(results_file_path, 'w') as f:
                    logger.info(f"Writing results to '{results_file_path}'...")
                    f.write('Results:\n')

                    for key, value in results.items():
                        if isinstance(value, pd.DataFrame):
                            f.write(f'{key}:\n{value.to_string()}\n\n')
                        else:
                            f.write(f'{key}: {value}\n')
            except Exception as e:
                logger.error(f"Unable to write results to the file '{results_file_path}'. Error encountered: {e}. "
                             f"Please check file permissions and ensure the directory exists.")
                sys.exit(1)

            for key, value in results.items():
                if isinstance(value, pd.DataFrame):
                    value.to_csv(f'{simulation_folder_path + "\\"}{key}.csv', index=False)
                else:
                    with open(f'{simulation_folder_path + "\\"}{key}.csv', 'w') as f:
                        f.write(str(value))

            if inputs['plt_show'] is True:
                plot_temperature_analysis(df_dispatch, results['Temperature_summary'], df_lmp, save_fig=True,
                                          fig_name=simulation_folder_path + '\\plot')

        else:
            logger.error('The problem does not have an optimal solution. Please review and revise the input data.')

        copy_file_to_destination('TempRegPy\\app.log', simulation_folder_path)

        return inputs, results

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}. The problem does not have an optimal solution. Please review "
                     f"and revise the input data.")
    finally:
        if 'e' in locals() or 'e' in globals():
            input("Press Enter to exit...")
