import numpy as np
import pandas as pd

from .functions import *
from .plotting import *
from .utils import *
from ortools.math_opt.python import mathopt



def build_and_run(inputs):

    #####################
    ###### CONVEX HULLS
    #####################
    gauges_hulls = {}
    for rn in inputs['river_nodes']:
        gauges_hulls[rn] = {}
        for i in inputs['time_horizon']:
            gauges_hulls[rn][i] = gauge_hulls_time(inputs, rn, i)

    logger.info('Assigning parameters for optimization:')
    parameters = {}
    parameters['LMP'] = {(n, t): value for n, row in inputs['LMP_values'].iterrows() for t, value in row.items()}
    parameters['T_gg_hourly_max'] = {(g, t): value for g, row in inputs['T_gauge_hourly_max_values'].iterrows() for t, value in row.items()}
    parameters['T_gg_hourly_min'] = {(g, t): value for g, row in inputs['T_gauge_hourly_min_values'].iterrows() for t, value in row.items()}
    parameters['dT_hourly'] = {(i, j, t): inputs['dT_hourly_values'].loc[f'{i} - {j}', t] if f'{i} - {j}' in inputs['dT_hourly_values'].index else 0
        for i in inputs['river_nodes'] for j in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    # parameters['T_gg_daily'] = {(g, s): value for g, row in inputs['T_gauge_daily'].iterrows() for s, value in row.items()}
    parameters['T_gg_daily'] = {(g, d): value for g, row in inputs['T_gauge_daily'].iterrows() for d, value in row.items()}
    # parameters['dT_daily'] = {(i, j): inputs['dT_daily'].loc[f'{i} - {j}', 'Single input'] if f'{i} - {j}' in inputs['dT_daily'].index else 0
    #     for i in inputs['river_nodes'] for j in inputs['river_nodes']
    # }
    parameters['dT_daily'] = {
        (i, j): inputs['dT_daily'].loc[f'{i} - {j}'].tolist() if f'{i} - {j}' in inputs['dT_daily'].index else [0] * len(inputs['dT_daily'].columns) for i in inputs['river_nodes'] for j in inputs['river_nodes']
    }
    parameters['T_bp'] = {(p, t): value for p, row in inputs['T_bp_values'].iterrows() for t, value in row.items()}
    parameters['T_tur'] = {(p, t): value for p, row in inputs['T_tur_values'].iterrows() for t, value in row.items()}
    parameters['T_source_hourly_max'] = {(p, t): inputs['T_source_hourly_max_values'].loc[f'{p}', t] if f'{p}' in inputs['T_source_hourly_max_values'].index else 25
        for p in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['T_source_hourly_min'] = {(p, t): inputs['T_source_hourly_min_values'].loc[f'{p}', t] if f'{p}' in inputs['T_source_hourly_min_values'].index else 0
        for p in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['T_sink_hourly_max'] = {(p, t): inputs['T_sink_hourly_max_values'].loc[f'{p}', t] if f'{p}' in inputs['T_sink_hourly_max_values'].index else 30
        for p in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['T_sink_hourly_min'] = {(p, t): inputs['T_sink_hourly_min_values'].loc[f'{p}', t] if f'{p}' in inputs['T_sink_hourly_min_values'].index else 0
        for p in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['Q_bp_min'] = {(p, t): value for p, row in inputs['Q_bp_min_values'].iterrows() for t, value in row.items()}
    parameters['Q_bp_max'] = {(p, t): value for p, row in inputs['Q_bp_max_values'].iterrows() for t, value in row.items()}
    parameters['Q_tur_min'] = {(p, t): value for p, row in inputs['Q_tur_min_values'].iterrows() for t, value in row.items()}
    parameters['Q_tur_max'] = {(p, t): value for p, row in inputs['Q_tur_max_values'].iterrows() for t, value in row.items()}
    parameters['Q_tot_min'] = {(p, t): value for p, row in inputs['Q_tot_min_values'].iterrows() for t, value in row.items()}
    parameters['Q_tot_max'] = {(p, t): value for p, row in inputs['Q_tot_max_values'].iterrows() for t, value in row.items()}
    parameters['Q_gg_min'] = {(p, t): value for p, row in inputs['Q_gg_min_values'].iterrows() for t, value in row.items()}
    parameters['Q_gg_max'] = {(p, t): value for p, row in inputs['Q_gg_max_values'].iterrows() for t, value in row.items()}
    parameters['Q_source_max'] = {(p, t): value for p, row in inputs['Q_source_max'].iterrows() for t, value in row.items()}
    parameters['Q_sink_max'] = {(p, t): value for p, row in inputs['Q_sink_max'].iterrows() for t, value in row.items()}
    parameters['P_tur_min'] = {(p, t): value for p, row in inputs['P_tur_min_values'].iterrows() for t, value in row.items()}
    parameters['P_tur_max'] = {(p, t): value for p, row in inputs['P_tur_max_values'].iterrows() for t, value in row.items()}
    parameters['Demand'] = {(n, t): value for n, row in inputs['Demand_values'].iterrows() for t, value in row.items()}
    parameters['Q_bp_ramp_up_max'] = {(p, t): value for p, row in inputs['Q_bp_ramp_up_max'].iterrows() for t, value in row.items()}
    parameters['Q_bp_ramp_down_max'] = {(p, t): value for p, row in inputs['Q_bp_ramp_down_max'].iterrows() for t, value in row.items()}
    parameters['Q_tur_ramp_up_max'] = {(p, t): value for p, row in inputs['Q_tur_ramp_up_max'].iterrows() for t, value in row.items()}
    parameters['Q_tur_ramp_down_max'] = {(p, t): value for p, row in inputs['Q_tur_ramp_down_max'].iterrows() for t, value in row.items()}
    parameters['Q_tot_ramp_up_max'] = {(p, t): value for p, row in inputs['Q_tot_ramp_up_max'].iterrows() for t, value in row.items()}
    parameters['Q_tot_ramp_down_max'] = {(p, t): value for p, row in inputs['Q_tot_ramp_down_max'].iterrows() for t, value in row.items()}
    parameters['Q_gg_ramp_up_max'] = {(p, t): value for p, row in inputs['Q_gg_ramp_up_max'].iterrows() for t, value in row.items()}
    parameters['Q_gg_ramp_down_max'] = {(p, t): value for p, row in inputs['Q_gg_ramp_down_max'].iterrows() for t, value in row.items()}
    parameters['Flow_i_j_max'] = {
        (i, j, t): inputs['Flow_i_j_max'].loc[f'{i} - {j}', t] if f'{i} - {j}' in inputs['Flow_i_j_max'].index else 0
        for i in inputs['river_nodes'] for j in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['Flow_i_j_min'] = {
        (i, j, t): (
            inputs['Flow_i_j_min'].loc[f'{i} - {j}', t] if not inputs['Flow_i_j_min'].empty and f'{i} - {j}' in inputs[
                'Flow_i_j_min'].index else 0)
        for i in inputs['river_nodes'] for j in inputs['river_nodes'] for t in inputs['time_horizon']
    }
    parameters['Plants_to_nodes'] = {(p, n): value for p, row in inputs['plants_nodes'].iterrows() for n, value in row.items()}
    parameters['C_bp_ramp_up'] = {(p, s): value for p, row in inputs['C_bp_ramp_up'].iterrows() for s, value in row.items()}
    parameters['C_bp_ramp_down'] = {(p, s): value for p, row in inputs['C_bp_ramp_down'].iterrows() for s, value in row.items()}
    parameters['C_tur_ramp_up'] = {(p, s): value for p, row in inputs['C_tur_ramp_up'].iterrows() for s, value in row.items()}
    parameters['C_tur_ramp_down'] = {(p, s): value for p, row in inputs['C_tur_ramp_down'].iterrows() for s, value in row.items()}
    parameters['C_tot_ramp_up'] = {(p, s): value for p, row in inputs['C_tot_ramp_up'].iterrows() for s, value in row.items()}
    parameters['C_tot_ramp_down'] = {(p, s): value for p, row in inputs['C_tot_ramp_down'].iterrows() for s, value in row.items()}
    parameters['C_gg_ramp_up'] = {(p, s): value for p, row in inputs['C_gg_ramp_up'].iterrows() for s, value in row.items()}
    parameters['C_gg_ramp_down'] = {(p, s): value for p, row in inputs['C_gg_ramp_down'].iterrows() for s, value in row.items()}
    parameters['C_source'] = {(p, s): value for p, row in inputs['C_source'].iterrows() for s, value in row.items()}
    parameters['C_sink'] = {(p, s): value for p, row in inputs['C_sink'].iterrows() for s, value in row.items()}
    parameters['Q_tot_daily'] = {(p, d): value for p, row in inputs['Q_tot_daily'].iterrows() for d, value in row.items()}
    parameters['Q_max_daily_change'] = {(p, d): value for p, row in inputs['Q_max_daily_change'].iterrows() for d, value in row.items()}
    parameters['WaterToPowerConversion'] = {(p, d): value for p, row in inputs['WaterToPowerConversion'].iterrows() for d, value in row.items()}
    parameters['bp_ramp_up_cyc'] = {(p, s): value for p, row in inputs['bp_ramp_up_cyc'].iterrows() for s, value in row.items()}
    parameters['bp_ramp_down_cyc'] = {(p, s): value for p, row in inputs['bp_ramp_down_cyc'].iterrows() for s, value in row.items()}
    parameters['bp_ramp_tot_cyc'] = {(p, s): value for p, row in inputs['bp_ramp_tot_cyc'].iterrows() for s, value in row.items()}
    parameters['bp_forced_commit'] = {(p, t): value for p, row in inputs['bp_forced_commit'].iterrows() for t, value in row.items()}

    def log_sample_summary(logger, name, parameter_dict, sample_size=5):
        logger.debug(f"{name} assigned (sample): {list(parameter_dict.values())[:sample_size]} ... (total: {len(parameter_dict)})")

    # Log summary of node values
    log_sample_summary(logger, "LMP values", parameters['LMP'])
    log_sample_summary(logger, "Demand values", parameters['Demand'])

    # Log summary of temperature and demand parameters
    log_sample_summary(logger, "T_gg_hourly_max values", parameters['T_gg_hourly_max'])
    log_sample_summary(logger, "T_gg_hourly_min values", parameters['T_gg_hourly_min'])
    log_sample_summary(logger, "dT_hourly values", parameters['dT_hourly'])

    # Log total counts for other parameters
    log_sample_summary(logger, "T_bp", parameters['T_bp'])
    log_sample_summary(logger, "T_tur", parameters['T_tur'])
    log_sample_summary(logger, "Q_bp_min", parameters['Q_bp_min'])
    log_sample_summary(logger, "Q_bp_max", parameters['Q_bp_max'])
    log_sample_summary(logger, "Q_tur_min", parameters['Q_tur_min'])
    log_sample_summary(logger, "Q_tur_max", parameters['Q_tur_max'])
    log_sample_summary(logger, "Q_tot_min", parameters['Q_tot_min'])
    log_sample_summary(logger, "Q_tot_max", parameters['Q_tot_max'])
    log_sample_summary(logger, "Q_gg_min", parameters['Q_gg_min'])
    log_sample_summary(logger, "Q_gg_max", parameters['Q_gg_max'])
    log_sample_summary(logger, "P_tur_min", parameters['P_tur_min'])
    log_sample_summary(logger, "P_tur_max", parameters['P_tur_max'])
    log_sample_summary(logger, 'Q_bp_ramp_up_max', parameters['Q_bp_ramp_up_max'])
    log_sample_summary(logger, 'Q_bp_ramp_down_max', parameters['Q_bp_ramp_down_max'])
    log_sample_summary(logger, 'Q_tur_ramp_up_max', parameters['Q_tur_ramp_up_max'])
    log_sample_summary(logger, 'Q_tur_ramp_down_max', parameters['Q_tur_ramp_down_max'])
    log_sample_summary(logger, 'Q_tot_ramp_up_max', parameters['Q_tot_ramp_up_max'])
    log_sample_summary(logger, 'Q_tot_ramp_down_max', parameters['Q_tot_ramp_down_max'])
    log_sample_summary(logger, 'Q_gg_ramp_up_max', parameters['Q_gg_ramp_up_max'])
    log_sample_summary(logger, 'Q_gg_ramp_down_max', parameters['Q_gg_ramp_down_max'])
    log_sample_summary(logger, 'C_bp_ramp_up', parameters['C_bp_ramp_up'])
    log_sample_summary(logger, 'C_bp_ramp_down', parameters['C_bp_ramp_down'])
    log_sample_summary(logger, 'C_tur_ramp_up', parameters['C_tur_ramp_up'])
    log_sample_summary(logger, 'C_tur_ramp_down', parameters['C_tur_ramp_down'])
    log_sample_summary(logger, 'C_tot_ramp_up', parameters['C_tot_ramp_up'])
    log_sample_summary(logger, 'C_tot_ramp_down', parameters['C_tot_ramp_down'])
    log_sample_summary(logger, 'C_gg_ramp_up', parameters['C_gg_ramp_up'])
    log_sample_summary(logger, 'C_gg_ramp_down', parameters['C_gg_ramp_down'])
    log_sample_summary(logger, 'C_source', parameters['C_source'])
    log_sample_summary(logger, 'C_sink', parameters['C_sink'])
    log_sample_summary(logger, 'Q_tot_daily', parameters['Q_tot_daily'])
    log_sample_summary(logger, 'Q_max_daily_change', parameters['Q_max_daily_change'])
    log_sample_summary(logger, 'WaterToPowerConversion', parameters['WaterToPowerConversion'])
    log_sample_summary(logger, 'bp_ramp_up_cyc', parameters['bp_ramp_up_cyc'])
    log_sample_summary(logger, 'bp_ramp_down_cyc', parameters['bp_ramp_down_cyc'])
    log_sample_summary(logger, 'bp_forced_commit', parameters['bp_forced_commit'])

    model = mathopt.Model(name="By-Pass Optimization")

    # Decision variables
    variables = {}
    # Variables
    variables['q_tot_min'] = [[model.add_variable(lb=min(value for (p, t), value in parameters['Q_tot_min'].items() if p == plant),
                                                 ub=max(value for (p, t), value in parameters['Q_tot_max'].items() if p == plant),
                                                 name=f'q_tot_min_{plant}_{d}') for d in inputs['daily_horizon']] for plant in inputs['river_nodes']
                              ]
    variables['p_tur'] = [[model.add_variable(lb=parameters['P_tur_min'][p, t],
                                              ub=parameters['P_tur_max'][p, t],
                                              name=f'p_tur_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_bp'] = [[model.add_variable(lb=0,
                                             ub=parameters['Q_bp_max'][p, t],
                                             name=f'q_bp_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tur'] = [[model.add_variable(lb=parameters['Q_tur_min'][p, t],
                                              ub=parameters['Q_tur_max'][p, t],
                                              name=f'q_tur_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tot'] = [[model.add_variable(lb=parameters['Q_tot_min'][p, t],
                                              ub=parameters['Q_tot_max'][p, t],
                                              name=f'q_tot_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_gg'] = [[model.add_variable(lb=parameters['Q_gg_min'][p, t],
                                             ub=parameters['Q_gg_max'][p, t],
                                             name=f'q_gg_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                        ]
    variables['q_source'] = [[model.add_variable(lb=0,
                                                 ub=parameters['Q_source_max'][p, 'Single input'],
                                                 name=f'q_source_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                            ]
    variables['q_sink'] = [[model.add_variable(lb=0,
                                               ub=parameters['Q_sink_max'][p, 'Single input'],
                                               name=f'q_sink_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                          ]
    variables['q_flow'] = [[[model.add_variable(lb=parameters['Flow_i_j_min'][i,j,t],
                                                ub=parameters['Flow_i_j_max'][i,j,t],
                                                name=f'q_flow_{i}_{j}_{t}') for t in inputs['time_horizon']] for j in inputs['river_nodes']] for i in inputs['river_nodes']
                          ]
    variables['q_in_flow'] = [[model.add_variable(lb=0,
                                                  ub=100000,
                                                  name=f'q_in_flow_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                             ]
    variables['q_out_flow'] = [[model.add_variable(lb=0,
                                                   ub=100000,
                                                   name=f'q_out_flow_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                              ]
    variables['X'] = [[model.add_variable(lb=0,
                                          ub=100000,
                                          name=f'X_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                     ]
    variables['T_gg'] = [[model.add_variable(lb=parameters['T_gg_hourly_min'][j,t],
                                                ub=parameters['T_gg_hourly_max'][j,t],
                                                name=f'T_gauge_{j}_{t}') for t in inputs['time_horizon']] for j in inputs['river_nodes']
                          ]
    variables['q_bp_ramp_up'] = [[model.add_variable(lb=0,
                                                     ub=parameters['Q_bp_ramp_up_max'][p, 'Single input'],
                                                     name=f'q_bp_ramp_up_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_bp_ramp_down'] = [[model.add_variable(lb=0,
                                                       ub=parameters['Q_bp_ramp_down_max'][p, 'Single input'],
                                                       name=f'q_bp_ramp_down_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tur_ramp_up'] = [[model.add_variable(lb=0,
                                                      ub=parameters['Q_tur_ramp_up_max'][p, 'Single input'],
                                                      name=f'q_tur_ramp_up_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tur_ramp_down'] = [[model.add_variable(lb=0,
                                                        ub=parameters['Q_tur_ramp_down_max'][p, 'Single input'],
                                                        name=f'q_tur_ramp_down_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tot_ramp_up'] = [[model.add_variable(lb=0,
                                                      ub=parameters['Q_tot_ramp_up_max'][p, 'Single input'],
                                                      name=f'q_tot_ramp_up_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_tot_ramp_down'] = [[model.add_variable(lb=0,
                                                        ub=parameters['Q_tot_ramp_down_max'][p, 'Single input'],
                                                        name=f'q_tot_ramp_down_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_gg_ramp_up'] = [[model.add_variable(lb=0,
                                                     ub=parameters['Q_gg_ramp_up_max'][p, 'Single input'],
                                                     name=f'q_gg_ramp_up_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    variables['q_gg_ramp_down'] = [[model.add_variable(lb=0,
                                                       ub=parameters['Q_gg_ramp_down_max'][p, 'Single input'],
                                                       name=f'q_gg_ramp_down_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                         ]
    # Slack Variables
    variables['slack_q_tot_daily_min'] = [[model.add_variable(lb=0,
                                                             ub=25000,
                                                             name=f'slack_q_tot_daily_min_{p}_{d}') for d in inputs['daily_horizon']] for p in inputs['river_nodes']
                         ]
    variables['slack_q_tot_daily_max'] = [[model.add_variable(lb=0,
                                                             ub=25000,
                                                             name=f'slack_q_tot_daily_max_{p}_{d}') for d in inputs['daily_horizon']] for p in inputs['river_nodes']
                         ]
    variables['slack_q_bp_cyc_min'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_bp_cyc_min_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_bp_cyc_max'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_bp_cyc_max_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_tur_cyc_min'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_tur_cyc_min_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_tur_cyc_max'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_tur_cyc_max_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_tot_cyc_min'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_tot_cyc_min_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_tot_cyc_max'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_tot_cyc_max_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_gg_cyc_min'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_gg_cyc_min_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_q_gg_cyc_max'] = [model.add_variable(lb=0,
                                                          ub=25000,
                                                          name=f'slack_q_gg_cyc_max_{p}') for p in inputs['river_nodes']
                         ]
    variables['slack_p_node_min'] = [[model.add_variable(lb=0,
                                                       ub=10000,
                                                       name=f'slack_p_tur_min_{n}_{t}') for t in inputs['time_horizon']] for n in inputs['nodes']
                         ]
    variables['slack_p_node_max'] = [[model.add_variable(lb=0,
                                                       ub=10000,
                                                       name=f'slack_p_tur_max_{n}_{t}') for t in inputs['time_horizon']] for n in inputs['nodes']
                         ]
    # MILP Variables
    if inputs['sim_type'] == 'MILP':
        variables['q_bp_rr_up_change'] = [[model.add_binary_variable(name=f'q_bp_rr_up_change_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                                          ]
        variables['q_bp_rr_down_change'] = [[model.add_binary_variable(name=f'q_bp_rr_down_change_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                                            ]
        variables['q_bp_committed'] = [[model.add_integer_variable(lb=parameters['bp_forced_commit'][p,t], ub=1,
                                                                   name=f'q_bp_committed_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                                       ]
        variables['q_tur_committed'] = [[model.add_binary_variable(name=f'q_tur_committed_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
                                        ]

    logger.info(f"Optimization variables sucesfully created. {list(variables)}")

    ###################################
    ######## OBJECTIVE FUNCTION
    ###################################
    obj_fnc = mathopt.LinearExpression()
    for p in range(len(inputs['river_nodes'])):
        plant = inputs['river_nodes'][p]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            obj_fnc += (sum(variables['p_tur'][p][t] * parameters['LMP'][n, time] * parameters['Plants_to_nodes'][plant, n] for n in inputs['nodes'])
                        - variables['q_bp_ramp_up'][p][t] * parameters['C_bp_ramp_up'][plant, 'Single input']
                        - variables['q_bp_ramp_down'][p][t] * parameters['C_bp_ramp_down'][plant, 'Single input']
                        - variables['q_tur_ramp_up'][p][t] * parameters['C_tur_ramp_up'][plant, 'Single input']
                        - variables['q_tur_ramp_down'][p][t] * parameters['C_tur_ramp_down'][plant, 'Single input']
                        - variables['q_tot_ramp_up'][p][t] * parameters['C_tot_ramp_up'][plant, 'Single input']
                        - variables['q_tot_ramp_down'][p][t] * parameters['C_tot_ramp_down'][plant, 'Single input']
                        - variables['q_gg_ramp_up'][p][t] * parameters['C_gg_ramp_up'][plant, 'Single input']
                        - variables['q_gg_ramp_down'][p][t] * parameters['C_gg_ramp_down'][plant, 'Single input']
                        )
        for d in range(len(inputs['daily_horizon'])):
            obj_fnc += - (variables['slack_q_tot_daily_min'][p][d] + variables['slack_q_tot_daily_max'][p][d]) * 1000

        obj_fnc += - (variables['slack_q_bp_cyc_min'][p] + variables['slack_q_bp_cyc_min'][p] +
                      variables['slack_q_tur_cyc_min'][p] + variables['slack_q_tur_cyc_min'][p] +
                      variables['slack_q_tot_cyc_min'][p] + variables['slack_q_tot_cyc_max'][p] +
                      variables['slack_q_gg_cyc_min'][p] + variables['slack_q_gg_cyc_max'][p]) * 1000

    for t in range(len(inputs['time_horizon'])):
        obj_fnc += (- sum((variables['slack_p_node_min'][n][t] + variables['slack_p_node_max'][n][t]) * 1000 for n in range(len(inputs['nodes']))))

    model.maximize(obj_fnc)

    ###################################
    ######## constants ###
    ###################################
    hours_per_day = 24
    days_per_week = 7

    # Constraints
    constraints = []
    constraint_id_main = []
    constraint_id_unit = []
    constraint_id_time = []

    ###################################
    ######## TOTAL RELEASE
    ###################################
    # Add constraints for total hourly release
    for p in range(len(inputs['river_nodes'])):
        for t in range(len(inputs['time_horizon'])):
            constraint_id_main.append('Total hourly release')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_bp'][p][t] + variables['q_tur'][p][t] == variables['q_tot'][p][t]))

    # # Add constraint for total daily release
    # for p in range(len(inputs['river_nodes'])):
    #     constraint_id_main.append('Total daily release')
    #     constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
    #     constraint_id_time.append('')
    #     constraints.append(model.add_linear_constraint(sum(variables['q_tot'][p][t] for t in range(len(inputs['time_horizon']))) + variables['slack_q_tot_daily_min'][p] == parameters['Q_tot_daily'][inputs['river_nodes'][p],'Single input'] + variables['slack_q_tot_daily_max'][p]))

    for p in range(len(inputs['river_nodes'])):
        for d in range(days_per_week):
            constraint_id_main.append('Total daily release')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'D{d + 1:02}')

            start_hour = d * hours_per_day
            end_hour = start_hour + hours_per_day

            constraints.append(
                model.add_linear_constraint(
                    sum(variables['q_tot'][p][t] for t in range(start_hour, end_hour)) +
                    variables['slack_q_tot_daily_min'][p][d] == parameters['Q_tot_daily'][
                        (inputs['river_nodes'][p], f'D{d + 1:02}')] + variables['slack_q_tot_daily_max'][p][d]
                )
            )

    ###################################
    ######## BINARY VARIABLES
    ###################################
    if inputs['sim_type'] == 'MILP':
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                constraint_id_main.append('Hourly bypass ramp up change status upper bound')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp_ramp_up'][p][t] <= variables['q_bp_rr_up_change'][p][t]*parameters['Q_bp_ramp_up_max'][inputs['river_nodes'][p], 'Single input'],
                                                               name=f'q_bp_rr_up_change UB for {p} at {t}'))

        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day
                constraint_id_main.append('Daily bypass ramp up change limit')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'D{d + 1:02}')
                constraints.append(model.add_linear_constraint(parameters['bp_ramp_up_cyc'][inputs['river_nodes'][p], 'Single input'] >= sum(variables['q_bp_rr_up_change'][p][t] for t in range(start_hour,end_hour)),
                                                               name=f'q_bp_rr_up_change UB for {p} at {f'D{d + 1:02}'}'))

        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                constraint_id_main.append('Hourly bypass ramp down change status upper bound')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp_ramp_down'][p][t] <= variables['q_bp_rr_down_change'][p][t]*parameters['Q_bp_ramp_down_max'][inputs['river_nodes'][p], 'Single input'],
                                                               name=f'q_bp_rr_down_change UB for {p} at {t}'))

        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day
                constraint_id_main.append('Daily bypass ramp down change limit')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'D{d + 1:02}')
                constraints.append(model.add_linear_constraint(parameters['bp_ramp_down_cyc'][inputs['river_nodes'][p], 'Single input'] >= sum(variables['q_bp_rr_down_change'][p][t] for t in range(start_hour,end_hour)),
                                                               name=f'q_bp_rr_down_change UB for {p} at {f'D{d + 1:02}'}'))

        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                constraint_id_main.append('Bypass committed status upper bound')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp_committed'][p][t]*parameters['Q_bp_max'][inputs['river_nodes'][p],inputs['time_horizon'][t]] >= variables['q_bp'][p][t]))

        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                constraint_id_main.append('Bypass committed status lower bound')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp_committed'][p][t]*parameters['Q_bp_min'][inputs['river_nodes'][p],inputs['time_horizon'][t]] <= variables['q_bp'][p][t]))

        ####TODO: Introduce a daily set
        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day
                constraint_id_main.append('Daily bypass ramp total change limit')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'D{d + 1:02}')
                constraints.append(model.add_linear_constraint(parameters['bp_ramp_tot_cyc'][inputs['river_nodes'][p], 'Single input'] >= sum(variables['q_bp_rr_up_change'][p][t] + variables['q_bp_rr_down_change'][p][t] for t in range(start_hour,end_hour)),
                                                               name=f'q_bp_rr_total_change UB for {p} at {f'D{d + 1:02}'}'))

    ###################################
    ######## FLOWS BETWEEN NODES
    ###################################
    # Add constraint for outflows from a node p
    for i in range(len(inputs['river_nodes'])):
        for t in range(len(inputs['time_horizon'])):
            constraint_id_main.append('Outgoing flows')
            constraint_id_unit.append(f'{inputs['river_nodes'][i]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_out_flow'][i][t] == sum(variables['q_flow'][i][j][t] for j in range(len(inputs['river_nodes'])))))

    # Add a constraint for inflows into a node p
    for i in range(len(inputs['river_nodes'])):
        for t in range(len(inputs['time_horizon'])):
            constraint_id_main.append('Incoming flows')
            constraint_id_unit.append(f'{inputs['river_nodes'][i]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_in_flow'][i][t] == sum(variables['q_flow'][j][i][t] for j in range(len(inputs['river_nodes'])))))

    # Add constraints for hourly flows between nodes
    for p in range(len(inputs['river_nodes'])):
        for t in range(len(inputs['time_horizon'])):
            constraint_id_main.append('Total hourly InFlows')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_gg'][p][t] + variables['q_bp'][p][t] + variables['q_tur'][p][t] ==
                                                           variables['q_source'][p][t] + variables['q_in_flow'][p][t]))
                                                            # variables['q_in_flow'][p][t]))

    # Add constraints for hourly flows between nodes
    for p in range(len(inputs['river_nodes'])):
        for t in range(len(inputs['time_horizon'])):
            constraint_id_main.append('Total hourly OutFlows')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_gg'][p][t] + variables['q_bp'][p][t] + variables['q_tur'][p][t] ==
                                                           variables['q_sink'][p][t] + variables['q_out_flow'][p][t]))
                                                           # variables['q_out_flow'][p][t]))
    ###################################
    ######## CYCLIC CONDITIONS
    ###################################
    # Add constraint for bypass cyclic condition
    if inputs['cnstr_bp_t01t24']:
        logger.info('Bypass release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Bypass release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        for p in range(len(inputs['river_nodes'])):
            constraint_id_main.append('q_bp t01 = t24')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append('')
            constraints.append(model.add_linear_constraint(variables['q_bp'][p][0] + variables['slack_q_bp_cyc_min'][p] == variables['q_bp'][p][len(inputs['time_horizon'])-1] + variables['slack_q_bp_cyc_max'][p]))

    # Add constraint for turbine cyclic condition
    if inputs['cnstr_tur_t01t24']:
        logger.info('Turbine release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Turbine release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        for p in range(len(inputs['river_nodes'])):
            constraint_id_main.append('q_tur t01 = t24')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append('')
            constraints.append(model.add_linear_constraint(
                variables['q_tur'][p][0] + variables['slack_q_tur_cyc_min'][p] == variables['q_tur'][p][len(inputs['time_horizon']) - 1] + variables['slack_q_tur_cyc_max'][p]))

    # Add constraint for total release cyclic condition
    if inputs['cnstr_tot_t01t24']:
        logger.info('Total release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Total release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        for p in range(len(inputs['river_nodes'])):
            constraint_id_main.append('q_tot t01 = t24')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append('')
            constraints.append(model.add_linear_constraint(
                variables['q_tot'][p][0] + variables['slack_q_tot_cyc_min'][p] == variables['q_tot'][p][len(inputs['time_horizon']) - 1] + variables['slack_q_tot_cyc_max'][p]))

    # Add constraint for gauge cyclic condition
    if inputs['cnstr_gg_t01t24']:
        logger.info('Gauge release cyclic constraints (H01=H24) have been included in the optimization.')
        if inputs['cnstr_temp_t01t24']:
            logger.warning('Gauge release cyclic constraints (H01=H24) may conflict with cyclic temperature constraints.')
        for p in range(len(inputs['river_nodes'])):
            constraint_id_main.append('q_gg t01 = t24')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append('')
            constraints.append(model.add_linear_constraint(
                variables['q_gg'][p][0] + variables['slack_q_gg_cyc_min'][p] == variables['q_gg'][p][len(inputs['time_horizon']) - 1] + variables['slack_q_gg_cyc_min'][p]))

    ###################################
    ######## OPERATING RANGE
    ###################################
    if inputs['cnstr_oper_range'] is True:
        logger.info('Maximum daily operating range constraints have been included in the optimization.')
        # Add constraints for maximum total daily ramping rate 8000CFS
        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day
                for t in range(start_hour, end_hour):
                    constraint_id_main.append('Maximum daily operating range - min')
                    constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                    constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                    constraints.append(model.add_linear_constraint(variables['q_tot_min'][p][d] <= variables['q_tot'][p][t]))
        # Add constraints for maximum total daily ramping rate 8000CFS
        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day
                for t in range(start_hour, end_hour):
                    constraint_id_main.append('Maximum daily operating range - max')
                    constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                    constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                    constraints.append(model.add_linear_constraint(variables['q_tot_min'][p][d] + parameters['Q_max_daily_change'][inputs['river_nodes'][p],f'D{d + 1:02}'] >= variables['q_tot'][p][t]))

    ###################################
    ######## RAMPING RATES
    ###################################
    # Add constraint for bypass ramping rates
    if inputs['cnstr_bp_ramp'] is True:
        logger.info('Bypass release ramping constraints have been included in the optimization.')
        # Add constraint for bypass ramping up rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Bypass ramping up')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp'][p][t] - variables['q_bp'][p][prev_t] <= variables['q_bp_ramp_up'][p][t]))

        # Add constraint for bypass ramping down rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp down constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Bypass ramping down')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_bp'][p][prev_t] - variables['q_bp'][p][t] <= variables['q_bp_ramp_down'][p][t]))

    # Add constraint for turbine ramping rates
    if inputs['cnstr_tur_ramp'] is True:
        logger.info('Turbine release ramping constraints have been included in the optimization.')
        # Add constraint for turbine ramping up rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Turbine ramping up')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_tur'][p][t] - variables['q_tur'][p][prev_t] <= variables['q_tur_ramp_up'][p][t]))

        # Add constraint for turbine ramping down rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp down constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Turbine ramping down')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_tur'][p][prev_t] - variables['q_tur'][p][t] <= variables['q_tur_ramp_down'][p][t]))

    # Add constraint for total ramping rates
    if inputs['cnstr_tot_ramp'] is True:
        logger.info('Total release ramping constraints have been included in the optimization.')
        # Add constraint for total ramping up rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Total ramping up')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_tot'][p][t] - variables['q_tot'][p][prev_t] <= variables['q_tot_ramp_up'][p][t]))

        # Add constraint for total ramping down rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp down constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Total ramping down')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_tot'][p][prev_t] - variables['q_tot'][p][t] <= variables['q_tot_ramp_down'][p][t]))

    # Add constraint for gauge ramping rates
    if inputs['cnstr_gg_ramp'] is True:
        logger.info('Gauge release ramping constraints have been included in the optimization.')
        # Add constraint for gauge ramping up rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp up constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Gauge ramping up')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_gg'][p][t] - variables['q_gg'][p][prev_t] <= variables['q_gg_ramp_up'][p][t]))

        # Add constraint for gauge ramping down rate
        for p in range(len(inputs['river_nodes'])):
            for t in range(len(inputs['time_horizon'])):
                prev_t = t-1 if t > 0 else len(inputs['time_horizon'])-1  # Handling wrap-around
                # Add ramp down constraints
                # If t is the first time period, compare it with the last time period to ensure cyclical ramp constraint
                constraint_id_main.append('Gauge ramping down')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_gg'][p][prev_t] - variables['q_gg'][p][t] <= variables['q_gg_ramp_down'][p][t]))

    ###################################
    ######## POWER BALANCE
    ###################################
    # Add constraints for power output
    for p in range(len(inputs['river_nodes'])):
        for d in range(days_per_week):
            start_hour = d * hours_per_day
            end_hour = start_hour + hours_per_day
            for t in range(start_hour, end_hour):
                constraint_id_main.append('Power Output')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['p_tur'][p][t] == variables['q_tur'][p][t] * parameters['WaterToPowerConversion'][inputs['river_nodes'][p],f'D{d + 1:02}']))

    # Add constraints for power balance
    for n in range(len(inputs['nodes'])):
        node = inputs['nodes'][n]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            constraint_id_main.append('Power Demand Balance')
            constraint_id_unit.append(f'{inputs['nodes'][n]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(parameters['Demand'][node,time] + variables['slack_p_node_min'][n][t] - variables['slack_p_node_max'][n][t] - sum(variables['p_tur'][p][t] * parameters['Plants_to_nodes'][inputs['river_nodes'][p],node] for p in range(len(inputs['river_nodes']))) >= 0))

    ###################################
    ######## TEMPERATURE
    ###################################

    for j in range(0, 4):
        for p in range(len(inputs['river_nodes'])):
            plant = inputs['river_nodes'][p]
            for t in range(len(inputs['time_horizon'])):
                time = inputs['time_horizon'][t]
                constraint_id_main.append('Hull constraint: Surface' + str(j) + ' for ' + gauges_hulls[plant][time].label)
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                if gauges_hulls[plant][time].inequality_signs[j] == '<=':
                    constraints.append(
                        model.add_linear_constraint(gauges_hulls[plant][time].plane_equations[j][0] * variables['q_gg'][p][t] +
                                   gauges_hulls[plant][time].plane_equations[j][1] * variables['T_gg'][p][t] +
                                   gauges_hulls[plant][time].plane_equations[j][2] * variables['X'][p][t] <=
                                   gauges_hulls[plant][time].plane_equations[j][3]))
                if gauges_hulls[plant][time].inequality_signs[j] == '>=':
                    constraints.append(
                        model.add_linear_constraint(gauges_hulls[plant][time].plane_equations[j][0] * variables['q_gg'][p][t] +
                                   gauges_hulls[plant][time].plane_equations[j][1] * variables['T_gg'][p][t] +
                                   gauges_hulls[plant][time].plane_equations[j][2] * variables['X'][p][t] >=
                                   gauges_hulls[plant][time].plane_equations[j][3]))

    # Add constraint for maximum hourly temperature limit on the outflow side
    for p in range(len(inputs['river_nodes'])):
        plant = inputs['river_nodes'][p]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            constraint_id_main.append('Maximum hourly temperature (OutFlow)')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['X'][p][t] <= variables['q_sink'][p][t]*parameters['T_sink_hourly_max'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_max'][inputs['river_nodes'][j],time]-parameters['dT_hourly'][inputs['river_nodes'][p],inputs['river_nodes'][j],time])*variables['q_flow'][p][j][t] for j in range(len(inputs['river_nodes']))),
                                                               name = f'Maximum hourly temperature (OutFlow) for {inputs['river_nodes'][p]} at {inputs['time_horizon'][t]}'
                                                               ))

    # Add constraint for minimum hourly temperature limit on the outflow side
    for p in range(len(inputs['river_nodes'])):
        plant = inputs['river_nodes'][p]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            constraint_id_main.append('Minimum hourly temperature (OutFlow)')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            constraints.append(model.add_linear_constraint(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['X'][p][t] >= variables['q_sink'][p][t]*parameters['T_sink_hourly_min'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_min'][inputs['river_nodes'][j],time]-parameters['dT_hourly'][inputs['river_nodes'][p],inputs['river_nodes'][j],time])*variables['q_flow'][p][j][t] for j in range(len(inputs['river_nodes']))),
                                                               name = f'Minimum hourly temperature (OutFlow) for {inputs['river_nodes'][p]} at {inputs['time_horizon'][t]}'
                                                               ))

    # Add constraint for maximum hourly temperature limit on the inflow side
    for p in range(len(inputs['river_nodes'])):
        plant = inputs['river_nodes'][p]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            constraint_id_main.append('Maximum hourly temperature (InFlow)')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            if plant in inputs['gauges']:
                constraints.append(model.add_linear_constraint(variables['X'][p][t] <= variables['q_source'][p][t]*parameters['T_source_hourly_max'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_max'][inputs['river_nodes'][j],time]+parameters['dT_hourly'][inputs['river_nodes'][j],inputs['river_nodes'][p],time])*variables['q_flow'][j][p][t] for j in range(len(inputs['river_nodes']))),
                                                               name = f'Maximum hourly temperature (InFlow) for {inputs['river_nodes'][p]} at {inputs['time_horizon'][t]}'
                                                               ))

    # Add constraint for minimum hourly temperature limit on the inflow side
    for p in range(len(inputs['river_nodes'])):
        plant = inputs['river_nodes'][p]
        for t in range(len(inputs['time_horizon'])):
            time = inputs['time_horizon'][t]
            constraint_id_main.append('Minimum hourly temperature (InFlow)')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append(f'{inputs['time_horizon'][t]}')
            if plant in inputs['gauges']:
                constraints.append(model.add_linear_constraint(variables['X'][p][t] >= variables['q_source'][p][t]*parameters['T_source_hourly_min'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_min'][inputs['river_nodes'][j],time]+parameters['dT_hourly'][inputs['river_nodes'][j],inputs['river_nodes'][p],time])*variables['q_flow'][j][p][t] for j in range(len(inputs['river_nodes']))),
                                                               name=f'Minimum hourly temperature (InFlow) for {inputs['river_nodes'][p]} at {inputs['time_horizon'][t]}'
                                                               ))

    # Add constraint for maximum daily temperature limit on the outflow side
    for p in range(len(inputs['river_nodes'])):
        for d in range(days_per_week):
            start_hour = d * hours_per_day
            end_hour = start_hour + hours_per_day
            plant = inputs['river_nodes'][p]
            constraint_id_main.append('Maximum daily average temperature (OutFlow)')
            constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
            constraint_id_time.append('')
            constraints.append(model.add_linear_constraint(sum(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['X'][p][t] for t in range(start_hour,end_hour)) <= parameters['T_gg_daily'][inputs['river_nodes'][p],f'D{d + 1:02}'] * sum(variables['q_sink'][p][t] for t in range(start_hour,end_hour)) + sum((parameters['T_gg_daily'][inputs['river_nodes'][j], f'D{d + 1:02}'] - parameters['dT_daily'][inputs['river_nodes'][p],inputs['river_nodes'][j]][d])*sum(variables['q_flow'][p][j][t] for t in range(start_hour,end_hour)) for j in range(len(inputs['river_nodes']))),
                                                               name = f'Maximum daily average temperature (OutFlow) for {inputs['river_nodes'][p]} at {f'D{d + 1:02}'}'
                                                               ))
    time_limit = False
    if inputs['sim_time'] >= 0:
        import datetime
        time_delta=datetime.timedelta(seconds=inputs['sim_time'])
        params = mathopt.SolveParameters(enable_output=True, time_limit=time_delta)
    else:
        params = mathopt.SolveParameters(enable_output=True)
    if inputs['sim_type'] == 'MILP':
        result = mathopt.solve(model, mathopt.SolverType.HIGHS, params=params)
        duals = None
        assert result.termination.reason in {mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE}
    else:
        result = mathopt.solve(model, mathopt.SolverType.GLOP, params=params)
        duals = result.dual_values()
        assert result.termination.reason in {mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE}

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

    status = result.termination.reason
    # Log solver status
    if status == result.termination.reason.OPTIMAL:
        logger.info("Optimization successful: Found optimal solution.")
    elif status == result.termination.reason.FEASIBLE:
        logger.info("Optimization terminated: Found feasible solution.")
    elif status == result.termination.reason.INFEASIBLE:
        logger.warning("Optimization terminated: Problem is infeasible.")
    elif status == result.termination.reason.NO_SOLUTION_FOUND:
        logger.warning("Optimization not solved: No solution found.")
    else:
        logger.error("Optimization failed: Solver encountered an error.")

    try:
        if (status == result.termination.reason.OPTIMAL) or (status == result.termination.reason.FEASIBLE):
            results = {}
            for var in ['Obj func']:
                results[var] = pd.DataFrame([result.objective_value()])
            # for var in ['q_tot_min']:
            #     results[var] = pd.DataFrame(result.variable_values(variables[var]),index=inputs['river_nodes'])/cfs_to_af

            for var in ['q_tot_min']:
                results[var] = pd.DataFrame(columns=inputs['river_nodes'], index=inputs['daily_horizon'])
                for p in range(len(inputs['river_nodes'])):
                    plant = inputs['river_nodes'][p]
                    for d in range(len(inputs['daily_horizon'])):
                        day = inputs['daily_horizon'][d]
                        results[var].loc[day,plant] = result.variable_values(variables[var][p][d])/CFS_TO_AFH

            for var in ['q_bp', 'q_tur', 'q_tot', 'q_gg', 'q_source', 'q_sink', 'q_in_flow', 'q_out_flow', 'X',
                        'q_bp_ramp_up', 'q_bp_ramp_down', 'q_tur_ramp_up', 'q_tur_ramp_down', 'q_tot_ramp_up',
                        'q_tot_ramp_down', 'q_gg_ramp_up', 'q_gg_ramp_down']:
                results[var] = {}
                for p in range(len(inputs['river_nodes'])):
                    results[var][inputs['river_nodes'][p]] = result.variable_values(variables[var][p])
                results[var] = pd.DataFrame(results[var], index=inputs['time_horizon'])/CFS_TO_AFH
            for var in ['p_tur', 'T_gg']:
                results[var] = {}
                for p in range(len(inputs['river_nodes'])):
                    results[var][inputs['river_nodes'][p]] = result.variable_values(variables[var][p])
                results[var] = pd.DataFrame(results[var], index=inputs['time_horizon'])
            if inputs['sim_type'] == 'MILP':
                for var in ['q_bp_rr_up_change', 'q_bp_rr_down_change', 'q_bp_committed']:
                    results[var] = {}
                    for p in range(len(inputs['river_nodes'])):
                        results[var][inputs['river_nodes'][p]] = result.variable_values(variables[var][p])
                    results[var] = pd.DataFrame(results[var], index=inputs['time_horizon'])
            for var in ['q_flow']:
                results[var] = {}
                for i in range(len(inputs['river_nodes'])):
                    for j in range(len(inputs['river_nodes'])):
                        results[var][inputs['river_nodes'][i],inputs['river_nodes'][j]] = result.variable_values(variables[var][i][j])
                results[var] = {f"{k[0]} - {k[1]}": v for k, v in results[var].items()}
                results[var] = pd.DataFrame(results[var], index=inputs['time_horizon'])/CFS_TO_AFH
            results['T_plant'] = (results['q_bp']*inputs['T_bp_values'].T + results['q_tur']*inputs['T_tur_values'].T)/(results['q_bp'] + results['q_tur'])

            def compute_gauge_temperatures(t_plant_df, q_flow_df, dt_hourly_values_df):
                t_results = pd.DataFrame(index=t_plant_df.index, columns=inputs['gauges'])

                for time_interval in t_plant_df.index:
                    t_plant = t_plant_df.loc[time_interval].dropna().to_dict()
                    t_gauge = {gauge: None for gauge in inputs['gauges']}

                    for col in q_flow_df.columns:
                        from_node, to_node = col.split(' - ')
                        if from_node in t_plant:
                            t_gauge[to_node] = t_plant[from_node] + dt_hourly_values_df.at['GCD - RM61', time_interval]

                    for gauge in inputs['gauges']:
                        t_results.at[time_interval, gauge] = t_gauge[gauge]

                return t_results

            t_gauge = compute_gauge_temperatures(results['T_plant'], results['q_flow'], inputs['dT_hourly_values'])
            t_gauge = t_gauge.astype('float64')
            results['T_plant'].astype('float64')
            results['T_plant'].update(t_gauge)
            results['T_avg_daily'] = results['T_plant'].mean()

            # Slack variables
            for var in ['slack_q_bp_cyc_min', 'slack_q_bp_cyc_max', 'slack_q_tur_cyc_min', 'slack_q_tur_cyc_max', 'slack_q_tot_cyc_min', 'slack_q_tot_cyc_max',
                        'slack_q_gg_cyc_min', 'slack_q_gg_cyc_max']:
                results[var] = pd.DataFrame(result.variable_values(variables[var]),index=inputs['river_nodes'])

            # Slack variables
            for var in ['slack_q_tot_daily_min', 'slack_q_tot_daily_max']:
                results[var] = pd.DataFrame(index=inputs['daily_horizon'], columns=inputs['river_nodes'])
                for p in range(len(inputs['river_nodes'])):
                    plant = inputs['river_nodes'][p]
                    results[var][plant] = result.variable_values(variables[var][p])


            for var in ['slack_p_node_min', 'slack_p_node_max']:
                results[var] = {}
                for p in range(len(inputs['nodes'])):
                    results[var][inputs['nodes'][p]] = result.variable_values(variables[var][p])
                results[var] = pd.DataFrame(results[var], index=inputs['time_horizon'])

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
                    value.to_csv(f'{simulation_folder_path + "/"}{key}.csv', index=False)
                else:
                    with open(f'{simulation_folder_path + "/"}{key}.csv', 'w') as f:
                        f.write(str(value))

            INPUTS_TO_SAVE = [
                "LMP_values",
                "Demand_values",
                "T_bp_values",
                "T_tur_values",
                "cnstr_temp_t01t24",
                "T_gauge_hourly_max_values",
                "T_gauge_hourly_min_values",
                "dT_hourly_values"
            ]

            for key in INPUTS_TO_SAVE:
                the_input = inputs[key]
                value.to_csv(f'{simulation_folder_path + "/INPUT_"}{key}.csv', index=False)

            def get_lmp_values_for_plant(plant_name, plant_to_node, lmp_values):
                # Generate the plant_node_mapping from plant_to_node
                plant_node_mapping = {entry.split(' - ')[0]: entry.split(' - ')[1] for entry in plant_to_node}

                # Get the node name for the given plant
                node_name = plant_node_mapping.get(plant_name)
                if node_name:
                    return lmp_values.loc[node_name]
                else:
                    raise ValueError(f"Plant {plant_name} not found in the mapping.")

            if inputs['plt_show']:
                for unit_name in inputs['plants']:
                    plot_data = pd.concat([results['q_bp'].loc[:, unit_name], results['q_tur'].loc[:, unit_name]], axis=1).set_axis(['q_bp', 'q_tur'], axis=1).clip(lower=0)
                    plot_data['LMP'] = get_lmp_values_for_plant(unit_name, inputs['plant_to_node'], inputs['LMP_values'])
                    plot_dispatch(plot_data[['q_bp','q_tur']], plot_data[['LMP']], save_fig=True, fig_name=simulation_folder_path + '/plot', unit_name=unit_name)
                    df_dispatch = inputs['Demand_values'].copy().T
                    df_dispatch = pd.concat([df_dispatch, results['p_tur']], axis=1)
                    plot_load_serving(df_dispatch,save_fig=True, fig_name=simulation_folder_path + '/pload_serving', unit_name=unit_name, node_name='Node 1')

        else:
            logger.error('The problem does not have an optimal solution. Please review and revise the input data.')

        copy_file_to_destination('TempRegPy/app.log', simulation_folder_path)

        return inputs, results, duals

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        logger.error(f"An unexpected error occurred: {e}. The problem does not have an optimal solution. Please review "
                     f"and revise the input data.")
