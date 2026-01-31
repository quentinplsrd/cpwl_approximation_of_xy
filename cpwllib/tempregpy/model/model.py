import datetime
import logging
from tqdm import tqdm
from ortools.math_opt.python import mathopt
from .xy_plot_and_constraints import MILP_or_QP_variables_and_constraints
from .product_linearization import ProductLinearizationConfig, do_mathopt_product_linearization
from dataclasses import dataclass
import enum
from ortools.math_opt.python import result

class Methods(enum.Enum):
    TRIANGLES = 'triangles'
    POLYGONS = 'polygons'
    SUM_OF_CONVEX = 'sum of convex'
    QUADRATIC = "quadratic"

@dataclass
class ModelConfig:
    name: str
    solver_type: mathopt.SolverType
    bilinear_method: Methods = Methods.TRIANGLES
    target_error: float = 0.05
    # product_cpwl_method: ProductCPWLMethod = ProductCPWLMethod.SINGLE
    # product_linearization: ProductLinearizationConfig 



class Model():
    """
    Base class for a model
    """
    def __init__(self, config: ModelConfig):
        self.model_name = config.name
        self.logger = logging.getLogger(f"Model.{self.model_name}")
        self.parameters = {}
        self.variables = {}
        self.config = config

        pass

    def populate_from_inputs(self, inputs):
        """
        Populate the model with inputs
        """
        logger = self.logger

        #####################
        ###### CONVEX HULLS
        #####################
        # gauges_hulls = {}
        # for rn in inputs['river_nodes']:
        #     gauges_hulls[rn] = {}
        #     for i in inputs['time_horizon']:
        #         gauges_hulls[rn][i] = gauge_hulls_time(inputs, rn, i)

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
        parameters['Q_rough_zone_min'] = {(p, r): value for p, row in inputs['Q_rough_zone_min'].T.iterrows() for r, value
                                          in row.items()}
        parameters['Q_rough_zone_max'] = {(p, r): value for p, row in inputs['Q_rough_zone_max'].T.iterrows() for r, value
                                          in row.items()}

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

        model = mathopt.Model(name=self.model_name)

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
        variables['QT_latent'] = [[model.add_variable(lb=0,
                                            ub=100000,
                                            name=f'QT_latent_{p}_{t}') for t in inputs['time_horizon']] for p in inputs['river_nodes']
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
            if inputs['cnstr_rough_zone']:
                variables['q_tur_committed'] = [[[model.add_integer_variable(lb=0, ub=1,
                                                 name=f'q_tur_committed_{p}_{r}_{t}') for t in inputs['time_horizon']]
                                                 for r in inputs['rough_zones']]
                                                 for p in inputs['river_nodes']
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
                obj_fnc += - (variables['slack_q_tot_daily_min'][p][d] + variables['slack_q_tot_daily_max'][p][d]) * inputs['C_slack']

            obj_fnc += - (variables['slack_q_bp_cyc_min'][p] + variables['slack_q_bp_cyc_max'][p] +
                        variables['slack_q_tur_cyc_min'][p] + variables['slack_q_tur_cyc_max'][p] +
                        variables['slack_q_tot_cyc_min'][p] + variables['slack_q_tot_cyc_max'][p] +
                        variables['slack_q_gg_cyc_min'][p] + variables['slack_q_gg_cyc_max'][p]) * inputs['C_slack']

        for t in range(len(inputs['time_horizon'])):
            obj_fnc += (- sum((variables['slack_p_node_min'][n][t] + variables['slack_p_node_max'][n][t]) * inputs['C_slack'] for n in range(len(inputs['nodes']))))

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
        ######## ROUGH ZONES
        ###################################
        if inputs['cnstr_rough_zone']:
            for p in range(len(inputs['plants'])):
                for t in range(len(inputs['time_horizon'])):
                    plant = inputs['plants'][p]
                    constraint_id_main.append('Rough zone min constraint')
                    constraint_id_unit.append(f'{inputs['plants'][p]}')
                    constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                    constraints.append(model.add_linear_constraint(
                        sum(variables['q_tur_committed'][p][r][t] * parameters['Q_rough_zone_min'][plant, inputs['rough_zones'][r]] for r in range(len(inputs['rough_zones']))) <= variables['q_tur'][p][t]))

            for p in range(len(inputs['plants'])):
                for t in range(len(inputs['time_horizon'])):
                    plant = inputs['plants'][p]
                    constraint_id_main.append('Rough zone max constraint')
                    constraint_id_unit.append(f'{inputs['plants'][p]}')
                    constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                    constraints.append(model.add_linear_constraint(
                        sum(variables['q_tur_committed'][p][r][t] * parameters['Q_rough_zone_max'][plant, inputs['rough_zones'][r]] for r in range(len(inputs['rough_zones']))) >= variables['q_tur'][p][t]))


            for p in range(len(inputs['plants'])):
                for t in range(len(inputs['time_horizon'])):
                    constraint_id_main.append('Sum of binary <=1')
                    constraint_id_unit.append(f'{inputs['plants'][p]}')
                    constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                    constraints.append(model.add_linear_constraint(sum(variables['q_tur_committed'][p][r][t]
                                                                       for r in range(len(inputs['rough_zones']))) <= 1))
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

        for p in range(len(inputs['river_nodes'])):
            for d in range(days_per_week):
                constraint_id_main.append('Total daily release')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'D{d + 1:02}')

                start_hour = d * hours_per_day
                end_hour = start_hour + hours_per_day

                constraints.append(
                    model.add_linear_constraint(
                        sum(variables['q_tot'][p][t] for t in range(start_hour, end_hour))
                        # + variables['slack_q_tot_daily_min'][p][d] 
                        == parameters['Q_tot_daily'][(inputs['river_nodes'][p], f'D{d + 1:02}')] 
                        # + variables['slack_q_tot_daily_max'][p][d]
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
        target_error = self.config.target_error
        if self.config.bilinear_method == Methods.POLYGONS:
            MILP_or_QP_variables_and_constraints(model,
                                                variables['q_gg'],
                                                variables['T_gg'],
                                                quadratic=False,
                                                target_error=target_error,
                                                partition_method='polygons',
                                                logarithmic_encoding=False)
        elif self.config.bilinear_method == Methods.TRIANGLES:
            MILP_or_QP_variables_and_constraints(model,
                                                variables['q_gg'],
                                                variables['T_gg'],
                                                quadratic=False,
                                                target_error=target_error,
                                                partition_method='triangles',
                                                logarithmic_encoding=False)
        elif self.config.bilinear_method == Methods.SUM_OF_CONVEX:
            MILP_or_QP_variables_and_constraints(model,
                                                variables['q_gg'],
                                                variables['T_gg'],
                                                quadratic=False,
                                                target_error=target_error,
                                                partition_method='sum of convex',
                                                logarithmic_encoding=False)
        elif self.config.bilinear_method == Methods.QUADRATIC:
            MILP_or_QP_variables_and_constraints(model,
                                                variables['q_gg'],
                                                variables['T_gg'],
                                                quadratic=True,
                                                target_error=target_error,
                                                partition_method='sum of convex',
                                                logarithmic_encoding=False)
        else:
            raise ValueError(f"Bilinear method {self.config.bilinear_method} not recognized.")

        # do_mathopt_product_linearization(
        #     method=self.config.product_linearization.method,
        #     model=model,
        #     X_var=variables["q_gg"],
        #     Y_var=variables["T_gg"],
        #     Z_var=variables["QT_latent"],
        # )

        # Add constraint for maximum hourly temperature limit on the outflow side
        for p in range(len(inputs['river_nodes'])):
            plant = inputs['river_nodes'][p]
            for t in range(len(inputs['time_horizon'])):
                time = inputs['time_horizon'][t]
                constraint_id_main.append('Maximum hourly temperature (OutFlow)')
                constraint_id_unit.append(f'{inputs['river_nodes'][p]}')
                constraint_id_time.append(f'{inputs['time_horizon'][t]}')
                constraints.append(model.add_linear_constraint(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['QT_latent'][p][t] <= variables['q_sink'][p][t]*parameters['T_sink_hourly_max'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_max'][inputs['river_nodes'][j],time]-parameters['dT_hourly'][inputs['river_nodes'][p],inputs['river_nodes'][j],time])*variables['q_flow'][p][j][t] for j in range(len(inputs['river_nodes']))),
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
                constraints.append(model.add_linear_constraint(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['QT_latent'][p][t] >= variables['q_sink'][p][t]*parameters['T_sink_hourly_min'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_min'][inputs['river_nodes'][j],time]-parameters['dT_hourly'][inputs['river_nodes'][p],inputs['river_nodes'][j],time])*variables['q_flow'][p][j][t] for j in range(len(inputs['river_nodes']))),
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
                    constraints.append(model.add_linear_constraint(variables['QT_latent'][p][t] <= variables['q_source'][p][t]*parameters['T_source_hourly_max'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_max'][inputs['river_nodes'][j],time]+parameters['dT_hourly'][inputs['river_nodes'][j],inputs['river_nodes'][p],time])*variables['q_flow'][j][p][t] for j in range(len(inputs['river_nodes']))),
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
                    constraints.append(model.add_linear_constraint(variables['QT_latent'][p][t] >= variables['q_source'][p][t]*parameters['T_source_hourly_min'][inputs['river_nodes'][p],time] + sum((parameters['T_gg_hourly_min'][inputs['river_nodes'][j],time]+parameters['dT_hourly'][inputs['river_nodes'][j],inputs['river_nodes'][p],time])*variables['q_flow'][j][p][t] for j in range(len(inputs['river_nodes']))),
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
                constraints.append(model.add_linear_constraint(sum(variables['q_tur'][p][t]*parameters['T_tur'][plant,time] + variables['q_bp'][p][t]*parameters['T_bp'][plant,time] + variables['QT_latent'][p][t] for t in range(start_hour,end_hour)) <= parameters['T_gg_daily'][inputs['river_nodes'][p],f'D{d + 1:02}'] * sum(variables['q_sink'][p][t] for t in range(start_hour,end_hour)) + sum((parameters['T_gg_daily'][inputs['river_nodes'][j], f'D{d + 1:02}'] - parameters['dT_daily'][inputs['river_nodes'][p],inputs['river_nodes'][j]][d])*sum(variables['q_flow'][p][j][t] for t in range(start_hour,end_hour)) for j in range(len(inputs['river_nodes']))),
                                                                name = f'Maximum daily average temperature (OutFlow) for {inputs['river_nodes'][p]} at {f'D{d + 1:02}'}'
                                                                ))
        # if inputs['cnstr_pattern_repeat']!='Off':
        #     # CONSTRAINT | Constrain group days to have the same hourly values for "q_bp"
        #     visited_groups = {}
        #     # variables['slack_day_hour_node'] = {}
        #     for d in range(days_per_week):
        #         group = inputs['day_groups'].iloc[d]['day_group']
        #         if not group in visited_groups:
        #             visited_groups[group] = d
        #             continue
        #         else:
        #             d_constrain_to = visited_groups[group]
        #             source_start_hour = d_constrain_to * hours_per_day
        #             target_start_hour = d * hours_per_day
        #             # for h in range(hours_per_day):
        #             for p in range(len(inputs['river_nodes'])):
        #                 if inputs['cnstr_pattern_repeat'] == 'Bypass Release':
        #                     target_vars = variables['q_bp'][p][target_start_hour:target_start_hour + hours_per_day]
        #                     source_vars = variables['q_bp'][p][source_start_hour:source_start_hour + hours_per_day]
        #                 if inputs['cnstr_pattern_repeat'] == 'Turbine Release':
        #                     target_vars = variables['q_tur'][p][target_start_hour:target_start_hour + hours_per_day]
        #                     source_vars = variables['q_tur'][p][source_start_hour:source_start_hour + hours_per_day]
        #                 if inputs['cnstr_pattern_repeat'] == 'Total Release':
        #                     target_vars = variables['q_tot'][p][target_start_hour:target_start_hour + hours_per_day]
        #                     source_vars = variables['q_tot'][p][source_start_hour:source_start_hour + hours_per_day]

        #                 h = 1
        #                 for target_var, source_var in zip(target_vars, source_vars):
        #                     # slack_var_name = f'slack_day_{d}_hour_{h}_node_{inputs["river_nodes"][p]}'
        #                     # slack_var = model.add_variable(name=slack_var_name, lb=0)
        #                     # variables['slack_day_hour_node'][(d, h, inputs["river_nodes"][p])] = slack_var

        #                     constraint_id_main.append('Constrain day group to same values')
        #                     constraint_id_unit.append(f'{inputs["river_nodes"][p]}')
        #                     constraint_id_time.append(f'D{d + 1:02}')
        #                     # constraints.append(model.add_linear_constraint(target_var == source_var + slack_var,
        #                     #                                                name=f'Constrain day {d} to day {d_constrain_to} at hour {h} for river node {inputs["river_nodes"][p]}'))
        #                     constraints.append(model.add_linear_constraint(target_var == source_var,
        #                                                                    name=f'Constrain day {d} to day {d_constrain_to} at hour {h} for river node {inputs["river_nodes"][p]}'))

        #                     # obj_fnc += slack_var * 100000
        #                     h += 1
        #                     # Uncomment the next line for debugging purposes
        #                 # logger.debug(f'Constrain day {d} to day {d_constrain_to} for river node {inputs["river_nodes"][p]}')
        #     del visited_groups


        model.maximize(obj_fnc)


        self.variables = variables
        self.constraints = constraints
        self.constraint_id_main = constraint_id_main
        self.constraint_id_unit = constraint_id_unit
        self.constraint_id_time = constraint_id_time
        self.parameters = parameters
        self.mathopt_model = model

    def solve(self, time_limit=10, optimality_gap=0.01) -> result.SolveResult:
        time_limit = datetime.timedelta(seconds=time_limit)
        params = mathopt.SolveParameters(
            enable_output=True, 
            time_limit=time_limit,
            relative_gap_tolerance=optimality_gap,
        )

        return mathopt.solve(
            opt_model=self.mathopt_model, 
            solver_type=self.config.solver_type,
            params=params
        )
    