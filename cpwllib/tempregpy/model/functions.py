import sys
from ..logging_config import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from .constants import *





class DataGenerator:
    def __init__(self, Q_tur_min, Q_tur_max, Q_bp_min, Q_bp_max, Q_tot_min, Q_tot_max):
        self.Q_tur_min = Q_tur_min
        self.Q_tur_max = Q_tur_max
        self.Q_bp_min = Q_bp_min
        self.Q_bp_max = Q_bp_max
        self.Q_tot_min = Q_tot_min
        self.Q_tot_max = Q_tot_max
        self.generate_data()

    def generate_data(self):
        x = np.linspace(self.Q_tur_min, self.Q_tur_max, 4, endpoint=True)
        y = np.linspace(self.Q_bp_min, self.Q_bp_max, 4, endpoint=True)
        z = np.linspace(self.Q_tot_min, self.Q_tot_max, 4, endpoint=True)
        Q_bp, Q_tur1 = np.meshgrid(y, x, indexing='ij')
        Q_tur, Q_bp1 = np.meshgrid(x, y, indexing='ij')
        Q_tot, Q_tot1 = np.meshgrid(z, z, indexing='ij')

        self.points_X = np.c_[Q_bp.flatten(), Q_tur1.flatten(), (Q_bp * Q_tur1).flatten()]
        self.points_Y = np.c_[Q_tur.flatten(), Q_bp1.flatten(), (Q_tur * Q_bp1).flatten()]
        self.points_Z = np.c_[Q_tot.flatten(), Q_tot1.flatten(), (Q_tot * Q_tot1).flatten()]

        self.Q_bp_mesh, self.Q_tur_mesh = np.meshgrid(y, x)
        self.X_mesh = self.Q_bp_mesh * self.Q_tur_mesh
        self.Q_tot_mesh, self.Q_tot1_mesh = np.meshgrid(z, z)
        self.Z_mesh = self.Q_tot_mesh * self.Q_tot1_mesh


class Plotter:
    def __init__(self, data_gen, hull_X, hull_Y, hull_Z):
        self.data_gen = data_gen
        self.hull_X = hull_X
        self.hull_Y = hull_Y
        self.hull_Z = hull_Z
        self.create_plots()

    def create_plots(self):
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')
        ax3 = fig.add_subplot(gs[1, 0], projection='3d')
        ax4 = fig.add_subplot(gs[1, 1], projection='3d')

        self.plot_scatter_and_hull(ax3, self.data_gen.points_X, self.hull_X, '$Q_{bp,t}$', '$Q_{tur,t-1}$', '$Q_{bp,t} \\cdot Q_{tur,t-1}$', 'Scatter plot with Convex Hull for $X$')
        self.plot_scatter_and_hull(ax4, self.data_gen.points_Z, self.hull_Z, '$Q_{tot,t}$', '$Q_{tot,t-1}$', '$Q_{tot,t} \\cdot Q_{tot,t-1}$', 'Scatter plot with Convex Hull for $Z$')
        self.plot_surface(ax1, self.data_gen.Q_bp_mesh, self.data_gen.Q_tur_mesh, self.data_gen.X_mesh, '$Q_{bp,t}$', '$Q_{tur,t-1}$', '$Q_{bp,t} \\cdot Q_{tur,t-1}$', 'Surface plot for $X$')
        self.plot_surface(ax2, self.data_gen.Q_tot_mesh, self.data_gen.Q_tot1_mesh, self.data_gen.Z_mesh, '$Q_{tot,t}$', '$Q_{tot,t-1}$', '$Q_{tot,t} \\cdot Q_{tot,t-1}$', 'Surface plot for $Z$')

        plt.tight_layout()
        # plt.savefig('Qbp-Qtur-T.png', dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_scatter_and_hull(self, ax, points, hull, xlabel, ylabel, zlabel, title):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o', label='Data points')
        for simplex in hull.hull.simplices:
            simplex_points = points[simplex]
            ax.plot_trisurf(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], color='r', alpha=0.3, label='Convex Hull')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.invert_xaxis()

    def plot_surface(self, ax, X, Y, Z, xlabel, ylabel, zlabel, title):
        ax.plot_surface(X, Y, Z, cmap='viridis', label='Surface')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        ax.invert_xaxis()


def process_double_index(variable, idx1, idx2, input=False):
    if input is False:
        data = {(t, u): variable[t, u].solution_value() for t in idx1 for u in idx2}
    else:
        data = {(t, u): variable[t, u] for t in idx1 for u in idx2}
    # Extract unit names and column names
    unit_names = sorted(set(unit for (_, unit) in data.keys()))
    column_names = sorted(set(column for (column, _) in data.keys()))

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data.values(), index=pd.MultiIndex.from_tuples(data.keys()), columns=['Values'])

    # Reshape DataFrame to desired format
    df = df.unstack().squeeze().to_frame().T

    # Set index and columns
    df.index = unit_names
    df.columns = column_names
    return df

def process_single_index(variable, idx, input=False, col_name=''):
    if input is False:
        data = {(t): variable[t].solution_value() for t in idx}
    else:
        data = {(t): variable[t] for t in idx}
    # Extract unit names and column names
    index_names = sorted(set(column for (column) in data.keys()))

    # Convert dictionary to DataFrame
    df = pd.DataFrame(data.values(), columns=[col_name])

    # Set index and columns
    df.index = index_names
    return df

def export_dual_values(constraints, constraint_id_main, constraint_id_unit, constraint_id_time):
    dual_values = pd.DataFrame({
        'Constraint Name': constraint_id_main,
        'Constraint idx1': constraint_id_unit,
        'Constraint idx2': constraint_id_time,
        'Dual Value': [constraint.dual_value() for constraint in constraints]
    })
    return dual_values


def initialize_inputs(keys, config, par_name, column, inputs, default_value):
    for key in keys:
        if key in config[par_name].values:
            inputs[key] = config.loc[config[par_name] == key, column].dropna(axis=1).values[0]
        else:
            inputs[key] = default_value

def initialize_inputs_with_index(keys, config, par_name, columns, inputs, index_key, default_value):
    for key in keys:
        if key in config[par_name].values:
            inputs[key] = config.loc[config[par_name] == key, columns].set_index('Name').reindex(inputs[index_key]).infer_objects(copy=False).fillna(default_value)
        else:
            log_error(key)

def initialize_inputs_with_conversion(keys_with_conversion, config, par_name, columns, inputs, index_key, default_value=0):
    for key, factor in keys_with_conversion.items():
        if key in config[par_name].values:
            inputs[key] = config.loc[config[par_name] == key, columns].set_index('Name').reindex(inputs[index_key]).infer_objects(copy=False).fillna(default_value) * factor
        else:
            log_error(key)
            inputs[key] = pd.DataFrame()

def log_error(key):
    logger.error(f'There is no such key {key}')
    pass


def load_config(file_path):
    full_path = os.path.abspath(file_path)

    # Log the file name and full path
    logger.info(f'Reading Excel file: {os.path.basename(file_path)}')
    logger.info(f'Full path: {full_path}')

    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(file_path)
        logger.info('File has been successfully read.')
        # Parse the relevant sheet
        try:
            config = excel_data.parse('Inputs')
            logger.info('Inputs have been succesfully parsed')
            par_name = 'Python Variable Names'
            input_type = 'Single input'

            # Initialize the input_data dictionary
            inputs = {}
            inputs['version'] = config.loc[config[par_name] == 'version', input_type].values[0]
            inputs['file_path'] = file_path

            # Hourly values
            inputs['time_horizon'] = [f'H{i:02}' for i in range(1, 169)]
            inputs['daily_horizon'] = [f'H{i:02}' for i in range(1, 8)]

            # Define list of plants
            plant_keys = ['plants', 'gauges', 'nodes', 'river_topology', 'plant_to_node', 'sinks', 'sources']
            initialize_inputs(plant_keys, config, par_name, inputs['time_horizon'], inputs, [])

            # Create incidence matrix for Plants and Gauges
            inputs['river_nodes'] = np.concatenate((inputs['plants'], inputs['gauges']))

            # Process constraint related inputs
            constraint_keys = [
                'sim_folder', 'plt_show', 'cnstr_bp_ramp', 'cnstr_tur_ramp', 'cnstr_tot_ramp', 'cnstr_gg_ramp',
                'cnstr_temp_t01t24', 'cnstr_bp_t01t24', 'cnstr_tur_t01t24', 'cnstr_tot_t01t24', 'cnstr_gg_t01t24',
                'cnstr_oper_range', 'sim_type', 'sim_time'
            ]
            # Loop through each key and extract the corresponding value
            for key in constraint_keys:
                inputs[key] = config.loc[config[par_name] == key, input_type].values[0]

            ##################################
            # Process plant related input data
            ##################################
            plant_input_keys = ['Q_tot_daily']
            if inputs['version'] == 'v0.1.2':
                initialize_inputs_with_index(plant_input_keys, config, par_name, ['Name', input_type], inputs, 'river_nodes', 0)
            else:
                initialize_inputs_with_index(plant_input_keys, config, par_name, ['Name']+inputs['daily_horizon'], inputs, 'river_nodes', 0)

            plant_input_keys_with_conversion = {
                'Q_max_daily_change': CFS_TO_AFH,
                'WaterToPowerConversion': 1 / CFS_TO_AFH,
            }
            initialize_inputs_with_conversion(plant_input_keys_with_conversion, config, par_name, ['Name']+inputs['daily_horizon'], inputs, 'river_nodes')

            plant_keys_with_conversion = {
                'Q_bp_ramp_down_max': CFS_TO_AFH,
                'Q_bp_ramp_up_max': CFS_TO_AFH,
                'Q_tur_ramp_down_max': CFS_TO_AFH,
                'Q_tur_ramp_up_max': CFS_TO_AFH,
                'Q_tot_ramp_up_max': CFS_TO_AFH,
                'Q_tot_ramp_down_max': CFS_TO_AFH,
                'Q_gg_ramp_up_max': CFS_TO_AFH,
                'Q_gg_ramp_down_max': CFS_TO_AFH,
                'bp_ramp_down_cyc': 1,
                'bp_ramp_up_cyc': 1,
                'bp_ramp_tot_cyc': 1,
            }
            initialize_inputs_with_conversion(plant_keys_with_conversion, config, par_name, ['Name', input_type],
                                              inputs, 'river_nodes')
            # Hourly values
            plant_hourly_keys = {
                'Q_bp_min_values': CFS_TO_AFH,
                'Q_bp_max_values': CFS_TO_AFH,
                'Q_tur_min_values': CFS_TO_AFH,
                'Q_tur_max_values': CFS_TO_AFH,
                'Q_tot_min_values': CFS_TO_AFH,
                'Q_tot_max_values': CFS_TO_AFH,
                'Q_gg_min_values': CFS_TO_AFH,
                'Q_gg_max_values': CFS_TO_AFH,
                'T_bp_values': 1,
                'T_tur_values': 1,
                'P_tur_min_values': 1,
                'P_tur_max_values': 1,
                'bp_forced_commit': 1,
            }
            initialize_inputs_with_conversion(plant_hourly_keys, config, par_name, ['Name']+inputs['time_horizon'], inputs, 'river_nodes')

            river_connection_hourly_keys = {
                'Flow_i_j_max': CFS_TO_AFH,
                'Flow_i_j_min': CFS_TO_AFH,
                'T_source_hourly_max_values': 1,
                'T_source_hourly_min_values': 1,
                'T_sink_hourly_max_values': 1,
                'T_sink_hourly_min_values': 1,
            }
            initialize_inputs_with_conversion(river_connection_hourly_keys, config, par_name, ['Name']+inputs['time_horizon'], inputs, 'river_topology')


            ##################################
            # Process gauge related input data
            ##################################
            gauge_daily_keys = ['T_gauge_daily']
            initialize_inputs_with_index(gauge_daily_keys, config, par_name, ['Name']+inputs['daily_horizon'], inputs, 'river_nodes', 0)

            gauge_keys = [
                'T_ramp_up_delta', 'T_ramp_down_delta'
            ]
            initialize_inputs_with_index(gauge_keys, config, par_name, ['Name', input_type], inputs, 'river_nodes', 0)
            gauge_hourly_min_keys = [
                'T_gauge_hourly_min_values'
            ]
            initialize_inputs_with_index(gauge_hourly_min_keys, config, par_name, ['Name']+inputs['time_horizon'], inputs, 'gauges', 0)

            gauge_hourly_max_keys = [
                'T_gauge_hourly_max_values'
            ]
            initialize_inputs_with_index(gauge_hourly_max_keys, config, par_name, ['Name'] + inputs['time_horizon'], inputs, 'gauges', 25)

            inputs['T_gauge_hourly_min_values'] = inputs['T_gauge_hourly_min_values'].reindex(inputs['river_nodes'], fill_value=0)
            inputs['T_gauge_hourly_max_values'] = inputs['T_gauge_hourly_max_values'].reindex(inputs['river_nodes'], fill_value=25)
            #################################
            # Process node related input data
            #################################
            node_keys = []
            initialize_inputs_with_index(node_keys, config, par_name, ['Name', input_type], inputs, 'nodes', 0)
            node_hourly_keys = [
                'LMP_values', 'Demand_values'
            ]
            initialize_inputs_with_index(node_hourly_keys, config, par_name, ['Name'] + inputs['time_horizon'], inputs, 'nodes', 0)

            ###########################################
            # Process plant to gauge related input data
            ###########################################
            plant_to_gauge_keys = [
                # 'dT_daily'
            ]
            initialize_inputs_with_index(plant_to_gauge_keys, config, par_name, ['Name', input_type], inputs,
                                         'river_topology', 0)
            plant_to_gauge_daily_keys = [
                'dT_daily'
            ]
            initialize_inputs_with_index(plant_to_gauge_daily_keys, config, par_name, ['Name']+inputs['daily_horizon'], inputs,
                                         'river_topology', 0)
            plant_to_gauge__hourly_keys = [
                'dT_hourly_values'
            ]
            initialize_inputs_with_index(plant_to_gauge__hourly_keys, config, par_name, ['Name']+inputs['time_horizon'], inputs,
                                         'river_topology', 0)

            ##########################################
            # Process plant to node related input data
            ##########################################
            plant_to_node_keys = []
            initialize_inputs_with_index(plant_to_node_keys, config, par_name, ['Name', input_type], inputs,
                                         'plant_to_node', 0)

            ###################
            # Process cost data
            ###################
            # Hourly values
            cost_keys = {
                'C_bp_ramp_up': 1/CFS_TO_AFH,
                'C_bp_ramp_down': 1/CFS_TO_AFH,
                'C_tur_ramp_up': 1/CFS_TO_AFH,
                'C_tur_ramp_down': 1/CFS_TO_AFH,
                'C_tot_ramp_up': 1/CFS_TO_AFH,
                'C_tot_ramp_down': 1/CFS_TO_AFH,
                'C_gg_ramp_up': 1 / CFS_TO_AFH,
                'C_gg_ramp_down': 1 / CFS_TO_AFH,
            }
            initialize_inputs_with_conversion(cost_keys, config, par_name, ['Name', input_type], inputs, 'river_nodes')

            # Create a DataFrame for the incidence matrix
            inputs['river_incidence_matrix'] = pd.DataFrame(0, index=inputs['river_nodes'], columns=inputs['river_topology'])
            inputs['river_adjecency_matrix'] = pd.DataFrame(0, index=inputs['river_nodes'], columns=inputs['river_topology'])

            inputs['plants_nodes'] = pd.DataFrame(0, index=inputs['river_nodes'], columns=inputs['nodes'])
            for connection in inputs['plant_to_node']:
                plant, node = connection.split(' - ')
                inputs['plants_nodes'].loc[plant, node] = 1

            inputs['C_source'] = pd.DataFrame(10000, index=inputs['river_nodes'], columns=['Single input'])
            inputs['C_source'].loc[inputs['sources']] = 0
            inputs['Q_source_max'] = pd.DataFrame(0, index=inputs['river_nodes'], columns=['Single input'])
            inputs['Q_source_max'].loc[inputs['sources']] = 10000
            inputs['C_sink'] = pd.DataFrame(10000, index=inputs['river_nodes'], columns=['Single input'])
            inputs['C_sink'].loc[inputs['sinks']] = 0
            inputs['Q_sink_max'] = pd.DataFrame(0, index=inputs['river_nodes'], columns=['Single input'])
            inputs['Q_sink_max'].loc[inputs['sinks']] = 10000
            inputs['Q_hulls'] = {}
            for i in inputs['time_horizon']:
                inputs['Q_hulls'][i] = pd.DataFrame(index=inputs['river_nodes'], columns=['Qmin', 'Qmax', 'Tmin', 'Tmax'])
                inputs['Q_hulls'][i].loc[:, 'Qmin'] = inputs['Q_gg_min_values'][i]
                inputs['Q_hulls'][i].loc[:, 'Qmax'] = inputs['Q_gg_max_values'][i]
                inputs['Q_hulls'][i].loc[:, 'Tmin'] = inputs['T_gauge_hourly_min_values'][i]
                inputs['Q_hulls'][i].loc[:, 'Tmax'] = inputs['T_gauge_hourly_max_values'][i]

            keys = ['Q_tot_daily','Q_max_daily_change','WaterToPowerConversion','T_gauge_daily','dT_daily']
            for key in keys:
                inputs[key].columns = [
                    col.replace('H', 'D') if col.startswith('H') else col
                    for col in inputs[key].columns
                ]
            inputs['daily_horizon'] = [f'D{i:02}' for i in range(1, 8)]

            return inputs
        except Exception as e:
            logger.error(f'Not able to parse Inputs. Error: {e}')
            sys.exit(1)
    except Exception as e:
        logger.error(f'Failed to read the file. Error: {e}')
        sys.exit(1)
