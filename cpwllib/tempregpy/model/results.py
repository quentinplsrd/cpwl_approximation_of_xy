import pandas as pd
from .constants import CFS_TO_AFH
from ..logging_config import *


def parse_model_results(inputs, result, variables):
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

    for var in ['q_bp', 'q_tur', 'q_tot', 'q_gg', 'q_source', 'q_sink', 'q_in_flow', 'q_out_flow', 'QT_latent',
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
                    t_gauge[to_node] = t_plant[from_node] + dt_hourly_values_df.at[f'{from_node} - {to_node}', time_interval]

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

    logger.info("Results have been parsed successfully")
    return results
