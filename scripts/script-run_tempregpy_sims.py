#%% Import packages
import os
import pandas as pd
import polars as pl
from cpwllib.tempregpy import Model, ModelConfig
from ortools.math_opt.python import mathopt
from cpwllib.tempregpy.utils import solve_result_gap
from enum import Enum
from cpwllib.tempregpy.model import Methods
import cpwllib.tempregpy.user as user
from cpwllib import DATA_DIR
import sys
from google.protobuf import text_format

#%% Run sims

df_dicts = []

SOLVE_TIME_LIMIT = 3600
OPTIMALITY_GAP = 0.005

for solver in [mathopt.SolverType.GSCIP, mathopt.SolverType.HIGHS]:
    for method in [Methods.SUM_OF_CONVEX, Methods.TRIANGLES, Methods.POLYGONS, Methods.QUADRATIC]:
        for target_error in [0.05,0.01,0.005,0.001]:
            if method == Methods.QUADRATIC and solver != mathopt.SolverType.GSCIP:
                continue

            if method==Methods.QUADRATIC:
                target_error=0.000001

            # df_.write_parquet(f"sim_results_{solver}_{method}_{target_error}.parquet")
            if os.path.exists(f"sim_results_{solver}_{method}_{target_error}.parquet"):
                print(f"Skipping existing results for solver {solver.name} and method {method.value} with target error {target_error}")
                df_ = pl.read_parquet(f"sim_results_{solver}_{method}_{target_error}.parquet")
                df_dicts.append(df_.to_dicts()[0])

                continue


            print(f"Running with solver {solver.name} and method {method.value} with target error {target_error}")
                    

            config = ModelConfig(
                name="CPWL Test Model",
                solver_type=solver,
                bilinear_method=method,
                target_error=target_error,
            )

            model = Model(config)


            excel_file = pd.ExcelFile(
                os.path.abspath(DATA_DIR / "User Inputs.xlsx")
            )
            excel_data = excel_file.parse("Inputs")
            inputs = user.load_inputs_from_excel(excel_data)
            # continue
            model.populate_from_inputs(inputs)

            integer_count = 0
            continuous_count = 0
            for var in model.mathopt_model.variables():
                if var.integer:
                    integer_count += 1
                else:
                    continuous_count += 1
                     
            print(f"Model has {integer_count} integer variables")

            model_proto = model.mathopt_model.export_model()

            # Save to file (text format)
            with open(f"{solver}_{method}_{target_error}.txt", "w") as f:
                f.write(text_format.MessageToString(model_proto))

            solve_result = model.solve(time_limit=SOLVE_TIME_LIMIT, optimality_gap=OPTIMALITY_GAP)
            solved = (
                    solve_result.termination.reason is mathopt.TerminationReason.FEASIBLE
                    or solve_result.termination.reason is mathopt.TerminationReason.OPTIMAL    
                )

            df_dicts.append({
                "solver": solver.name,
                "method": method.value,
                "status": solve_result == None,
                "target_error": target_error,
                "runtime": solve_result.solve_time() if solve_result else None,
                "variables": model.mathopt_model.get_num_variables(),
                "integer_variables": integer_count,
                "continuous_variables": continuous_count,
                "linear_constraints": model.mathopt_model.get_num_linear_constraints(),
                "quadratic_constraints": model.mathopt_model.get_num_quadratic_constraints(),
                "node_count": solve_result.solve_stats.node_count if solve_result else None,
                "simplex_iterations": solve_result.solve_stats.simplex_iterations if solve_result else None,
                "objective_value": solve_result.objective_value() if solved else None,
                "best_bound": solve_result.best_objective_bound() if solve_result else None
            })

            df_ = pl.DataFrame(df_dicts[-1])

            df_.write_parquet(f"sim_results_{solver}_{method}_{target_error}.parquet")

            if method==Methods.QUADRATIC:
                break

df = pl.DataFrame(df_dicts)

df.write_parquet("tempregpy_sim_results.parquet")
print(df)
# %%
