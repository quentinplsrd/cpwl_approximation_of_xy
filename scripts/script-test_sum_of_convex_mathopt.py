from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import matplotlib.pyplot as plt
from ortools.math_opt.python import mathopt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from cpwllib.implementation import MILP_or_QP_variables_and_constraints, N_from_target_error, list_faces_from_N, list_faces_from_N_DC, \
    plot_faces_3d, equations_from_faces_3d, evaluate_z_from_equations, \
    equations_sum_convex

target_error = 0.001
N = N_from_target_error(target_error)

# produce random values between 30 and 50 in a 5-by-10 matrix
N_nodes = 20
N_time = 10
X_Y_UB = 50.
X_Y_LB = 30.

x_fix, y_fix = X_Y_LB + (X_Y_UB-X_Y_LB)*np.random.rand(2,N_nodes,N_time)

# create a dummy model
model = mathopt.Model(name='Test')
# dictionary of variables mimicking bypass tool's T and Q
variables = {}
variables['T_gg'] = [[model.add_variable(lb=30., ub=50., name=f'T_gauge_{j}_{t}') 
                      for t in range(N_time)] for j in range(N_nodes)]
variables['q_gg'] = [[model.add_variable(lb=30., ub=50., name=f'q_gg_{j}_{t}') 
                      for t in range(N_time)] for j in range(N_nodes)]

# MOST IMPORTANT FUNCTION
# This add necessary variables to the model
new_variables = MILP_or_QP_variables_and_constraints(model, variables['T_gg'], variables['q_gg'],
                                   target_error=target_error, 
                                   partition_method='sum of convex',
                                   logarithmic_encoding=True)

# dummy objective function
model.maximize(new_variables["Z"][0][0])

# now fix T and q to force the solution
for p in range(N_nodes):
    for t in range(N_time):
        variables['T_gg'][p][t].lower_bound = x_fix[p][t]
        variables['T_gg'][p][t].upper_bound = x_fix[p][t]
        variables['q_gg'][p][t].lower_bound = y_fix[p][t]
        variables['q_gg'][p][t].upper_bound = y_fix[p][t]
   
# solve the problem
params = mathopt.SolveParameters(enable_output=True)
result = mathopt.solve(model, solver_type=mathopt.SolverType.GSCIP, params=params)
result.termination.reason
result.solve_stats

# Real Z values
real_Z_values = x_fix*y_fix

# Calculated by the model
model_Z_values = np.array([[result.variable_values()[new_variables["Z"][p][t]]
                  for t in range(N_time)] for p in range(N_nodes)])

Difference = real_Z_values-model_Z_values

print(f"Average error: theory = {(X_Y_UB-X_Y_LB)**2/(48*N**2):.3f}, actual = {abs(Difference).mean():.3f}")
print(f"Max error: theory = {(X_Y_UB-X_Y_LB)**2/(16*N**2):.3f}, actual = {abs(Difference).max():.3f}")
