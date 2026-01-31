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
# get triangles faces
faces = list_faces_from_N(N, method='triangles')
fig = plot_faces_3d(faces)
fig.savefig("figure-triangles.pdf", dpi=300)
fig.savefig("figure-triangles.png", dpi=300)

# get polygon faces
faces = list_faces_from_N(N, method='polygons')
fig = plot_faces_3d(faces)
fig.savefig("figure-polygons.pdf", dpi=300)
fig.savefig("figure-polygons.png", dpi=300)

# produce a random (x,y) number
x_fix, y_fix = np.random.rand(2)
print(f"x: {x_fix:.4f}, y: {y_fix:.4f},")
print(f"Real z = x*y: {x_fix*y_fix:.4f}")

faces = list_faces_from_N(N, method='triangles')
list_coeffs, list_equations = equations_from_faces_3d(faces)
z_values = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
                                     list_coeffs, list_equations)
print(f"Approximated z from triangles: {z_values[0]:.4f}")

faces = list_faces_from_N(N, method='polygons')
list_coeffs, list_equations = equations_from_faces_3d(faces)
z_values = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
                                     list_coeffs, list_equations)
print(f"Approximated z from polygons: {z_values[0]:.4f}")

list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k = equations_sum_convex(N)
z_values_j = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
                                     list_coeffs_j, list_equations_j)
z_values_k = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
                                     list_coeffs_k, list_equations_k)
print(f"Approximated z from sum of convex: {(z_values_j+z_values_k)[0]:.4f}")
