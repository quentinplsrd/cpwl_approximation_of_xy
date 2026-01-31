#%% import packages
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:29:08 2025

@author: qploussard
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib.ticker import LogLocator, ScalarFormatter
matplotlib.rcParams["axes3d.mouserotationstyle"] = 'azel'
from scipy.spatial import ConvexHull
from ortools.math_opt.python import mathopt

from matplotlib import gridspec

from typing import List, Tuple, Union


#%% functions


def list_shape(ndarray):
    if isinstance(ndarray, list):
        outermost_size = len(ndarray)
        if outermost_size == 0:
          return (0,)
        row_shape = list_shape(ndarray[0])
        return (outermost_size, *row_shape)
    else:
        return ()
    
def flatten_recursive(nested_list):
    flattened_list = []
    for item in nested_list:
        if isinstance(item, list):
            flattened_list.extend(flatten_recursive(item))
        else:
            flattened_list.append(item)
    return flattened_list

def reshape_list(flat_list, shape):
  """
  Reshapes a flat list into a nested list with a given shape.

  Args:
    flat_list: The flat list to reshape.
    shape: A tuple representing the desired shape (e.g., (2, 3) for 2x3).

  Returns:
    A nested list with the specified shape.
  """
  if not isinstance(flat_list, list):
    raise TypeError("Input must be a list.")
  if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
    raise TypeError("Shape must be a tuple of integers.")

  total_elements = len(flat_list)
  product_of_shape = 1
  for dim in shape:
    product_of_shape *= dim

  if total_elements != product_of_shape:
    raise ValueError("The number of elements in the flat_list does not match the total elements in the specified shape.")

  nested_list = []
  current_index = 0
  for i in range(shape[0]):
    row = []
    for j in range(shape[1]):
      if len(shape) > 2:
        inner_list = []
        for k in range(shape[2]):
          inner_list.append(flat_list[current_index])
          current_index += 1
        row.append(inner_list)
      else:
        row.append(flat_list[current_index])
        current_index += 1
    nested_list.append(row)
  return nested_list


def binary_rep(k,N):
    p = (N - 1).bit_length()
    return ((k >> np.arange(p)) & 1).astype(np.int32)


def equations_from_faces_3d(faces):
    list_coeffs = []
    list_equations = []
    for face in faces:
        # find the linear coeffs
        # only keep 3 points to calculate linear coeff
        M = face[:3,:]*1
        M[:,2] = 1.
        Z = face[:3,2]*1
        # coeffs = (a,b,c) in z = a*x + b*y + c
        coeffs = np.linalg.solve(M,Z)
        list_coeffs.append(coeffs)
        # find the domain inequalities
        hull_2d = ConvexHull(face[:,:2])
        list_equations.append(hull_2d.equations)
        
    return list_coeffs, list_equations

def evaluate_z_from_equations(x_y_array, list_coeffs, list_equations):
    z_values = []
    for pt in x_y_array:
        x = pt[0]
        y = pt[1]
        belong_to_domain = [(eq[:,0]*x + eq[:,1]*y + eq[:,2] <= 0.).all() for eq in list_equations]
        index_domain = np.where(belong_to_domain)[0][0]
        z = sum(list_coeffs[index_domain]*np.array([x, y, 1.0]))
        z_values.append(z)
    return np.array(z_values)

def equations_sum_convex(N):
    list_coeffs_j = []
    list_equations_j = []
    list_coeffs_k = []
    list_equations_k = []
    for j in range(2*N):
        tmp_eq = []
        if j>0:
            tmp_eq.append([-1,-1,j/N])
        if j<2*N-1:
            tmp_eq.append([1,1,-(j+1)/N])
        tmp_eq = np.array(tmp_eq)
        list_equations_j.append(tmp_eq)
        a = (j+1)/(2*N) - 0.5
        b = j/(2*N) + 0.5
        c = -j*(j+1)/(4*N**2) + 0.25
        list_coeffs_j.append(np.array([a,b,c]))
    for k in range(2*N):
        tmp_eq = []
        if k<2*N-1:
            tmp_eq.append([-1,1,1-(k+1)/N])
        if k>0:
            tmp_eq.append([1,-1,-1+k/N])
        tmp_eq = np.array(tmp_eq)
        list_equations_k.append(tmp_eq)
        a = k/(2*N)
        b = -k/(2*N)
        c = k*(k+1)/(4*N**2) - (2*k+1)/(4*N)
        list_coeffs_k.append(np.array([a,b,c]))
    return list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k


def evaluate_gn(x_y_array, N):
    j_vec = np.arange(2*N)
    k_vec = np.arange(2*N)
    mat_plus = np.c_[(2*j_vec+1)/(4*N), (2*j_vec+1)/(4*N), - (j_vec*(j_vec+1))/(4*N**2)]
    mat_minus = np.c_[-(2*(k_vec-N)+1)/(4*N), (2*(k_vec-N)+1)/(4*N), - (k_vec*(k_vec+1) - 2*N*k_vec - N*(1-N))/(4*N**2)]
    x_y_1 = np.insert(x_y_array,2,1,axis=1)
    z_plus = np.max(np.matmul(mat_plus,x_y_1.T),axis=0)
    z_minus = np.max(np.matmul(mat_minus,x_y_1.T),axis=0)
    return z_plus, z_minus


def list_faces_from_N_DC(N):
    faces_plus = []
    faces_minus = []
    eps = 1e-3/N
    # first, assume 'polygons' method
    # later, split the squares into triangles if 'triangles' method
    for j in range(2*N):
        pts = []
        if j==0:
            pts.append([0,0,evaluate_gn(np.array([[0,0]]), N)[0][0]])
        elif j<N:
            pts.append([0,j/N,evaluate_gn(np.array([[0,j/N]]), N)[0][0]])
            pts.append([j/N,0,evaluate_gn(np.array([[j/N,0]]), N)[0][0]])
        else:
            pts.append([(j-N)/N,1,evaluate_gn(np.array([[(j-N)/N,1]]), N)[0][0]])
            pts.append([1,(j-N)/N,evaluate_gn(np.array([[1,(j-N)/N]]), N)[0][0]])
        if j==2*N-1:
            pts.append([1,1,evaluate_gn(np.array([[1,1]]), N)[0][0]])
        elif j<N:
            pts.append([(j+1)/N,0,evaluate_gn(np.array([[(j+1)/N,0]]), N)[0][0]])
            pts.append([0,(j+1)/N,evaluate_gn(np.array([[0,(j+1)/N]]), N)[0][0]])
        else:
            pts.append([1,(j+1-N)/N,evaluate_gn(np.array([[1,(j+1-N)/N]]), N)[0][0]])
            pts.append([(j+1-N)/N,1,evaluate_gn(np.array([[(j+1-N)/N,1]]), N)[0][0]])
        pts = np.array(pts)
        faces_plus.append(pts)
    for k in range(2*N):
        pts = []
        if k==0:
            pts.append([1,0,evaluate_gn(np.array([[1,0]]), N)[1][0]])
        elif k<N:
            pts.append([1-k/N,0,evaluate_gn(np.array([[1-k/N,0]]), N)[1][0]])
            pts.append([1,k/N,evaluate_gn(np.array([[1,k/N]]), N)[1][0]])
        else:
            pts.append([0,(k-N)/N,evaluate_gn(np.array([[0,(k-N)/N]]), N)[1][0]])
            pts.append([1-(k-N)/N,1,evaluate_gn(np.array([[1-(k-N)/N,1]]), N)[1][0]])
        if k==2*N-1:
            pts.append([0,1,evaluate_gn(np.array([[0,1]]), N)[1][0]])
        elif k<N:
            pts.append([1,(k+1)/N,evaluate_gn(np.array([[1,(k+1)/N]]), N)[1][0]])
            pts.append([1-(k+1)/N,0,evaluate_gn(np.array([[1-(k+1)/N,0]]), N)[1][0]])
        else:
            pts.append([1-(k+1-N)/N,1,evaluate_gn(np.array([[1-(k+1-N)/N,1]]), N)[1][0]])
            pts.append([0,(k+1-N)/N,evaluate_gn(np.array([[0,(k+1-N)/N]]), N)[1][0]])
        pts = np.array(pts)
        faces_minus.append(pts)
        
    return faces_plus, faces_minus
    

    
def N_from_target_error(target_error):
    return int(np.ceil(1/(4*np.sqrt(target_error))))

def list_faces_from_N(N, method='triangles'):
    faces = []
    eps = 1e-3/N
    # first, assume 'polygons' method
    # later, split the squares into triangles if 'triangles' method
    for j in range(2*N):
        for k in range(2*N):
            x = (j-k)/(2*N) + 0.5
            y = (j+k+1)/(2*N) - 0.5
            if (x>=0) and (x<=1) and (y>=0) and (y<=1):
                pts = []
                for delta in [(-1,0),(0,-1),(1,0),(0,1)]:
                        xx = x + delta[0]/(2*N)
                        yy = y + delta[-1]/(2*N)
                        if (xx>=0-eps) and (xx<=1+eps) and (yy>=0-eps) and (yy<=1+eps):
                            pts.append([xx,yy,xx*yy])
                pts = np.array(pts)
                faces.append(pts)
    if method=='triangles':
        faces_triang=[]
        for face in faces:
            if len(face)==3:
                faces_triang.append(face)
            else:
                x,y = face.mean(axis=0)[0:2]
                if (y>=x - eps) ^ (x+y>=1-eps):
                    face = face[face[:,0].argsort(),:]
                else:
                    face = face[face[:,1].argsort(),:]
                faces_triang.append(face[0:3,:])
                faces_triang.append(face[1:4,:])
        faces = list(faces_triang)
    return faces

def plot_faces_3d(faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if isinstance(faces, tuple):
        for i, f in enumerate(faces):
            faceCollection = Poly3DCollection(f,shade=False,facecolors=f'C{i}',edgecolors='k',alpha=0.6)
            ax.add_collection3d(faceCollection)
    else:
        faceCollection = Poly3DCollection(faces,shade=False,facecolors='C1',edgecolors='k',alpha=0.5)
        ax.add_collection3d(faceCollection)

def plot_faces_2d(faces):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    faceCollection = PolyCollection([face[:,:2] for face in faces], facecolors='white', edgecolors='k')
    ax.add_collection(faceCollection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    # ax.set_aspect('equal', adjustable='datalim')
    # ax.relim()
    # ax.autoscale_view()


# NOTE:
# For quadratic (QP) mode, run: 
# MILP_or_QP_variables_and_constraints(model, X, Y, quadratic=True)
# For MILP mode, run:
# MILP_or_QP_variables_and_constraints(model, X, Y, 
#                                       quadratic=False,
#                                       target_error= [target_error],
#                                       partition_method= [method_name],
#                                       logarithmic_encoding= [True/False])


def MILP_or_QP_variables_and_constraints(model, X, Y, 
                                         quadratic=False,
                                         target_error=0.01,
                                         partition_method='triangles',
                                         logarithmic_encoding=True):
    
    # if quadratic: uses the real constraint Z = X*Y
    # make sure the solver is SCIP or Gurobi
    # if not: uses the PWL approximations with one of the three partition methods
    
    # the three partition methods are:
    # - triangles
    # - polygons
    # - sum of convex
    
    if not quadratic:
        if partition_method not in ['triangles','polygons','sum of convex']:
            print(f'The partition method "{partition_method}" is not valid.')
            return
    
    # Get dimension of variables_X and variables_Y
    N_nodes = len(X)
    N_time = len(X[0])
    
    # # initial variables for Z = X*Y (Q and T)
    # X = model.add_variable(name="X")
    # Y = model.add_variable(name="Y")
    Z = [[model.add_variable(name=f"Z({p},{t})") 
          for t in range(N_time)] for p in range(N_nodes)]

    
    # Resolution level
    N = N_from_target_error(target_error)
    
    
    # rescaled variables
    x = [[model.add_variable(name=f"x({p},{t})", lb=0., ub=1.)
          for t in range(N_time)] for p in range(N_nodes)]
    y = [[model.add_variable(name=f"y({p},{t})", lb=0., ub=1.)
          for t in range(N_time)] for p in range(N_nodes)]
    z = [[model.add_variable(name=f"z({p},{t})", lb=0., ub=1.)
          for t in range(N_time)] for p in range(N_nodes)]
    
    if quadratic:
        
        print('Make sure to use SCIP or Gurobi as the solver for quadratic constraints')
        
    else:
    
        z_c = [[[] for t in range(N_time)] for p in range(N_nodes)]
        
        xx = [[[] for t in range(N_time)] for p in range(N_nodes)]
        yy = [[[] for t in range(N_time)] for p in range(N_nodes)]
        zz = [[[] for t in range(N_time)] for p in range(N_nodes)]
        bv = [[[] for t in range(N_time)] for p in range(N_nodes)]
        gv = [[[] for t in range(N_time)] for p in range(N_nodes)]
        
        if partition_method in ['triangles','polygons']:
            N_C = 1
            # get polygons/triangles faces
            faces = list_faces_from_N(N, method=partition_method)
            # get linear coeffs and region inequalities
            list_coeffs, list_equations = equations_from_faces_3d(faces)
            COEFFS = [list_coeffs]
            EQUATIONS = [list_equations]
        else:
            N_C = 2
            # get linear coeffs and region inequalities
            list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k = equations_sum_convex(N)
            COEFFS = [list_coeffs_j, list_coeffs_k]
            EQUATIONS = [list_equations_j, list_equations_k]

    for p in range(N_nodes):
        for t in range(N_time):
            
            X_min = X[p][t].lower_bound
            X_max = X[p][t].upper_bound
            Y_min = Y[p][t].lower_bound
            Y_max = Y[p][t].upper_bound

            # link between initial and rescaled variables
            model.add_linear_constraint(X[p][t] == X_min + (X_max-X_min)*x[p][t], name=f"X_to_x({p},{t})")
            model.add_linear_constraint(Y[p][t] == Y_min + (Y_max-Y_min)*y[p][t], name=f"Y_to_y({p},{t})")
            model.add_linear_constraint(Z[p][t] == X_min*Y_min + Y_min*(X_max-X_min)*x[p][t]
                                        + X_min*(Y_max-Y_min)*y[p][t] + (Y_max-Y_min)*(X_max-X_min)*z[p][t], name=f"Z_to_z({p},{t})")
            
            
            if quadratic:
                
                model.add_quadratic_constraint((0.0 <= x[p][t]*y[p][t] - z[p][t]) <= 0.0, name=f"z_is_xy({p},{t})")
            
            else:
                      
                # z_c = [model.add_variable(name=f"z_c({cc})") for cc in range(N_C)]
                z_c[p][t] = [model.add_variable(name=f"z_c({p},{t},{cc})") for cc in range(N_C)]
                # z = zc_1 + zc_2
                model.add_linear_constraint(z[p][t] == sum(z_c[p][t]), name=f"sum_convex({p},{t})")  
            
                # xx = []
                # yy = []
                # zz = []
                # bv = []
            
                for cc in range(N_C):
                    coeffs = list(COEFFS[cc])
                    equations = list(EQUATIONS[cc])
            
                    # number of regions
                    NN = len(equations)
                    # P = np.ceil(np.log2(NN)).astype(int)
                    P = (NN-1).bit_length()
                    
                    # define regional components of x, y, z
                    xx_c = [model.add_variable(name=f"xx({p},{t},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
                    yy_c = [model.add_variable(name=f"yy({p},{t},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
                    zz_c = [model.add_variable(name=f"zz({p},{t},{cc},{i})") for i in range(NN)]
                    # binary variable indicating which region is active
                    if logarithmic_encoding:
                        bv_c =  [model.add_variable(name=f"bv({p},{t},{cc},{i})", lb=0., ub=1.) for i in range(NN)]
                        gv_c =  [model.add_binary_variable(name=f"gv({p},{t},{cc},{q})") for q in range(P)]
                    else:
                        bv_c =  [model.add_binary_variable(name=f"bv({p},{t},{cc},{i})") for i in range(NN)]
                
                    xx[p][t].append(xx_c)
                    yy[p][t].append(yy_c)
                    zz[p][t].append(zz_c)
                    bv[p][t].append(bv_c)
                    
                    if logarithmic_encoding:
                        gv[p][t].append(gv_c)
                    
                    # x, y, z are equal to the sum of their components
                    model.add_linear_constraint(x[p][t] == sum(xx_c), name=f"sum_xx({p},{t},{cc})")
                    model.add_linear_constraint(y[p][t] == sum(yy_c), name=f"sum_yy({p},{t},{cc})")
                    model.add_linear_constraint(z_c[p][t][cc] == sum(zz_c), name=f"sum_zz({p},{t},{cc})")
                    # only one region can be active at a time
                    model.add_linear_constraint(sum(bv_c) == 1., name=f"sum_bv({p},{t},{cc})")
                    
                    if logarithmic_encoding:
                        binary_array = np.array([binary_rep(i,NN) for i in range(NN)])
                        for q in range(P):
                            model.add_linear_constraint(
                                sum([bv_c[i] for i in range(NN) if binary_array[i,q]==1]) <= gv_c[q],
                                name=f"gv_lb({p},{t},{cc},{q})")
                            model.add_linear_constraint(
                                sum([bv_c[i] for i in range(NN) if binary_array[i,q]==0]) <= 1 - gv_c[q],
                                name=f"gv_ub({p},{t},{cc},{q})")
                        # Previous, inefficient constraints    
                        # for i in range(NN):
                        #     bin_rep = binary_rep(i,NN)
                        #     model.add_linear_constraint(
                        #         bv_c[i] >= 1 - P + 
                        #         sum([bin_rep[q]*gv_c[q] + (1-bin_rep[q])*(1 - gv_c[q]) for q in range(P)]), 
                        #         name=f"bv_lb({p},{t},{cc},{i})")
                        #     for q in range(P):
                        #         model.add_linear_constraint(bv_c[i] <= bin_rep[q]*gv_c[q] + (1-bin_rep[q])*(1 - gv_c[q]), name=f"bv_ub({p},{t},{cc},{i},{q})")
            
                    for i in range(NN):
                        a, b, c = coeffs[i]
                        # relationship between x,y,z on a given region
                        model.add_linear_constraint(zz_c[i] == a*xx_c[i] + b*yy_c[i] + c*bv_c[i])
                        # if a region is not active, the coordinates of the component is (0,0)
                        model.add_linear_constraint(xx_c[i] <= bv_c[i])
                        model.add_linear_constraint(yy_c[i] <= bv_c[i])
                        for j in range(len(equations[i])):
                            aa, bb, cc = equations[i][j]
                            # inequalities defining the region
                            model.add_linear_constraint(aa*xx_c[i] + bb*yy_c[i] + cc*bv_c[i] <= 0.)

    if quadratic:
        variables = {'X': X, 'Y': Y, 'Z': Z,
                     'x': x, 'y': y, 'z': z}
    else:         
        variables = {'X': X, 'Y': Y, 'Z': Z,
                     'x': x, 'y': y, 'z': z,
                     'xx': xx, 'yy': yy, 'zz': zz,
                     'z_c': z_c, 'bv': bv, 'gv': gv}
    
    return variables
    
# #%% code to test the functions

# target_error = 0.001

# N = N_from_target_error(target_error)
# # get triangles faces
# faces = list_faces_from_N(N, method='triangles')
# plot_faces_3d(faces)
# # get polygon faces
# faces = list_faces_from_N(N, method='polygons')
# plot_faces_3d(faces)

# # produce a random (x,y) number
# x_fix, y_fix = np.random.rand(2)
# print(f"x: {x_fix:.4f}, y: {y_fix:.4f},")
# print(f"Real z = x*y: {x_fix*y_fix:.4f}")

# faces = list_faces_from_N(N, method='triangles')
# list_coeffs, list_equations = equations_from_faces_3d(faces)
# z_values = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
#                                      list_coeffs, list_equations)
# print(f"Approximated z from triangles: {z_values[0]:.4f}")

# faces = list_faces_from_N(N, method='polygons')
# list_coeffs, list_equations = equations_from_faces_3d(faces)
# z_values = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
#                                      list_coeffs, list_equations)
# print(f"Approximated z from polygons: {z_values[0]:.4f}")

# list_coeffs_j, list_equations_j, list_coeffs_k, list_equations_k = equations_sum_convex(N)
# z_values_j = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
#                                      list_coeffs_j, list_equations_j)
# z_values_k = evaluate_z_from_equations(np.array([[x_fix, y_fix]]), 
#                                      list_coeffs_k, list_equations_k)
# print(f"Approximated z from sum of convex: {(z_values_j+z_values_k)[0]:.4f}")



# # produce random values between 30 and 50 in a 5-by-10 matrix
# N_nodes = 20
# N_time = 10
# X_Y_UB = 50.
# X_Y_LB = 30.

# x_fix, y_fix = X_Y_LB + (X_Y_UB-X_Y_LB)*np.random.rand(2,N_nodes,N_time)

# # create a dummy model
# model = mathopt.Model(name='Test')
# # dictionary of variables mimicking bypass tool's T and Q
# variables = {}
# variables['T_gg'] = [[model.add_variable(lb=30., ub=50., name=f'T_gauge_{j}_{t}') 
#                       for t in range(N_time)] for j in range(N_nodes)]
# variables['q_gg'] = [[model.add_variable(lb=30., ub=50., name=f'q_gg_{j}_{t}') 
#                       for t in range(N_time)] for j in range(N_nodes)]

# # MOST IMPORTANT FUNCTION
# # This add necessary variables to the model
# new_variables = MILP_or_QP_variables_and_constraints(model, variables['T_gg'], variables['q_gg'],
#                                    target_error=target_error,
#                                    quadratic=True, 
#                                    partition_method='sum of convex',
#                                    logarithmic_encoding=True)

# # dummy objective function
# model.maximize(new_variables["Z"][0][0])

# # now fix T and q to force the solution
# for p in range(N_nodes):
#     for t in range(N_time):
#         variables['T_gg'][p][t].lower_bound = x_fix[p][t]
#         variables['T_gg'][p][t].upper_bound = x_fix[p][t]
#         variables['q_gg'][p][t].lower_bound = y_fix[p][t]
#         variables['q_gg'][p][t].upper_bound = y_fix[p][t]
   
# # solve the problem
# params = mathopt.SolveParameters(enable_output=True)
# %time result = mathopt.solve(model, solver_type=mathopt.SolverType.GSCIP, params=params)
# result.termination.reason
# result.solve_stats

# # Real Z values
# real_Z_values = x_fix*y_fix

# # Calculated by the model
# model_Z_values = np.array([[result.variable_values()[new_variables["Z"][p][t]]
#                   for t in range(N_time)] for p in range(N_nodes)])

# Difference = real_Z_values-model_Z_values

# print(f"Average error: theory = {(X_Y_UB-X_Y_LB)**2/(48*N**2):.3f}, actual = {abs(Difference).mean():.3f}")
# print(f"Max error: theory = {(X_Y_UB-X_Y_LB)**2/(16*N**2):.3f}, actual = {abs(Difference).max():.3f}")


# #%% Try equations from Maris


# def g_plus(points,N):
#     x = points[:,0]
#     y = points[:,1]
#     return np.max(
#         np.array(
#             [(2*j+1)*x/(4*N) + (2*j+1)*y/(4*N) - (j*(j+1))/(4*N**2) for j in range(2*N)]
#             ),
#         axis=0)

# def g_minus(points,N):
#     x = points[:,0]
#     y = points[:,1]
#     return np.max(
#         np.array(
#             [-(2*k+1)*x/(4*N) + (2*k+1)*y/(4*N) - (k*(k+1))/(4*N**2) for k in range(-N,N)]
#             ),
#         axis=0)

# # 1D points between 0 and 1
# x = np.linspace(0, 1, 101)
# y = np.linspace(0, 1, 101)

# # create a 2D grid
# X, Y = np.meshgrid(x, y)

# # stack into coordinate pairs if needed
# points = np.column_stack([X.ravel(), Y.ravel()])

# Z = (g_plus(points,N) - g_minus(points,N)).reshape(-1,len(x))


# fig = plt.figure(figsize=(7,5))
# ax = fig.add_subplot(111, projection="3d")
# surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none")

# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("f(x,y)")
# fig.colorbar(surf, shrink=0.5, aspect=5)

# plt.show()




# #%% test quadratic

# # create a dummy model
# model = mathopt.Model(name='Test')

# x = model.add_variable(lb=-1., ub=1., name="x")
# y = model.add_variable(lb=-1., ub=1., name="y")
# z = model.add_variable(lb=0., ub=1., name="z")
# # a = model.add_binary_variable(name="a")
# # a = model.add_variable(lb=0., ub=1., name="a")

# # model.add_quadratic_constraint((0.0 <= x*x - y*y - z) <= 0.0)
# model.add_quadratic_constraint((0.0 <= x*y - z) <= 0.0)
# model.add_linear_constraint(x + y == 1.)
# # model.add_quadratic_constraint((0.0 <= z*x - a) <= 0.0)
# # model.add_linear_constraint(x<=a)
                               
# model.maximize(z)

# # x.lower_bound = 0.5
# # x.upper_bound = 0.5
# # y.lower_bound = 0.5
# # y.upper_bound = 0.5

# params = mathopt.SolveParameters(enable_output=True)
# %time result = mathopt.solve(model, solver_type=mathopt.SolverType.GSCIP, params=params)
# result.termination.reason
# result.solve_stats
# result.variable_values()[x]
# result.variable_values()[y]


# #%% For article figures

# plot_article_figures = False

# #%% R_0

# if plot_article_figures:
    
#     # fig, ax = plt.subplots(1, 2, figsize=(10,5),dpi=150)
#     fig = plt.figure(figsize=(10,5),dpi=150, constrained_layout=True)
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
#     ax=[]
#     ax.append(fig.add_subplot(gs[0]))
#     ax.append(fig.add_subplot(gs[1]))
    
#     ax[0].set_xlabel('x')
#     ax[0].set_ylabel('y')
    
#     # Parameters
#     x0, y0 = 0, 0
#     delta = 1
    
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax[0].plot(diamond[:, 0], diamond[:, 1], 'k-', lw=2, label='Diamond Square')
    
#     # Plot centroid
#     # ax.plot(x0, y0, 'ro', label='Centroid')
#     ax[0].text(x0+0.05, y0+0.05, f'$(x_0,y_0)$')
    
#     # Plot axes
#     for k in [-1,0,1]:
#         ax[0].axhline(y=y0+k*delta, color='gray', linestyle='--', linewidth=1)
#         ax[0].axvline(x=x0+k*delta, color='gray', linestyle='--', linewidth=1)
    
#     # Annotate axis lines
#     # ax.text(x0 + 0.05, y0 + delta + 0.1, r'$x = x_0$', color='gray')
#     # ax.text(x0 + delta + 0.1, y0 + 0.05, r'$y = y_0$', color='gray')
    
#     # Define lines (edges of the diamond)
#     lines = [
#         ((x0 - delta, y0), (x0, y0 + delta)),
#         ((x0, y0 + delta), (x0 + delta, y0)),
#         ((x0 + delta, y0), (x0, y0 - delta)),
#         ((x0, y0 - delta), (x0 - delta, y0))
#     ]
    
#     offset = 0.3  # outward distance from center for label placement
    
#     for (x1, y1), (x2, y2) in lines:
#         # Midpoint
#         x_mid = (x1 + x2) / 2
#         y_mid = (y1 + y2) / 2
    
#         # Direction vector of the edge
#         dx = x2 - x1
#         dy = y2 - y1
    
#         # Normal vector pointing outward from (x0, y0)
#         nx = y_mid - y0
#         ny = -(x_mid - x0)
#         norm = np.hypot(nx, ny)
#         nx /= norm
#         ny /= norm
    
#         # Offset position away from center
#         x_text = x_mid #+ offset * nx
#         y_text = y_mid + 0.05 #+ offset * ny
    
#         # Compute slope and angle
#         m = (y2 - y1) / (x2 - x1)
#         angle = np.degrees(np.arctan2(dy, dx))
    
#         # Ensure text is upright
#         if angle > 90:
#             angle -= 180
#         elif angle < -90:
#             angle += 180
    
#         # Place label
#         ax[0].text(
#             x_text, y_text,
#             rf"$y = y_0 {'-' if m<0 else '+'}x {'+' if m<0 else '-'}x_0 {'+' if (x_mid-x0)*m<0 else '-'}\delta$",
#             fontsize=9,
#             rotation=angle,
#             rotation_mode='anchor',
#             ha='center',
#             va='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='none', alpha=0.7)
#         )
    
#     # Symbolic tick labels
#     xticks = [x0 - delta, x0, x0 + delta]
#     yticks = [y0 - delta, y0, y0 + delta]
#     ax[0].set_xticks(xticks)
#     ax[0].set_yticks(yticks)
#     ax[0].set_xticklabels([r'$x_0 - \delta$', r'$x_0$', r'$x_0 + \delta$'])
#     ax[0].set_yticklabels([r'$y_0 - \delta$', r'$y_0$', r'$y_0 + \delta$'])
    
#     # Final formatting
#     ax[0].set_aspect('equal')
#     ratio_lim = 1.0
#     ax[0].set_xlim(x0 - ratio_lim * delta, x0 + ratio_lim * delta)
#     ax[0].set_ylim(y0 - ratio_lim * delta, y0 + ratio_lim * delta)
#     # ax.legend()
#     ax[0].set_title('$R_0$ domain')
    
#     # alternative plot for |E|
    
#     # Create a square grid
#     x = np.linspace(-1, 1, 1000)
#     y = np.linspace(-1, 1, 1000)
#     x, y = np.meshgrid(x, y)
    
#     z = x * y
    
#     mask_idx = (x + y <= 1) & (x + y >= -1) & (y - x <= 1) & (y - x >= -1)
    
#     z[~mask_idx] = np.nan
    
    
#     # Plot
    
#     c = ax[1].pcolormesh(x, y, abs(z), cmap='viridis', shading='auto')
#     ax[1].set_aspect('equal')
#     # Add colorbar
#     cb_ax = fig.add_subplot(gs[2])
#     cb = fig.colorbar(c, cax=cb_ax)
    
#     # cb = fig.colorbar(c, ax=ax[1], pad=0.1)
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax[1].plot(diamond[:, 0], diamond[:, 1], 0, 'black', linewidth=0.3)
    
#     # Custom x and y ticks
#     ax[1].set_xticks([x0 - delta, x0, x0 + delta])
#     ax[1].set_xticklabels([r'$x_0 - \delta$', r'$x_0$', r'$x_0 + \delta$'])
    
#     ax[1].set_yticks([y0 - delta, y0, y0 + delta])
#     ax[1].set_yticklabels([r'$y_0 - \delta$', r'$y_0$', r'$y_0 + \delta$'])
    
#     ax[1].set_xlabel('x')
#     # ax[1].set_ylabel('y')
    
#     # Set abstract tick labels based on \delta
#     cb.set_ticks([0, 0.25])
#     cb.set_ticklabels([r'$0$', r'$\delta^2/4$'])
    
#     ax[1].set_title(r'Value of $|E(x,y)|$')
    
#     # plt.tight_layout()
#     plt.show()

# #%% R_0 alternative

# if plot_article_figures:
    
#     # fig, ax = plt.subplots(1, 2, figsize=(10,5),dpi=150)
#     fig = plt.figure(figsize=(10,5),dpi=150, constrained_layout=True)
#     gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
#     ax=[]
#     ax.append(fig.add_subplot(gs[0]))
#     ax.append(fig.add_subplot(gs[1]))
    
#     ax[0].set_xlabel('x')
#     ax[0].set_ylabel('y')
    
#     # Parameters
#     x0, y0 = 0, 0
#     delta = 1
    
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax[0].plot(diamond[:, 0], diamond[:, 1], 'k-', lw=2, label='Diamond Square')
    
#     # Plot centroid
#     # ax.plot(x0, y0, 'ro', label='Centroid')
#     ax[0].text(x0+0.05, y0+0.05, f'$(0,0)$')
    
#     # Plot axes
#     for k in [-1,0,1]:
#         ax[0].axhline(y=y0+k*delta, color='gray', linestyle='--', linewidth=1)
#         ax[0].axvline(x=x0+k*delta, color='gray', linestyle='--', linewidth=1)
    
#     # Annotate axis lines
#     # ax.text(x0 + 0.05, y0 + delta + 0.1, r'$x = x_0$', color='gray')
#     # ax.text(x0 + delta + 0.1, y0 + 0.05, r'$y = y_0$', color='gray')
    
#     # Define lines (edges of the diamond)
#     lines = [
#         ((x0 - delta, y0), (x0, y0 + delta)),
#         ((x0, y0 + delta), (x0 + delta, y0)),
#         ((x0 + delta, y0), (x0, y0 - delta)),
#         ((x0, y0 - delta), (x0 - delta, y0))
#     ]
    
#     offset = 0.3  # outward distance from center for label placement
    
#     for (x1, y1), (x2, y2) in lines:
#         # Midpoint
#         x_mid = (x1 + x2) / 2
#         y_mid = (y1 + y2) / 2
    
#         # Direction vector of the edge
#         dx = x2 - x1
#         dy = y2 - y1
    
#         # Normal vector pointing outward from (x0, y0)
#         nx = y_mid - y0
#         ny = -(x_mid - x0)
#         norm = np.hypot(nx, ny)
#         nx /= norm
#         ny /= norm
    
#         # Offset position away from center
#         x_text = x_mid #+ offset * nx
#         y_text = y_mid + 0.05 #+ offset * ny
    
#         # Compute slope and angle
#         m = (y2 - y1) / (x2 - x1)
#         angle = np.degrees(np.arctan2(dy, dx))
    
#         # Ensure text is upright
#         if angle > 90:
#             angle -= 180
#         elif angle < -90:
#             angle += 180
    
#         # Place label
#         ax[0].text(
#             x_text, y_text,
#             rf"$y = {'-' if m<0 else ''}x {'+' if (x_mid-x0)*m<0 else '-'}\rho$",
#             fontsize=9,
#             rotation=angle,
#             rotation_mode='anchor',
#             ha='center',
#             va='bottom',
#             bbox=dict(boxstyle='round,pad=0.5', fc='none', ec='none', alpha=0.7)
#         )
    
#     # Symbolic tick labels
#     xticks = [x0 - delta, x0, x0 + delta]
#     yticks = [y0 - delta, y0, y0 + delta]
#     ax[0].set_xticks(xticks)
#     ax[0].set_yticks(yticks)
#     ax[0].set_xticklabels([r'$- \rho$', r'0', r'$+ \rho$'])
#     ax[0].set_yticklabels([r'$- \rho$', r'0', r'$+ \rho$'])
    
#     # Final formatting
#     ax[0].set_aspect('equal')
#     ratio_lim = 1.0
#     ax[0].set_xlim(x0 - ratio_lim * delta, x0 + ratio_lim * delta)
#     ax[0].set_ylim(y0 - ratio_lim * delta, y0 + ratio_lim * delta)
#     # ax.legend()
#     ax[0].set_title(r'$R_{\rho}$ domain')
    
#     # alternative plot for |E|
    
#     # Create a square grid
#     x = np.linspace(-1, 1, 1000)
#     y = np.linspace(-1, 1, 1000)
#     x, y = np.meshgrid(x, y)
    
#     z = x * y
    
#     mask_idx = (x + y <= 1) & (x + y >= -1) & (y - x <= 1) & (y - x >= -1)
    
#     z[~mask_idx] = np.nan
    
    
#     # Plot
    
#     c = ax[1].pcolormesh(x, y, abs(z), cmap='viridis', shading='auto')
#     ax[1].set_aspect('equal')
#     # Add colorbar
#     cb_ax = fig.add_subplot(gs[2])
#     cb = fig.colorbar(c, cax=cb_ax)
    
#     # cb = fig.colorbar(c, ax=ax[1], pad=0.1)
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax[1].plot(diamond[:, 0], diamond[:, 1], 0, 'black', linewidth=0.3)
    
#     # Custom x and y ticks
#     ax[1].set_xticks([x0 - delta, x0, x0 + delta])
#     ax[1].set_xticklabels([r'$- \rho$', r'$0$', r'$+ \rho$'])
    
#     ax[1].set_yticks([y0 - delta, y0, y0 + delta])
#     ax[1].set_yticklabels([r'$- \rho$', r'$0$', r'$+ \rho$'])
    
#     ax[1].set_xlabel('x')
#     # ax[1].set_ylabel('y')
    
#     # Set abstract tick labels based on \delta
#     cb.set_ticks([0, 0.25])
#     cb.set_ticklabels([r'$0$', r'$\rho^2/4$'])
    
#     ax[1].set_title(r'Value of $|E_{\rho}(x,y)|$')
    
#     # plt.tight_layout()
#     plt.show()



# #%% plot z=x*y

# if plot_article_figures:
        
#     fig = plt.figure(figsize=(10,6),dpi=250)
#     ax = fig.add_subplot(projection='3d')
    
    
#     x, y = np.mgrid[0:1:1000j, 0:1:1000j]
#     z = x * y
    
#     surf = ax.plot_surface(x, y, z, cmap='viridis', alpha=1.0,
#                            antialiased=False, linewidth=0, edgecolor='none',)
    
    
#     # Set view angle (elevation and azimuth)
#     ax.view_init(elev=15, azim=-15)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
    
    
#     plt.tight_layout()
#     # plt.show()
    
#     plt.savefig('test.png')


# #%% plot z=(x-x_0)*(y-y_0)

# if plot_article_figures:
        
#     fig = plt.figure(figsize=(10,4),dpi=250)
#     ax = fig.add_subplot(projection='3d')
    
    
#     x, y = np.mgrid[-1:1:1000j, -1:1:1000j]
#     z = x * y
    
#     mask_idx = (x + y <= 1) & (x + y >= -1) & (y - x <= 1) & (y - x >= -1)
    
#     z[~mask_idx] = np.nan
    
#     z_0 = x*0.
#     z_0[~mask_idx] = np.nan
    
#     # Parameters
#     x0, y0 = 0, 0
#     delta = 1
    
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax.plot(diamond[:, 0], diamond[:, 1], 0, 'k-')
    
#     surf = ax.plot_surface(x, y, abs(z), cmap='viridis', alpha=1.0,
#                            antialiased=False, linewidth=0, edgecolor='none',)
    
#     # ax.plot_surface(x, y, z_0, color='C1', alpha=0.5,
#     #                        antialiased=False, linewidth=0, edgecolor='none',)
    
#     # Set view angle (elevation and azimuth)
#     ax.view_init(elev=15, azim=-15)
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
    
#     ax.set_zlim([0,1])
    
    
#     plt.tight_layout()
#     # plt.show()
    
#     plt.savefig('test.png')


# #%% alternative plot for |E|

# if plot_article_figures:
    
    
#     # Create a square grid
#     x = np.linspace(-1, 1, 1000)
#     y = np.linspace(-1, 1, 1000)
#     x, y = np.meshgrid(x, y)
    
#     z = x * y
    
#     mask_idx = (x + y <= 1) & (x + y >= -1) & (y - x <= 1) & (y - x >= -1)
    
#     z[~mask_idx] = np.nan
    
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(10,5),dpi=150)
#     c = ax.pcolormesh(x, y, abs(z), cmap='viridis', shading='auto')
#     ax.set_aspect('equal')
#     # Add colorbar
#     cb = fig.colorbar(c, ax=ax)
    
#     # Define diamond corners
#     diamond = np.array([
#         [x0,         y0 + delta],
#         [x0 + delta, y0        ],
#         [x0,         y0 - delta],
#         [x0 - delta, y0        ],
#         [x0,         y0 + delta]
#     ])
    
    
#     # Plot diamond
#     ax.plot(diamond[:, 0], diamond[:, 1], 0, 'black', linewidth=0.3)
    
#     # Custom x and y ticks
#     ax.set_xticks([x0 - delta, x0, x0 + delta])
#     ax.set_xticklabels([r'$x_0 - \delta$', r'$x_0$', r'$x_0 + \delta$'])
    
#     ax.set_yticks([y0 - delta, y0, y0 + delta])
#     ax.set_yticklabels([r'$y_0 - \delta$', r'$y_0$', r'$y_0 + \delta$'])
    
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
    
#     # Set abstract tick labels based on \delta
#     cb.set_ticks([0, 0.25])
#     cb.set_ticklabels([r'$0$', r'$\delta^2/4$'])
#     plt.show()
    


# #%% domains

# if plot_article_figures:
        
    
#     N=5
#     faces_T = list_faces_from_N(N, method='triangles')
#     faces_P = list_faces_from_N(N, method='polygons')
    
#     # plot_faces_3d(faces_P)
    
#     faces_plus, faces_minus = list_faces_from_N_DC(N)
    
#     point = (0.3,0.4)
#     is_in_polygon = [Path(f[:,:2]).contains_point(point) for f in faces_P]
#     selected_polygon = faces_P[np.where(is_in_polygon)[0][0]]
#     is_triangle_in_polygon = [Path(selected_polygon[:,:2]).contains_point(tuple(f[:,:2].mean(axis=0))) for f in faces_T]
    
    
#     fig, ax = plt.subplots(1,2,figsize=(10,5),dpi=150)
    
#     ax[0].set_xlim([0,1])
#     ax[0].set_ylim([0,1])
#     ax[0].set_xlabel('x')
#     ax[0].set_ylabel('y')
#     j=0
#     k=0
#     ax[0].plot([-1,2],[j/N+1,j/N-2],'k')
#     ax[0].plot([-1,2],[-1-1+j/N,2-1+j/N],'k--')
#     ax[0].add_collection(PolyCollection([selected_polygon[:,:2]], facecolors='C2', alpha=0.6))
#     for j in range(2*N):
#         ax[0].plot([-1,2],[j/N+1,j/N-2],'k')
#         ax[0].plot([-1,2],[-1-1+j/N,2-1+j/N],'k--')
#     ax[0].set_aspect('equal', adjustable='box')
#     ax[0].set_title('"DC" and "polygon" representation of $g_n$')
#     ax[0].legend(['Boundary between linear domains of $g_n^+$',
#                   'Boundary between linear domains of $g_n^-$',
#                   'A linear domain of $g_n$'])
    
    
#     ax[1].set_xlim([0,1])
#     ax[1].set_ylim([0,1])
#     ax[1].set_xlabel('x')
#     ax[1].set_ylabel('y')
#     faceCollection = PolyCollection([face[:,:2] for face in faces_T], facecolors='#FF000000', edgecolors='k')
#     ax[1].add_collection(PolyCollection([selected_polygon[:,:2]], facecolors='C2', alpha=0.6))
#     ax[1].add_collection(faceCollection)
#     ax[1].set_aspect('equal', adjustable='box')
#     ax[1].set_title('"triangle" representation of $g_n$')
#     ax[1].legend(['Domain of a pair of co-planar pieces'])
    
#     fig.tight_layout()

# #%% 3d plots

# if plot_article_figures:
    
    
#     fig = plt.figure(figsize=(10,4),dpi=150)
#     ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#     faceCollection = Poly3DCollection(faces_P,shade=False,facecolors='C2',edgecolors='k',alpha=0.5)
#     ax1.add_collection3d(faceCollection)
#     ax1.set_title('$g_n$')
    
#     ax2 = fig.add_subplot(1, 2, 2, projection='3d')
#     faceCollection = Poly3DCollection(faces_plus,shade=False,facecolors='C1',edgecolors='k',alpha=0.5)
#     ax2.add_collection3d(faceCollection)
#     faceCollection = Poly3DCollection(faces_minus,shade=False,facecolors='C0',edgecolors='k',alpha=0.5)
#     ax2.add_collection3d(faceCollection)
#     ax2.set_title('$g_n^+$ and $g_n^-$')
#     ax2.legend(['$g_n^+$','$g_n^-$'])
    
#     for ax in [ax1,ax2]:
#         ax.set_xlabel('x')
#         ax.set_ylabel('y')
#         ax.set_zlabel('z')
#         ax.view_init(elev=30, azim=-45)
    
#     fig.tight_layout()
   
# #%% plot graphs       

# if plot_article_figures:
        
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
    
#     x = np.linspace(0, 1, 50) # 100 points between 0 and 1 for the x-axis
#     y = np.linspace(0, 1, 50) # 100 points between 0 and 1 for the y-axis
#     X, Y = np.meshgrid(x, y)
#     Z = X * Y
    
#     ax.plot_surface(X, Y, Z, cmap='viridis')
    
#     # 6. Set labels and title for clarity
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title('z = xy')
    
    
#     target_error = 0.01
#     N = int(np.ceil(1/(4*np.sqrt(target_error))))
    
#     N = 8
#     eps = 1e-3/N
    
#     faces = []
#     for j in range(2*N):
#         for k in range(2*N):
#             x = (j-k)/(2*N) + 0.5
#             y = (j+k+1)/(2*N) - 0.5
#             if (x>=0) and (x<=1) and (y>=0) and (y<=1):
#                 pts = []
#                 for delta in [(-1,0),(0,-1),(1,0),(0,1)]:
#                         xx = x + delta[0]/(2*N)
#                         yy = y + delta[-1]/(2*N)
#                         if (xx>=0-eps) and (xx<=1+eps) and (yy>=0-eps) and (yy<=1+eps):
#                             pts.append([xx,yy,xx*yy])
#                 pts = np.array(pts)
#                 faces.append(pts)
                
#     faces_triang = []
#     for face in faces:
#         if len(face)==3:
#             faces_triang.append(face)
#         else:
#             x,y = face.mean(axis=0)[0:2]
#             if (y>=x - eps) ^ (x+y>=1-eps):
#                 face = face[face[:,0].argsort(),:]
#             else:
#                 face = face[face[:,1].argsort(),:]
#             faces_triang.append(face[0:3,:])
#             faces_triang.append(face[1:4,:])
            
    
#     x_plus = []
#     y_plus = []
#     xy_plus = []
#     x_minus = []
#     y_minus = []
#     xy_minus = []
    
#     for j in range(2*N+1):
#         tmp_x = []
#         tmp_y = []
#         tmp_xy = []
#         for k in range(2*N+1):
#             x = (j-k)/(2*N) + 0.5
#             y = (j+k)/(2*N) - 0.5
#             if (x>=0-eps) and (x<=1+eps) and (y>=0-eps) and (y<=1+eps):
#                 tmp_x.append(x)
#                 tmp_y.append(y)
#                 tmp_xy.append(x*y)
#         x_plus.append(tmp_x)
#         y_plus.append(tmp_y)
#         xy_plus.append(tmp_xy)
        
#     for k in range(2*N+1):
#         tmp_x = []
#         tmp_y = []
#         tmp_xy = []
#         for j in range(2*N+1):
#             x = (j-k)/(2*N) + 0.5
#             y = (j+k)/(2*N) - 0.5
#             if (x>=0-eps) and (x<=1+eps) and (y>=0-eps) and (y<=1+eps):
#                 tmp_x.append(x)
#                 tmp_y.append(y)
#                 tmp_xy.append(x*y)
#         x_minus.append(tmp_x)
#         y_minus.append(tmp_y)
#         xy_minus.append(tmp_xy)
    
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
#     faceCollection = Poly3DCollection(faces_triang,shade=False,facecolors='C1',edgecolors='k',alpha=0.5)
#     ax.add_collection3d(faceCollection)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title(f'N = {N}, {len(faces_triang)} pieces, error = {100/(4*N)**2:.3f}%')
            
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
#     faceCollection = Poly3DCollection(faces,shade=False,facecolors='C1',edgecolors='k',alpha=0.5)
#     ax.add_collection3d(faceCollection)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title(f'N = {N}, {len(faces)} pieces, error = {100/(4*N)**2:.3f}%')
    
    
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d',computed_zorder=False)
#     faceCollection = Poly3DCollection(faces,shade=False,facecolors='C1',alpha=0.5)
#     for n in range(len(x_plus)):
#         ax.plot(x_plus[n],y_plus[n],xy_plus[n],'C2')
#     for n in range(len(x_minus)):
#         ax.plot(x_minus[n],y_minus[n],xy_minus[n],'C3')
#     ax.plot([0,1,1,0,0],[0,0,1,1,0],[0,0,1,0,0],'k')
#     ax.add_collection3d(faceCollection)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     ax.set_title(f'N = {N}, {2*N} + {2*N} regions, error = {100/(4*N)**2:.3f}%')
    
#     print(f'Relative error: {100/(4*N)**2:.3f}%')


# #%%


# if plot_article_figures:
    
        
#     # Custom formatter to remove trailing ".0"
#     class NoDecimalScalarFormatter(ScalarFormatter):
#         def _set_format(self):
#             self.format = "%d"  # integer formatting only
    
#     # n from 1 to 32
#     # n = np.arange(1, 33)
#     n = 2**np.arange(6)
    
#     # Functions from the table (linear constraints)
#     triangle = 16*n**2 + 4
#     polygon = 10*n*(n+1) + 4
#     dc = 14*n + 9
#     triangle_log = 16*n**2 + 2*np.ceil(2*np.log2(2*n)) + 4
#     polygon_log = 10*n*(n+1) + 2*np.ceil(np.log2(2*n*(n+1))) + 4
#     dc_log = 14*n + 4*np.ceil(np.log2(2*n)) + 9
    
#     # Continuous variables
#     triangle_cont = 12*n**2 + 1
#     polygon_cont = 6*n*(n+1) + 1
#     dc_cont = 12*n + 3
#     triangle_log_cont = 16*n**2 + 1
#     polygon_log_cont = 8*n*(n+1) + 1
#     dc_log_cont = 16*n + 3
    
#     # Binary variables
#     triangle_bin = 4*n**2
#     polygon_bin = 2*n*(n+1)
#     dc_bin = 4*n
#     triangle_log_bin = np.ceil(2*np.log2(2*n))
#     polygon_log_bin = np.ceil(np.log2(2*n*(n+1)))
#     dc_log_bin = 2*np.ceil(np.log2(2*n))

    
#     # Approximation error
#     error = 1/(16*n**2)
    
#     # Create the plot
#     fig, ax1 = plt.subplots(1,3,figsize=(10,4),dpi=150)
#     plt.subplots_adjust(wspace=0.0)
    
#     # # Primary y-axis: linear constraints
#     # ax1.plot(n, triangle, 'o-', label='Triangle')
#     # ax1.plot(n, polygon, 's-', label='Polygon')
#     # ax1.plot(n, dc, '^-', label='DC')
#     # ax1.plot(n, triangle_log, 'o--', label='Triangle (LogEnc)')
#     # ax1.plot(n, polygon_log, 's--', label='Polygon (LogEnc)')
#     # ax1.plot(n, dc_log, '^--', label='DC (LogEnc)')
    
#     # Primary y-axis: linear constraints
#     l1, = ax1[0].plot(n, triangle, '--', color='C0', label='Triangle')
#     l2, = ax1[0].plot(n, polygon, '--', color='C1', label='Polygon')
#     l3, = ax1[0].plot(n, dc, '--', color='C2', label='DC')
#     l4, = ax1[0].plot(n, triangle_log, '-', color='C0', label='Triangle (LogEnc)')
#     l5, = ax1[0].plot(n, polygon_log, '-', color='C1', label='Polygon (LogEnc)')
#     l6, = ax1[0].plot(n, dc_log, '-', color='C2', label='DC (LogEnc)')
    
#     ax1[0].set_xscale('log', base=2)
#     ax1[0].set_yscale('log')
#     ax1[0].set_xlabel('n')
#     ax1[0].set_ylabel('Number of constraints, variables')
#     ax1[0].set_ylim([1,10**5])
    
#     ax1[0].set_title('Linear constraints')
    
#     # Apply the integer-only formatter
#     formatter = NoDecimalScalarFormatter()
#     formatter.set_scientific(False)
#     ax1[0].xaxis.set_major_formatter(formatter)
#     # ax1[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=10))
#     ax1[0].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax1[0].yaxis.set_major_formatter(formatter)
    

#     # Secondary y-axis: approximation error
#     ax2 = ax1[0].twinx()
#     l7, = ax2.plot(n, error, 'C3', label='Error')
#     ax2.set_yscale('log')
#     # ax2.set_ylabel('Maximum approximation error')
#     ax2.set_ylim([10**(-5),10**(0)])
#     ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax2.yaxis.set_major_formatter(formatter)
#     ax2.set_yticklabels([])
    
#     # ax1[0].legend(loc='upper center')
#     ax1[0].grid(True, which="both", ls="--", alpha=0.5)
    
    
#     # ax2.legend(loc='lower left')
    
    
#     ## second figure
    
    
#     # Primary y-axis: linear constraints
#     ax1[1].plot(n, triangle_bin, '--', color='C0', label='Triangle')
#     ax1[1].plot(n, polygon_bin, '--', color='C1', label='Polygon')
#     ax1[1].plot(n, dc_bin, '--', color='C2', label='DC')
#     ax1[1].plot(n, triangle_log_bin, '-', color='C0', label='Triangle (LogEnc)')
#     ax1[1].plot(n, polygon_log_bin, '-', color='C1', label='Polygon (LogEnc)')
#     ax1[1].plot(n, dc_log_bin, '-', color='C2', label='DC (LogEnc)')
    
#     ax1[1].set_xscale('log', base=2)
#     ax1[1].set_yscale('log')
#     ax1[1].set_xlabel('n')
#     # ax1[1].set_ylabel('Linear Constraints')
#     ax1[1].set_ylim([1,10**5])
    
#     ax1[1].set_title('Binary variables')
    
#     # Apply the integer-only formatter
#     formatter = NoDecimalScalarFormatter()
#     formatter.set_scientific(False)
#     ax1[1].xaxis.set_major_formatter(formatter)
#     ax1[1].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax1.yaxis.set_major_formatter(formatter)
#     ax1[1].set_yticklabels([])
    
#     # Secondary y-axis: approximation error
#     ax2 = ax1[1].twinx()
#     ax2.plot(n, error, 'C3', label='Error')
#     ax2.set_yscale('log')
#     # ax2.set_ylabel('Maximum approximation error')
#     ax2.set_ylim([10**(-5),10**(0)])
#     ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax2.yaxis.set_major_formatter(formatter)
#     ax2.set_yticklabels([])
    
#     # ax1[1].legend(loc='upper center')
#     ax1[1].grid(True, which="both", ls="--", alpha=0.5)
    
    
#     # ax2.legend(loc='lower left')
    
#     ## third figure
    
#     # Primary y-axis: linear constraints
#     ax1[2].plot(n, triangle_cont, '--', color='C0', label='Triangle')
#     ax1[2].plot(n, polygon_cont, '--', color='C1', label='Polygon')
#     ax1[2].plot(n, dc_cont, '--', color='C2', label='DC')
#     ax1[2].plot(n, triangle_log_cont, '-', color='C0', label='Triangle (LogEnc)')
#     ax1[2].plot(n, polygon_log_cont, '-', color='C1', label='Polygon (LogEnc)')
#     ax1[2].plot(n, dc_log_cont, '-', color='C2', label='DC (LogEnc)')
    
#     ax1[2].set_xscale('log', base=2)
#     ax1[2].set_yscale('log')
#     ax1[2].set_xlabel('n')
#     # ax1[1].set_ylabel('Linear Constraints')
#     ax1[2].set_ylim([1,10**5])
#     ax1[2].set_yticklabels([])
    
#     ax1[2].set_title('Continuous variables')
    
#     # Apply the integer-only formatter
#     formatter = NoDecimalScalarFormatter()
#     formatter.set_scientific(False)
#     ax1[2].xaxis.set_major_formatter(formatter)
#     ax1[2].yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax1.yaxis.set_major_formatter(formatter)
    
#     # Secondary y-axis: approximation error
#     ax2 = ax1[2].twinx()
#     ax2.plot(n, error, 'C3', label='Error')
#     ax2.set_yscale('log')
#     ax2.set_ylabel('Maximum approximation error')
#     ax2.set_ylim([10**(-5),10**(0)])
#     ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[], numticks=1))
#     # ax2.yaxis.set_major_formatter(formatter)
    
#     # ax1[1].legend(loc='upper center')
#     ax1[2].grid(True, which="both", ls="--", alpha=0.5)
    
    
#     # ax2.legend(loc='lower left')
    
#     # Make legend belong to the figure
#     fig.legend(
#         handles=[l1,l2,l3,l4,l5,l6,l7],  # first 9 entries for 3x3
#         loc='lower center',
#         bbox_to_anchor=(0.5, +0.05),
#         fontsize=9,
#         ncol=7
#     )
    
#     plt.subplots_adjust(bottom=0.28)
    
#     # plt.title('Linear Constraints vs n (Log-Log) with Approximation Error')
#     # plt.tight_layout()
#     plt.show()
    
