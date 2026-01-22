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
    