import numpy as np
from scipy.spatial import ConvexHull
from ..logging_config import *
from ortools.math_opt.python import mathopt


class HullComputation:
    def __init__(self, points, label):
        self.points = points
        self.label = label
        self.hull, self.plane_equations, self.inequality_signs, self.simplex_centroids = self.compute_hull_and_planes()

    def compute_hull_and_planes(self):
        if (self.points.shape[0] < 4 or
                np.all(self.points == self.points[0]) or
                np.array_equal(self.points[0], self.points[1]) or
                np.array_equal(self.points[2], self.points[3])):
            return self.handle_degenerate_case()

        hull = ConvexHull(self.points)
        plane_equations = []
        inequality_signs = []
        simplex_centroids = []
        centroid = np.mean(self.points, axis=0)

        def normalized_plane_equation(point1, point2, point3):
            v1 = point2 - point1
            v2 = point3 - point1
            normal = np.cross(v1, v2)
            coef_a, coef_b, coef_c = normal
            coef_d = np.dot(normal, point1)
            norm = np.linalg.norm(normal)
            return coef_a / norm, coef_b / norm, coef_c / norm, coef_d / norm

        for simplex in hull.simplices:
            p1, p2, p3 = self.points[simplex]
            a, b, c, d = normalized_plane_equation(p1, p2, p3)
            plane_equations.append((a, b, c, d))
            value_at_centroid = a * centroid[0] + b * centroid[1] + c * centroid[2] - d
            inequality_signs.append("<=" if value_at_centroid < 0 else ">=")
            simplex_centroids.append(np.mean([p1, p2, p3], axis=0))

        return hull, plane_equations, inequality_signs, simplex_centroids

    def handle_degenerate_case(self):
        centroid = np.mean(self.points, axis=0)
        return None, [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 1, 1, 0)], ["<=", "<=", "<=", "<="], [centroid] * 4
        # if len(np.unique(self.points, axis=0)) == 1:
        #     # All points are the same
        #     return None, [(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (1, 1, 1, 0)], ["<=", "<=", "<=", "<="], [centroid] * 4
        #
        # plane_equations = []
        # inequality_signs = []
        # for i in range(4):
        #     plane_equations.append((1, 0, 0, 0))
        #     inequality_signs.append("<=")
        # plane_equations[1] = (-1, 0, 0, 0)
        # inequality_signs[1] = ">="
        # plane_equations[2] = (0, 1, 0, 0)
        # inequality_signs[2] = "<="
        # plane_equations[3] = (0, -1, 0, 0)
        # inequality_signs[3] = ">="
        #
        # return None, plane_equations, inequality_signs, [centroid] * 4

    def print_plane_equations(self):
        label = self.label
        logger.debug(f"Plane Equations for {label}:")
        for i, ((a, b, c, d), sign) in enumerate(zip(self.plane_equations, self.inequality_signs)):
            logger.debug(f"Plane {i}: {a:.3f}x + {b:.3f}y + {c:.3f}z {sign} {d:.3f}")

def gauge_hulls(inputs, rd):
    x = np.linspace(inputs['Q_hulls'].loc[rd, 'Qmin'], inputs['Q_hulls'].loc[rd, 'Qmax'], 2)
    y = np.linspace(inputs['Q_hulls'].loc[rd, 'Tmin'], inputs['Q_hulls'].loc[rd, 'Tmax'], 2)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    hull_computation = HullComputation(points, rd)
    hull_computation.print_plane_equations()
    return hull_computation
    

def gauge_hulls(inputs, rd):
    x = np.linspace(inputs['Q_hulls'].loc[rd, 'Qmin'], inputs['Q_hulls'].loc[rd, 'Qmax'], 2)
    y = np.linspace(inputs['Q_hulls'].loc[rd, 'Tmin'], inputs['Q_hulls'].loc[rd, 'Tmax'], 2)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    hull_computation = HullComputation(points, rd)
    hull_computation.print_plane_equations()
    return hull_computation

def gauge_hulls_time(inputs, rd, i):
    x = np.linspace(inputs['Q_hulls'][i].loc[rd, 'Qmin'], inputs['Q_hulls'][i].loc[rd, 'Qmax'], 2)
    y = np.linspace(inputs['Q_hulls'][i].loc[rd, 'Tmin'], inputs['Q_hulls'][i].loc[rd, 'Tmax'], 2)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    hull_computation = HullComputation(points, rd)
    hull_computation.print_plane_equations()
    return hull_computation


def gen_hulls(x_bound, y_bound):
    x = np.linspace(x_bound[0], x_bound[1], 2)
    y = np.linspace(y_bound[0], y_bound[1], 2)
    X, Y = np.meshgrid(x, y)
    Z = X * Y

    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    hull_computation = HullComputation(points, "not applicable")
    # hull_computation.print_plane_equations()
    return hull_computation
    

def do_mathopt_simple_product_linearization(
    model: mathopt.Model,
    X_var: mathopt.Variable,
    Y_var: mathopt.Variable,
    Z_var: mathopt.Variable
):
    logger.debug("Performing simple product linearization for variables: "
                f"{X_var.name}, {Y_var.name}, {Z_var.name}")
    # print(f"X_var: {X_var}, Y_var: {Y_var}, Z_var: {Z_var}")
    
    hull = gen_hulls(
        (X_var.lower_bound, X_var.upper_bound),
        (Y_var.lower_bound, Y_var.upper_bound)
    )

    constraints = []

    for j in range(0, 4):
        if hull.inequality_signs[j] == '<=':
            constraints.append(
                model.add_linear_constraint(
                    hull.plane_equations[j][0] * X_var +
                    hull.plane_equations[j][1] * Y_var +
                    hull.plane_equations[j][2] * Z_var <=
                    hull.plane_equations[j][3]
                )
            )
        elif hull.inequality_signs[j] == '>=':
            constraints.append(
                model.add_linear_constraint(
                    hull.plane_equations[j][0] * X_var +
                    hull.plane_equations[j][1] * Y_var +
                    hull.plane_equations[j][2] * Z_var >=
                    hull.plane_equations[j][3]
                )
            )
            
    return constraints
    # gauges_hulls = {}
    # for rn in inputs['river_nodes']:
    #     gauges_hulls[rn] = {}
    #     for i in inputs['time_horizon']:
    #         gauges_hulls[rn][i] = gauge_hulls_time(inputs, rn, i)
