from dataclasses import dataclass
import enum
from ..logging_config import *
from .product_linearization_simple import do_mathopt_simple_product_linearization
from cpwllib.implementation import do_product_linearization

@enum.unique
class ProductLinearizationMethod(enum.Enum):
    SINGLE = 'Single'
    TRIANGLES = 'Triangles'
    POLYGONS = 'Polygons'
    SUM_OF_CONVEX = 'Sum of Convex'


@dataclass
class ProductLinearizationConfig:
    method: ProductLinearizationMethod = ProductLinearizationMethod.SINGLE
    N: int = 10  # Number of segments for piecewise linearization


def do_mathopt_product_linearization(
    config: ProductLinearizationConfig,
    model,
    X_var,
    Y_var,
    Z_var
):
    """
    Perform product linearization in MathOpt.

    Args:
        method: The product linearization method to use.
        model: The MathOpt model.
        X_var: The first variable.
        Y_var: The second variable.
        Z_var: The product variable.
    """
    variables = []
    constraints = []

    method = config.method

    if method == ProductLinearizationMethod.SINGLE:
        constraints = do_mathopt_simple_product_linearization(model, X_var, Y_var, Z_var)
    elif method == ProductLinearizationMethod.TRIANGLES:
        q_vars = do_product_linearization(
            model,
            partition_method='triangles',
            N=config.N,
            X_var=X_var,
            Y_var=Y_var,
            Z_var=Z_var
        )
    elif method == ProductLinearizationMethod.POLYGONS:
        q_vars = do_product_linearization(
            model,
            partition_method='polygons',
            N=config.N,
            X_var=X_var,
            Y_var=Y_var,
            Z_var=Z_var
        )
    elif method == ProductLinearizationMethod.SUM_OF_CONVEX:
        q_vars = do_product_linearization(
            model,
            partition_method='sum of convex',
            N=config.N,
            X_var=X_var,
            Y_var=Y_var,
            Z_var=Z_var
        )

    return constraints, variables
