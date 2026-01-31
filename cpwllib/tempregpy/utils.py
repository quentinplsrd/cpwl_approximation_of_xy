import shutil
import os
from typing import Optional, Tuple

from ortools.math_opt.python import mathopt


def copy_file_to_destination(source_filename, destination_directory):
    # Get the current directory
    current_directory = os.getcwd()

    # Construct the full path of the source file (assuming it's in the parent directory)
    source_path = os.path.join(os.path.dirname(current_directory), source_filename)

    # Construct the full path of the destination directory
    destination_path = os.path.join(current_directory, destination_directory)

    try:
        # Create the destination directory if it doesn't exist
        os.makedirs(destination_path, exist_ok=True)

        # Copy the file to the destination directory
        shutil.copy(source_path, destination_path)
        print(f"File '{source_path}' successfully copied to '{destination_path}'")
    except FileNotFoundError:
        print(f"Error: File '{source_path}' not found.")
    except PermissionError:
        print(f"Error: Permission denied when accessing '{destination_path}'")


def solve_result_gap(result: mathopt.SolveResult) -> Optional[float]:
    """Compute the relative gap between primal objective and best bound from an OR-Tools SolveResult.

    Returns a non-negative float representing the relative gap (|primal - bound| / max(1, |primal|, |bound|)).
    If a gap cannot be computed (missing bound, infeasible/unbounded or result is None), returns None.

    The function handles both minimization and maximization by using the sign of
    result.objective_value and result.best_objective_bound where appropriate. OR-Tools
    provides `best_objective_bound` which is a dual bound for the problem: for minimization
    it is a lower bound, for maximization it is an upper bound.
    """
    if result is None:
        return None

    # Check result status: if infeasible or unbounded, no meaningful gap
    if hasattr(result, 'status'):
        # OR-Tools SolveResult.status is an enum; treat INFEASIBLE, INVALID, or UNBOUNDED as no-gap
        try:
            st = result.status
            if st in (mathopt.SolveResultStatus.INFEASIBLE, mathopt.SolveResultStatus.UNBOUNDED, mathopt.SolveResultStatus.UNKNOWN):
                return None
        except Exception:
            # If status is not comparable, ignore
            pass

    # Extract primal objective
    try:
        primal = float(result.objective_value)
    except Exception:
        return None

    # Extract best bound
    best_bound = None
    if hasattr(result, 'best_objective_bound'):
        try:
            bb = result.best_objective_bound
            if bb is not None:
                best_bound = float(bb)
        except Exception:
            best_bound = None

    if best_bound is None:
        return None

    # Relative gap: |primal - bound| / max(1.0, |primal|, |bound|)
    denom = max(1.0, abs(primal), abs(best_bound))
    gap = abs(primal - best_bound) / denom
    return gap


# Example usage:
# source_filename = "file_to_copy.txt"  # Replace with your source file name
# destination_directory = "destination_folder"  # Replace with your destination directory
#
# copy_file_to_destination(source_filename, destination_directory)