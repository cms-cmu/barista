import argparse
import numpy as np
from typing import List, Union

def combine_expected_limits(limits_list: List[Union[int, float]]) -> float:
    """
    Computes the combined expected upper limit (UL_combined) for N independent
    channels/analyses using the inverse-square approximation:

    1 / (UL_combined)^2 = Sum[1 / (UL_i)^2]
    
    This is the standard approximation for combining sensitivities (expected 
    significances) of independent exclusion limits in particle physics.
    """
    if not limits_list:
        # This check is technically redundant if required=True is used in argparse,
        # but it's good practice for the function itself.
        raise ValueError("The input list of limits cannot be empty.")

    # Convert to a NumPy array for fast, vectorized calculation
    limits = np.array(limits_list, dtype=float)

    # Calculate the sum of the inverse squares: Sum[1 / (UL_i)^2]
    sum_of_inverse_squares = np.sum(1 / (limits**2))

    # Calculate the combined limit: sqrt( 1 / Sum[1 / (UL_i)^2] )
    UL_combined = np.sqrt(1 / sum_of_inverse_squares)

    return UL_combined

def main():
    """
    Parses command-line arguments and prints the combined limit.
    """
    parser = argparse.ArgumentParser(
        description="Compute the combined expected upper limit (UL) for multiple independent analyses using the inverse-square approximation."
    )
    
    # Define the --values argument
    parser.add_argument(
        '--values',
        nargs='+',  # Expects one or more arguments
        type=float, # Converts inputs to floats (handles integers too)
        required=True, # Ensures the user must provide values
        help='A list of expected upper limits (UL_i) from independent channels, separated by spaces.'
    )

    args = parser.parse_args()

    # Get the list of limit values
    limits = args.values
    
    # Ensure all limits are positive
    if any(ul <= 0 for ul in limits):
        raise ValueError("All expected limits must be positive numbers.")

    # Compute the result
    combined_ul = combine_expected_limits(limits)

    # --- Output ---
    print("\n⚛️ Combined Limit Calculation (Inverse-Square Approximation) ⚛️")
    print("-" * 50)
    print(f"Individual Limits (UL_i): {limits}")
    print(f"Number of Channels: {len(limits)}")
    print("-" * 50)
    print(f"Combined Expected Upper Limit: {combined_ul:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    main()