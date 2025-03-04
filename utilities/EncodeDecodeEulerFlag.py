import argparse

def args_fun():
    """
    This function is for use with Euler cluster to differentiate parallel runs with an index.
    Returns:
    """
    parser = argparse.ArgumentParser(description='Generate F1T data.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--euler_experiment_index', default=-1, type=int,
                        help='Additional index. -1 to skip.')
    args = parser.parse_args()

    if args.euler_experiment_index == -1:
        args.euler_experiment_index = None

    return args


euler_index = args_fun().euler_experiment_index


def encode_flag(index_A, index_B, index_C, size_A, size_B):
    """
    Encode three indices into a single integer flag using mixed-radix representation.

    Explanation:
    -------------
    We treat each index as a digit in a positional numeral system where:
      - 'index_A' is the least significant digit (base = size_A),
      - 'index_B' is the next digit (base = size_B),
      - 'index_C' is the most significant digit.

    The encoding formula is:
      flag = index_A + size_A * (index_B + size_B * index_C)

    Why this works:
    ----------------
    - Multiplying 'index_B' by size_A shifts its contribution so that all possible values of A
      (which total size_A) fit before the next change.
    - Similarly, multiplying 'index_C' by (size_A * size_B) ensures that its contribution
      accounts for all combinations of A and B.
    This way, every unique triplet (index_A, index_B, index_C) corresponds to a unique flag.
    """
    return index_A + size_A * (index_B + size_B * index_C)


def decode_flag(flag, size_A, size_B):
    """
    Decode the integer flag back into the three indices.

    Explanation:
    -------------
    We reverse the encoding process step by step:

    1. Extract index_A:
       - Since index_A was added without scaling, it is the remainder when flag is divided by size_A.

    2. Remove the contribution of A:
       - Integer division by size_A effectively "shifts" the flag right, removing index_A's part.

    3. Extract index_B:
       - Now the least significant digit of the shifted flag corresponds to index_B (modulo size_B).

    4. Extract index_C:
       - Finally, integer division by size_B gives the remaining value, which is index_C.
    """
    index_A = flag % size_A  # Peel off A's contribution (non-obvious: using modulus to isolate the lowest digit)
    flag //= size_A  # Remove the A part so B and C are left
    index_B = flag % size_B  # The next digit for B is now the remainder modulo size_B
    index_C = flag // size_B  # The rest corresponds to C
    return index_A, index_B, index_C


def get_max_flag(size_A, size_B, size_C):
    """
    Calculate the maximum valid flag for the given vector sizes.

    Explanation:
    -------------
    - There are size_A * size_B * size_C total combinations.
    - Since indexing starts at 0, the maximum flag is (total combinations - 1).

    For example, if:
      size_A = 4, size_B = 5, size_C = 6,
    then the total number of flags is 4 * 5 * 6 = 120,
    so the maximum flag is 120 - 1 = 119.
    """
    return size_A * size_B * size_C - 1
