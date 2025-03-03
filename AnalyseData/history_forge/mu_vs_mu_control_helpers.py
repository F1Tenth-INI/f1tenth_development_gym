import re


def decode_mu_from_filename(filename: str):
    """
    Extracts the mu and mu_control parameters from a filename.

    The function looks for a pattern of the form:
        "mu_<number>_mu_control_<number>_"
    in the provided filename. It returns the parameters as floats.

    Parameters:
        filename (str): The name (or path) of the file.

    Returns:
        tuple or None: Returns a tuple (mu, mu_control) if extraction
        and conversion are successful. Otherwise, returns None.
    """
    # Define a regex pattern that finds two groups of numbers:
    # one after 'mu_' and one after 'mu_control_'.
    pattern = re.compile(r"mu_([0-9.]+)_mu_control_([0-9.]+)_")

    # Use re.findall to capture all occurrences of the pattern.
    # In case of multiple matches, we use the last one.
    matches = re.findall(pattern, filename)
    if not matches:
        print(f"WARNING: Could not extract parameters from {filename}")
        return None

    mu_str, mu_control_str = matches[-1]

    # Convert the extracted strings to floats.
    try:
        mu_val = float(mu_str)
        mu_control_val = float(mu_control_str)
    except ValueError:
        print(f"WARNING: Conversion error in file {filename}")
        return None

    return mu_val, mu_control_val
