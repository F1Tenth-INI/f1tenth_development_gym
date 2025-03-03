import os


def append_index_to_csv_files(directory):
    """
    Append a zero-padded index to each CSV file in the directory.

    Instead of completely renaming the file (losing its original name),
    this function splits the file name into its base and extension, and
    then appends a dash followed by a three-digit index before the extension.

    The files are processed in sorted order to ensure that the numbering
    follows a predictable sequence.
    """
    # Retrieve and sort all file names in the directory to ensure consistent numbering.
    files = sorted(os.listdir(directory))

    # Filter the list to include only files that end with '.csv'
    csv_files = [f for f in files if f.endswith(".csv")]

    # Debug print: show which CSV files were detected.
    print("CSV files found for processing:")
    for filename in csv_files:
        print(filename)

    # Iterate over the CSV files, appending a zero-padded index to each file name.
    for i, filename in enumerate(csv_files, start=1):
        # Split the filename into the base part and the extension.
        # This is useful in case the base name contains dots.
        root, ext = os.path.splitext(filename)

        # Construct the new file name by appending '-' and a three-digit index.
        # For example, 'data.csv' becomes 'data-001.csv'.
        new_filename = f"{root}_{i:03d}{ext}"

        # Create full paths for renaming.
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)

        # Print the renaming action to help with debugging.
        print(f"Renaming '{old_file}' to '{new_file}'")

        # Perform the renaming.
        os.rename(old_file, new_file)

    print("Renaming completed successfully.")


# Adjusted line to access a file with the proper padded index.
# The formatted_index should be an integer; :03d ensures it is displayed with three digits (e.g., 001, 002, ...)

# Note: Make sure that 'formatted_index' is defined in your context (for example, as an integer) before using this line.

# Example usage: execute renaming if this script is run as the main program.
if __name__ == "__main__":
    # Specify the target directory where the CSV files are located.
    target_directory = "../ExperimentRecordings/Experiments_03_03_2025"

    # Check if the target directory exists before processing.
    if not os.path.isdir(target_directory):
        print(f"Error: The directory '{target_directory}' does not exist.")
    else:
        append_index_to_csv_files(target_directory)
