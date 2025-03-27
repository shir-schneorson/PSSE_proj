import numpy as np


def generate_Ybus(bus_count, line_data):
    """
    Generates Ybus from line data (assuming no transformers).

    Parameters:
    bus_count : int
        Number of buses.
    line_data : ndarray
        Nx5 matrix [from, to, r, x, b_shunt].

    Returns:
    Ybus : ndarray
        Bus admittance matrix (bus_count x bus_count).
    """
    # Initialize Ybus
    Ybus = np.zeros((bus_count, bus_count), dtype=complex)

    # Loop over each transmission line
    for k in range(line_data.shape[0]):
        from_bus = int(line_data[k, 0]) - 1  # Convert to zero-based index
        to_bus = int(line_data[k, 1]) - 1  # Convert to zero-based index
        r = line_data[k, 2]  # Resistance
        x = line_data[k, 3]  # Reactance
        b = line_data[k, 4]  # Line shunt susceptance

        # Compute series admittance
        y = 1 / (r + 1j * x)

        # Update Ybus
        Ybus[from_bus, from_bus] += y + 1j * b / 2
        Ybus[to_bus, to_bus] += y + 1j * b / 2
        Ybus[from_bus, to_bus] -= y
        Ybus[to_bus, from_bus] -= y

    return Ybus
