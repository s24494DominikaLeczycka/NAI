import numpy as np

def knapsack_brute_force(file_path):
    # Load data from file
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    # The first line is the capacity of the knapsack
    capacity = int(lines[0])

    # The remaining lines are the weights and values of the items
    weights = []
    values = []
    for line in lines[1:]:
        w, v = map(int, line.split())
        weights.append(w)
        values.append(v)

    # Convert lists to numpy arrays for efficient computation
    weights = np.array(weights)
    values = np.array(values)

    # Number of items
    n = len(weights)

    # Store the best value and corresponding characteristic vector
    best_value = 0
    best_vector = None

    # Iterate over all possible combinations
    for i in range(2**n):
        # Create a characteristic vector using modulo operation
        characteristic_vector = []
        temp = i
        for _ in range(n):
            characteristic_vector.append(temp % 2)
            temp //= 2
        characteristic_vector.reverse()

        # Convert to numpy array
        characteristic_vector = np.array(characteristic_vector)

        # Compute total weight and value of this combination
        total_weight = np.sum(weights * characteristic_vector)
        total_value = np.sum(values * characteristic_vector)

        # If the total weight is less than or equal to the capacity and total value is greater than the best value
        # Update the best value and corresponding characteristic vector
        if total_weight <= capacity and total_value > best_value:
            best_value = total_value
            best_vector = characteristic_vector

    # Print the best value and corresponding characteristic vector
    print(f'Best Value: {best_value}')
    print(f'Best Characteristic Vector: {best_vector}')


# Run the function
knapsack_brute_force('knapsack_data.txt')
