import numpy as np

def knapsack_brute_force(weights, values, capacity):
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
        num = i
        for _ in range(n):
            characteristic_vector.append(num % 2)
            num //= 2
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

# Define weights and values
weights = [10, 20, 30]
values = [60, 100, 120]
capacity = 50

# Run the function
knapsack_brute_force(weights, values, capacity)
