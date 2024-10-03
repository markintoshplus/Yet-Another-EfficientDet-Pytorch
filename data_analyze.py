import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Function to process a single JSON file
def process_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    iterations = [entry[1] for entry in data]
    losses = [entry[2] for entry in data]
    timestamps = [datetime.fromtimestamp(entry[0]) for entry in data]
    
    return iterations, losses, timestamps

# Get all JSON files in the train_data folder
json_files = [f for f in os.listdir('train_data') if f.endswith('.json')]

# Plot setup
plt.figure(figsize=(12, 6))

# Process each JSON file
for json_file in json_files:
    file_path = os.path.join('train_data', json_file)
    iterations, losses, timestamps = process_json_file(file_path)
    
    # Plot the loss over iterations for this file
    plt.plot(iterations, losses, label=json_file)

# Plot settings
plt.title('Validation Loss over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.yscale('log')  # Using log scale for better visualization
plt.grid(True)
plt.legend()
plt.show()

# Print statistics for each file
for json_file in json_files:
    file_path = os.path.join('train_data', json_file)
    iterations, losses, timestamps = process_json_file(file_path)
    
    print(f"\nStatistics for {json_file}:")
    print(f"Total number of data points: {len(losses)}")
    print(f"Start time: {timestamps[0]}")
    print(f"End time: {timestamps[-1]}")
    print(f"Duration: {timestamps[-1] - timestamps[0]}")
    print(f"Initial loss: {losses[0]:.2f}")
    print(f"Final loss: {losses[-1]:.2f}")
    print(f"Loss reduction: {(1 - losses[-1]/losses[0])*100:.2f}%")