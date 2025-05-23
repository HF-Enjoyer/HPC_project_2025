import matplotlib.pyplot as plt

# Define the log file
log_file = "execution_times.log"

# Read the log file
with open(log_file, 'r') as file:
    lines = file.readlines()

# Parse the log file
n_steps = None
n_eq = None
n_traj_values = []
time_values = []

for line in lines:
    if "N_STEPS" in line and "N_EQ" in line and "N_TRAJ" in line:
        try:
            parts = line.split()
            # Debug: Print the parts to see what's being split
            #print(parts)

            n_steps = int(parts[1].strip(','))
            n_eq = int(parts[3].strip(','))
            n_traj = int(parts[5].strip(','))
            time = float(parts[7])

            n_traj_values.append(n_traj)
            time_values.append(time)
        except IndexError as e:
            print(f"Error parsing line: {line}. Error: {e}")
        except ValueError as e:
            print(f"Error converting value in line: {line}. Error: {e}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_traj_values, time_values, marker='o')
plt.title(f"Execution Time vs. Number of Trajectories\nN_STEPS: {n_steps}, N_EQ: {n_eq}")
plt.xlabel("Number of Trajectories (N_TRAJ)")
plt.ylabel("Execution Time (seconds)")
plt.grid(True)

# Save the plot to a file
plot_file = "execution_time_plot.png"
plt.savefig(plot_file)
print(f"Plot saved to {plot_file}")
