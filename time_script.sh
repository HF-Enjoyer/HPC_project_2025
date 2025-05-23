#!/bin/bash

# Define the parameters
num_steps=10000
n_eq=1000

# Define the range for num_trajectories
start_trajectories=16
end_trajectories=4096
#step_trajectories=64

# Define the log file
log_file="execution_times.log"

# Clear the log file
> "$log_file"

# Loop over the range of num_trajectories
for ((num_trajectories=start_trajectories; num_trajectories<=end_trajectories; num_trajectories*=2)); do
    echo "Running with N_STEPS: $num_steps, N_EQ: $n_eq, N_TRAJ: $num_trajectories"

    # Measure the execution time
    start_time=$(date +%s.%N)
    ./a.out "$num_steps" "$n_eq" "$num_trajectories"
    end_time=$(date +%s.%N)

    # Calculate the elapsed time
    elapsed_time=$(echo "$end_time - $start_time" | bc)

    # Log the result
    echo "N_STEPS: $num_steps, N_EQ: $n_eq, N_TRAJ: $num_trajectories, Time: $elapsed_time seconds" >> "$log_file"
done

echo "Execution times logged to $log_file"