#!/bin/bash

EXEC=omp_version.exe
OUTPUT="speedup_results.txt"
N_RUNS=12

echo "Threads    Time(s)" > "$OUTPUT"

for ((i=1; i<=N_RUNS; i++)); do
    echo "Running with $i thread(s)..."
    ./"$EXEC" "$i" > temp_output.txt
    TIME=$(grep "time:" temp_output.txt | awk '{print $2}')
    echo "$i        $TIME" >> "$OUTPUT"
done

rm temp_output.txt
echo "Benchmarking complete. Results saved to $OUTPUT."
