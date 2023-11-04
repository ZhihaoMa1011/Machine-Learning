
**run.sh:**
```bash
#!/bin/bash
# run.sh

# This is a simple shell script to run Perceptron implementations.

echo "Running Perceptron implementations..."

# Run standard Perceptron
echo "Standard Perceptron:"
python perceptron_standard.py

# Run voted Perceptron
echo "Voted Perceptron:"
python perceptron_voted.py

# Run average Perceptron
echo "Average Perceptron:"
python perceptron_average.py

echo "All Perceptron implementations have been executed."
