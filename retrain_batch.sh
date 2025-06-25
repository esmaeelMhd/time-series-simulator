#!/bin/bash

# Define the different values for the variable
values=(1 2 3 4)

# Loop through each value and run the Python script with it as an argument
for value in "${values[@]}"
do
   python retrain_models.py --experiment "$value"
done