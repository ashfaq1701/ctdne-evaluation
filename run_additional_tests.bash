#!/bin/bash

# Check if --weighted_node2vec flag is passed
WEIGHTED_FLAG=""
if [[ "$*" == *--weighted_node2vec* ]]; then
  WEIGHTED_FLAG="--weighted_node2vec"
fi

# Define datasets
DATASETS=(
  ia_contact
  ia_contacts_hypertext_2009
  ia_enron_employees
  ia_radoslaw_email
)

# Define walk biases and initial edge biases
WALK_BIASES=(Uniform Linear)
INIT_EDGE_BIASES=(Uniform Linear)

# Loop through combinations and execute commands
for DATASET in "${DATASETS[@]}"; do
  for WALK_BIAS in "${WALK_BIASES[@]}"; do
    for INIT_EDGE_BIAS in "${INIT_EDGE_BIASES[@]}"; do
      python index.py --dataset $DATASET --walk_bias $WALK_BIAS --initial_edge_bias $INIT_EDGE_BIAS $WEIGHTED_FLAG
      python index.py --dataset $DATASET --walk_bias $WALK_BIAS --initial_edge_bias $INIT_EDGE_BIAS --edge_operator hadamard $WEIGHTED_FLAG
    done
  done
done
