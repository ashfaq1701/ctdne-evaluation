#!/bin/bash

# Check if --weighted_node2vec flag is passed
WEIGHTED_FLAG=""
if [[ "$*" == *--weighted_node2vec* ]]; then
  WEIGHTED_FLAG="--weighted_node2vec"
fi

AUC_BY_PROBS_FLAG=""
if [[ "$*" == *--auc_by_probs* ]]; then
  AUC_BY_PROBS_FLAG="--auc_by_probs"
fi

PICKER_TYPE_FLAG=""
if [[ "$*" == *--use_weight_based_picker* ]]; then
  PICKER_TYPE_FLAG="--use_weight_based_picker"
fi

# Define datasets
DATASETS=(
  ia_contact
  ia_contacts_hypertext_2009
  ia_enron_employees
  ia_radoslaw_email
)

EDGE_OPERATOR="best"
if [[ -n "$1" ]]; then
  EDGE_OPERATOR="$1"
fi

# Define walk biases and initial edge biases
WALK_BIASES=(Uniform Linear)
INIT_EDGE_BIASES=(Uniform Linear)

# Loop through combinations and execute commands
for DATASET in "${DATASETS[@]}"; do
  for WALK_BIAS in "${WALK_BIASES[@]}"; do
    for INIT_EDGE_BIAS in "${INIT_EDGE_BIASES[@]}"; do
      python index.py --dataset $DATASET --walk_bias $WALK_BIAS --initial_edge_bias $INIT_EDGE_BIAS --edge_operator $EDGE_OPERATOR $WEIGHTED_FLAG $AUC_BY_PROBS_FLAG $PICKER_TYPE_FLAG
    done
  done
done
