#!/bin/bash

# Check if --weighted_node2vec flag is passed
WEIGHTED_FLAG=""
if [[ "$*" == *--weighted_node2vec* ]]; then
  WEIGHTED_FLAG="--weighted_node2vec"
fi

# Define datasets
DATASETS=(
  fb_forum
  ia_contact
  ia_contacts_hypertext_2009
  ia_email_eu
  ia_enron_employees
  ia_radoslaw_email
  soc_sign_bitcoin_alpha
  wiki_elections
)

EDGE_OPERATORS="all"
if [[ -n "$1" ]]; then
  EDGE_OPERATORS="$1"
fi

# Loop through datasets and execute commands
for DATASET in "${DATASETS[@]}"; do
  python index.py --dataset $DATASET --walk_bias Exponential --initial_edge_bias Uniform --edge_operator $EDGE_OPERATORS $WEIGHTED_FLAG
done
