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
  fb_forum
  ia_contact
  ia_contacts_hypertext_2009
  ia_email_eu
  ia_enron_employees
  ia_radoslaw_email
  soc_sign_bitcoin_alpha
  wiki_elections
)

EDGE_OPERATOR="best"
if [[ -n "$1" ]]; then
  EDGE_OPERATOR="$1"
fi

# Loop through datasets and execute commands
for DATASET in "${DATASETS[@]}"; do
  python index.py --dataset $DATASET --walk_bias Exponential --initial_edge_bias Uniform --edge_operator $EDGE_OPERATOR $AUC_BY_PROBS_FLAG $PICKER_TYPE_FLAG
done
