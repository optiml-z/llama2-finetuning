#!/bin/bash

# Prompt the user for the EVAL_PATH
read -p "Enter the absolute path to the lm-evaluation-harness: " EVAL_PATH

# Directory containing YAML files
DIR="open_llm_leaderboard"

# Check if the directory exists
if [ ! -d "$DIR" ]; then
    echo "Error: Directory '$DIR' not found."
    exit 1
fi

# Iterate over YAML files in the directory and update them
for YAML_FILE in "$DIR"/*.yaml
do
    if [ -f "$YAML_FILE" ]; then
        sed -i 's|{\$EVAL_PATH}|'"$EVAL_PATH"'|g' "$YAML_FILE"
        echo "Updated $YAML_FILE with EVAL_PATH: $EVAL_PATH"
    fi
done
