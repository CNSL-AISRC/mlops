#!/bin/bash

set -e  # exit immediately if any command fails

COMPONENTS_DIR="components"

echo "🔍 Searching for components in $COMPONENTS_DIR..."

# Loop through all subdirectories
for dir in "$COMPONENTS_DIR"/*; do
  if [[ -d "$dir" ]]; then
    component_name=$(basename "$dir")
    component_file="$dir/$component_name.py"

    if [[ -f "$component_file" ]]; then
      echo "🚀 Building component: $component_name"
      kfp component build "$dir" --component-filepattern "$component_name.py" --push-image
      echo "✅ Successfully built: $component_name"
    else
      echo "⚠️ Skipped: No Python file $component_name.py in $dir"
    fi
  fi
done

echo "🏁 Done building all components!"
