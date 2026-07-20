# # Create a temporary file to store the script's output
OUTPUT_FILE=$(mktemp)

# # Run the first script and redirect all its output to the temporary file
python /home/nadinebadie/loco-mujoco-clean/examples/jax_rl_curriculum/experiment.py --config-name conf_P1 > "$OUTPUT_FILE"

# Display the script's full output to the console for you to see all prints
cat "$OUTPUT_FILE"

# Extract the CHECKPOINT_PATH from the output file
SAVE_PATH=$(grep "CHECKPOINT_PATH" "$OUTPUT_FILE" | cut -d: -f2)

# Check if a path was captured
if [ -z "$SAVE_PATH" ]; then
    echo "Error: Could not determine save path from the first script."
    # Clean up the temporary file
    rm "$OUTPUT_FILE"
    exit 1
fi

echo "Captured save path: $SAVE_PATH"

# Clean up the temporary file
rm "$OUTPUT_FILE"


python /home/nadinebadie/loco-mujoco-clean/examples/jax_rl_curriculum/experiment.py --config-name conf_P2_H2000 checkpoint_path="$SAVE_PATH"

python /home/nadinebadie/loco-mujoco-clean/examples/jax_rl_curriculum/experiment.py --config-name conf_P2 checkpoint_path="$SAVE_PATH"
