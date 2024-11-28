#!/bin/bash

# Directory to watch
WATCH_DIR="cifar-clip/checkpoints"

# Base command to execute
BASE_COMMAND="python cifar_clip.py test --gpu 0 --model"

# Output directories
OUTPUT_DIR="cifar-clip/output"
OUTPUT_DIR_EMA="cifar-clip/output-ema"

# Lists to hold new files
NON_EMA_LIST="/tmp/non_ema_file_list.txt"
EMA_LIST="/tmp/ema_file_list.txt"
: > "$NON_EMA_LIST" # Clear the non-ema file list
: > "$EMA_LIST"     # Clear the ema file list

# Ensure output directories exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR_EMA"

# Function to process a single file
process_file() {
    local FILE="$1"
    echo "Processing file: $FILE"

    # Run the script with the file as input
    $BASE_COMMAND "$FILE"

    # Determine output file name based on the input file
    if [[ "$FILE" == *-ema-*.pth ]]; then
        MODEL_NUM=$(basename "$FILE" | sed -E 's/model-ema-([0-9]+)\.pth/\1/')
        OUTPUT_PATH="$OUTPUT_DIR_EMA/img-$MODEL_NUM.png"
    else
        MODEL_NUM=$(basename "$FILE" | sed -E 's/model-([0-9]+)\.pth/\1/')
        OUTPUT_PATH="$OUTPUT_DIR/img-$MODEL_NUM.png"
    fi

    # Move the generated output to the new location
    if [ -f "$OUTPUT_DIR/img.png" ]; then
        mv "$OUTPUT_DIR/img.png" "$OUTPUT_PATH"
        echo "Output moved to: $OUTPUT_PATH"
    else
        echo "Error: Expected output file $OUTPUT_DIR/img.png not found!"
    fi
}

# Background process to poll and process files
process_files_from_lists() {
    while true; do
        if [[ -s "$NON_EMA_LIST" ]]; then
            FILE=$(head -n 1 "$NON_EMA_LIST") # Get the first file in the non-ema list
            sed -i '1d' "$NON_EMA_LIST"      # Remove the first line from the non-ema list
            process_file "$FILE"
        elif [[ -s "$EMA_LIST" ]]; then
            FILE=$(head -n 1 "$EMA_LIST") # Get the first file in the ema list
            sed -i '1d' "$EMA_LIST"       # Remove the first line from the ema list
            process_file "$FILE"
        else
            sleep 1 # Avoid busy-waiting
        fi
    done
}

# Start processing files in the background
process_files_from_lists &
PROCESS_PID=$! # Capture the background process PID

# Cleanup on exit
cleanup() {
    echo "Stopping background process..."
    kill $PROCESS_PID
    rm -f "$NON_EMA_LIST" "$EMA_LIST" # Remove the temporary file lists
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Check for existing files and add them to the appropriate list
echo "Checking for existing files in: $WATCH_DIR"
for FILE in "$WATCH_DIR"/*; do
    if [[ "$FILE" == *-ema-*.pth ]]; then
        echo "$FILE" >> "$EMA_LIST"
    elif [ -f "$FILE" ]; then
        echo "$FILE" >> "$NON_EMA_LIST"
    fi
done

# Check if inotify-tools is installed
if ! command -v inotifywait &> /dev/null; then
    echo "inotifywait is required but not installed. Install it with 'sudo apt install inotify-tools' (or equivalent)."
    cleanup
fi

# Watch the directory for new files and add them to the appropriate list
echo "Watching directory: $WATCH_DIR"
inotifywait -m -e create --format '%w%f' "$WATCH_DIR" | while read NEW_FILE; do
    if [[ "$NEW_FILE" == *-ema-*.pth ]]; then
        echo "$NEW_FILE" >> "$EMA_LIST"
    else
        echo "$NEW_FILE" >> "$NON_EMA_LIST"
    fi
done
