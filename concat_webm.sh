#!/bin/bash

# Directory containing the WebM files
AUDIO_DIR="audio_logs"

# Create a temporary file list
LIST_FILE="filelist.txt"

# Get the timestamp from the first file (assuming all files are from the same conversation)
TIMESTAMP=$(ls ${AUDIO_DIR}/conversation_*_user_0.webm 2>/dev/null | head -n 1 | grep -o 'conversation_[0-9]*_[0-9]*')

if [ -z "$TIMESTAMP" ]; then
    echo "No WebM files found in ${AUDIO_DIR}"
    exit 1
fi

# Create the file list
echo "Creating file list..."
> ${LIST_FILE}  # Clear the file if it exists

# Find all user WebM files and sort them numerically
for file in $(ls ${AUDIO_DIR}/conversation_*_user_*.webm | sort -V); do
    echo "file '${file}'" >> ${LIST_FILE}
done

# Output file
OUTPUT_FILE="${AUDIO_DIR}/${TIMESTAMP}_concatenated.webm"

echo "Concatenating files to ${OUTPUT_FILE}..."

# Concatenate the files
ffmpeg -f concat -safe 0 -i ${LIST_FILE} -c copy -y ${OUTPUT_FILE}

# Check if concatenation was successful
if [ $? -eq 0 ]; then
    echo "Successfully concatenated files to ${OUTPUT_FILE}"
else
    echo "Error concatenating files"
    exit 1
fi

# Clean up
rm ${LIST_FILE}

echo "Done!" 