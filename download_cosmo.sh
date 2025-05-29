#!/bin/bash
set -e

FINAL_DATA_DIR="/scratch-shared/scur2687/cosmo_dataset"
ZIP_FILE_NAME="cosmo_data.zip"
FULL_ZIP_PATH="$FINAL_DATA_DIR/$ZIP_FILE_NAME"
ZIP_URL='https://zenodo.org/api/records/11479419/files-archive'

echo "Starting download and unzip process..."
date

echo "Creating final data directory (if it doesn't exist): $FINAL_DATA_DIR"
mkdir -p "$FINAL_DATA_DIR"

if ! command -v curl &> /dev/null; then
    echo "Error: curl is not installed."
    exit 1
fi
if ! command -v unzip &> /dev/null; then
    echo "Error: unzip is not installed."
    exit 1
fi

echo "Downloading $ZIP_URL to $FULL_ZIP_PATH..."
curl -L -o "$FULL_ZIP_PATH" "$ZIP_URL"

if [ $? -ne 0 ]; then
    echo "Error: Download failed. Check URL or network."
    exit 1
fi
echo "Download complete."

echo "Unzipping $FULL_ZIP_PATH directly in $FINAL_DATA_DIR..."
unzip -q "$FULL_ZIP_PATH" -d "$FINAL_DATA_DIR"

if [ $? -ne 0 ]; then
    echo "Error: Unzip failed. The zip file is still at $FULL_ZIP_PATH"
    exit 1
fi
echo "Unzip complete."

echo "Cleaning up downloaded zip file: $FULL_ZIP_PATH..."
rm -f "$FULL_ZIP_PATH"

echo "Process complete. Data is in $FINAL_DATA_DIR (after zip removal)"
date
