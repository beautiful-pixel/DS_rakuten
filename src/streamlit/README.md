# Streamlit Application - Notebooks Download

This Streamlit application allows users to download the entire `notebooks` folder as a ZIP archive.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

From the repository root directory, run:

```bash
streamlit run src/streamlit/app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Features

- **Download Notebooks Folder**: Creates a ZIP archive of the entire `notebooks` folder
- **File Count Display**: Shows the number of files in the notebooks folder
- **Simple Interface**: Easy-to-use interface with a single button to prepare and download the archive

## Usage

1. Click the "🔽 Prepare Download" button
2. Wait for the ZIP archive to be created
3. Click the "📥 Download notebooks.zip" button to download the archive

The downloaded file will be named `notebooks.zip` and will contain all files from the notebooks folder.
