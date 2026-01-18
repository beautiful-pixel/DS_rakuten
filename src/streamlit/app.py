import streamlit as st
import os
import zipfile
import io
from pathlib import Path

# Get the path to the notebooks folder
REPO_ROOT = Path(__file__).parent.parent.parent
NOTEBOOKS_FOLDER = REPO_ROOT / "notebooks"

def create_zip_of_folder(folder_path):
    """
    Create a zip file of a folder and return it as bytes.
    
    Args:
        folder_path: Path to the folder to zip
        
    Returns:
        bytes: The zip file as bytes
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Calculate the archive name (relative path)
                arcname = os.path.relpath(file_path, folder_path.parent)
                zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()

def main():
    st.title("DS Rakuten - Notebooks Download")
    
    st.markdown("""
    ### Download Notebooks Folder
    
    This application allows you to download the entire notebooks folder as a ZIP archive.
    """)
    
    # Check if notebooks folder exists
    if not NOTEBOOKS_FOLDER.exists():
        st.error("Notebooks folder not found!")
        return
    
    # Count files in notebooks folder
    file_count = sum(1 for _ in NOTEBOOKS_FOLDER.rglob('*') if _.is_file())
    
    st.info(f"📁 Notebooks folder contains {file_count} file(s)")
    
    # Create download button
    if st.button("🔽 Prepare Download", type="primary"):
        with st.spinner("Creating ZIP archive..."):
            try:
                zip_data = create_zip_of_folder(NOTEBOOKS_FOLDER)
                st.success("ZIP archive created successfully!")
                
                # Provide download button
                st.download_button(
                    label="📥 Download notebooks.zip",
                    data=zip_data,
                    file_name="notebooks.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Error creating ZIP archive: {str(e)}")
    
    st.markdown("---")
    st.markdown("*Project based on cookiecutter data science template*")

if __name__ == "__main__":
    main()
