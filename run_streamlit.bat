@echo off
REM Quick launcher for Streamlit dashboard

echo ======================================================================
echo Rakuten Classification Dashboard
echo ======================================================================
echo.
echo Starting Streamlit server...
echo.
echo The dashboard will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.
echo ======================================================================
echo.

streamlit run streamlit_app.py

pause
