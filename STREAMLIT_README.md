# Rakuten Classification Dashboard

Complete Streamlit dashboard for analyzing model predictions and ensemble performance.

## Features

### 1. Home Page
- Project overview
- Quick statistics
- Model summary table

### 2. Model Performance
- Compare all models on accuracy, F1-score, precision, recall
- Visual comparisons with bar charts and heatmaps
- Text vs Image model comparison
- Download metrics as CSV

### 3. Ensemble Analysis
- Combine multiple models (weighted average or voting)
- Adjust model weights
- Compare ensemble vs single models
- Real-time performance metrics

### 4. Sample Explorer
- Examine individual sample predictions
- See how each model predicts specific samples
- Identify correct vs incorrect predictions
- View probability distributions

### 5. Error Analysis
- Confusion matrix visualization
- Confidence distribution analysis
- Most confused class pairs
- Hardest samples identification
- Error overlap between models

### 6. Class Analysis
- Per-class performance metrics
- Identify best/worst performing classes
- Class imbalance analysis
- Multi-model class comparison

### 7. Model Comparison
- Side-by-side model comparison
- Prediction agreement analysis
- Complementary strengths
- Disagreement case studies
- Per-class performance differences

## Installation

Ensure you have the required packages:

```bash
pip install streamlit plotly pandas numpy scikit-learn
```

## Running the Dashboard

From the project root directory:

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your default web browser at http://localhost:8501

## Project Structure

```
DS_rakuten/
├── streamlit_app.py              # Main application
├── pages/                        # Multi-page app pages
│   ├── 1_Model_Performance.py
│   ├── 2_Ensemble_Analysis.py
│   ├── 3_Sample_Explorer.py
│   ├── 4_Error_Analysis.py
│   ├── 5_Class_Analysis.py
│   └── 6_Model_Comparison.py
├── src/
│   └── streamlit_utils/          # Utility modules
│       ├── streamlit_data_loader.py      # Data loading functions
│       └── streamlit_visualizations.py   # Plotting functions
└── artifacts/
    ├── canonical_classes.json    # Class definitions
    └── exports/                  # Model predictions
        ├── camembert_canonical/
        ├── flaubert_canonical/
        ├── xlmr_canonical/
        ├── mdeberta_canonical/
        ├── vit_canonical/
        ├── swin_canonical/
        ├── resnet50_canonical/
        └── lenet_canonical/
```

## Data Requirements

The dashboard requires:

1. Model predictions in `artifacts/exports/*/val.npz`
2. Metadata files in `artifacts/exports/*/val_meta.json`
3. Canonical classes in `artifacts/canonical_classes.json`

All models must have:
- Same split signature: `cf53f8eb169b3531`
- Same classes fingerprint: `cdfa70b13f7390e6`
- Same number of samples: 10,827

## Usage Tips

### Navigation
- Use the sidebar to navigate between pages
- Models are automatically loaded on startup
- All pages share the same model data (cached)

### Model Performance
- Sort models by any metric
- Filter by text/image models
- Download results as CSV

### Ensemble Analysis
- Select 2+ models to combine
- Try different ensemble methods (average vs voting)
- Adjust weights for weighted average
- Compare with best single model

### Sample Explorer
- Enter sample index or use random button
- Select multiple models to visualize
- See which models agree/disagree
- View top-K predicted classes

### Error Analysis
- Select a model to analyze
- View confidence distributions
- Identify most confused classes
- Find highest-confidence errors
- Compare errors with other models

### Class Analysis
- Sort classes by any metric
- Identify best/worst performing classes
- Check class imbalance effects
- Compare class performance across models

### Model Comparison
- Select two models to compare
- View agreement/disagreement statistics
- Find complementary strengths
- Analyze per-class differences
- Get ensemble recommendations

## Performance Notes

- First load may take 10-20 seconds to load all model data
- Data is cached in session state for fast page switching
- Large visualizations may take a few seconds to render
- All computations are done on-demand

## Troubleshooting

### Models not loading
- Check that all .npz and .json files exist in artifacts/exports/
- Verify file permissions
- Check console for error messages

### Incorrect metrics
- Verify all models have matching split signatures
- Check that classes fingerprints match
- Ensure all models have 10,827 samples

### Visualization errors
- Update plotly: `pip install -U plotly`
- Clear browser cache
- Restart Streamlit server

### Performance issues
- Reduce number of models loaded
- Decrease sample sizes in visualizations
- Close other browser tabs

## Keyboard Shortcuts

- `R` - Rerun the app
- `C` - Clear cache
- `Ctrl+C` (in terminal) - Stop server

## Extending the Dashboard

To add new pages:

1. Create a new file in `pages/` with format `N_Page_Name.py`
2. Import required utilities from `src.streamlit_utils`
3. Check session state for loaded models
4. Add your analysis and visualizations

To add new visualizations:

1. Add plotting function to `src/streamlit_utils/streamlit_visualizations.py`
2. Import in your page
3. Call with appropriate data

## Dependencies

- streamlit >= 1.25.0
- plotly >= 5.14.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0

## License

Part of the Rakuten Product Classification project.
