Publication Figure Generator for PDAC Risk Calculator
This repository contains the source code and required machine learning artifacts for generating a publication-ready, full-page mockup of the PDAC risk calculator.
The 62.py script is a Streamlit-based utility designed to create high-resolution figures (300 DPI TIFF and vector PDF) of the web calculator interface. These generated figures can be directly inserted into a manuscript for journal submission.
📁 Repository Structure
62.py : The Streamlit application to generate and download the publication figure.
requirements.txt : Python dependencies required to run the application. Note: Versions of scikit-learn and numpy are strictly pinned to ensure compatibility with the serialized model.
data/ : Folder containing the trained model and background data.
best_model.pkl : The core Gradient Boosting model.
calibrated_model.pkl : Post-hoc probability calibrator.
training_parameters_mice.pkl : Feature standardization parameters (mean/std).
X_train.npy : Training data background used for SHAP explainer.
🚀 How to Run Locally
Clone the repository:
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:
It is highly recommended to use a virtual environment (e.g., conda or venv) to avoid version conflicts.
pip install -r requirements.txt
Launch the Figure Generator:
streamlit run 62.py
📊 Features & Usage
Once the Streamlit app is running in your browser (usually at http://localhost:8501):
Select a patient case: Choose from the pre-set "High-risk example" or "Low-risk example", or input custom clinical values manually.
Generate Figure: Click the "Generate Publication Figure" button. The app will render a full-page mockup containing:
The input patient measurements panel.
The predicted PDAC probability with risk stratification.
An individualized SHAP waterfall plot for explainability.
Download: Use the buttons provided at the bottom to download the figure in TIFF (300 DPI, LZW lossless compression) or PDF (Vector) format, ready for direct manuscript insertion.
⚠️ Disclaimer
This tool is intended for research use and clinical decision support only. It does not constitute a medical diagnosis and must not replace clinical judgment. The authors are not responsible for any clinical decisions made based on this tool.
