# NeuroCell GAN - Cell Growth Forecaster for Neuroblastoma (Streamlit Web App)

A web application built using **Streamlit** and pure Python that integrates:
1. **Trained 200-Epoch GAN Model**: [`livecell_gan_200epochs_final.h5`](file:///c:/Users/Arogya%20Mary/Downloads/cse7thsem/cie3/livecell_gan_200epochs_final.h5)
2. **Notebook Pipeline**: Preserves and replicates the preprocessing, segmentation, and evaluation from [`forecasting-the-cell-growth-of-neuroblastoma-cells.ipynb`](file:///c:/Users/Arogya%20Mary/Downloads/cse7thsem/cie3/forecasting-the-cell-growth-of-neuroblastoma-cells.ipynb).
3. **Groq LLM Integration**: Powered by `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` with built-in Dr. NeuroCell AI copilot for automated growth reports and interactive Q&A.

---

## 🚀 Features

- **Microscopy Time-Series Input**:
  - Preloaded LIVECell SH-SY5Y Neuroblastoma benchmark sequences (Exponential & Colony proliferation).
  - Custom upload support for 4 consecutive time-lapse microscopy frames (T1, T2, T3, T4).
- **Computer Vision Segmentation (Otsu & Contours)**:
  - Gaussian blur, inverted histogram equalization, and Otsu thresholding directly adapted from notebook Cell 1.
  - Confluence percentage calculation and contour cell counting.
- **GAN Growth Forecasting**:
  - Predicts future frame (T5 / next growth cycle).
  - Computes net confluence growth ($\Delta\%$), cell division rate, and proliferation velocity.
  - Morphological proliferation heatmap highlighting growth hotspots.
- **Temporal Occlusion Sensitivity**:
  - Displays the model's empirical occlusion sensitivities (Frame 1: 35.8%, Frame 2: 5.9%, Frame 3: 5.1%, Frame 4: 3.0%).
- **Groq LLM Bio-Oncology Copilot**:
  - **Automated Bio-Oncology Report**: Generates clinical and experimental evaluations directly from the forecasted metrics.
  - **Interactive Research Chatbot**: Real-time Q&A regarding cell culture, doubling time, LIVECell benchmark, and GAN spatiotemporal dynamics.

---

## 🏃 How to Run

1. Open PowerShell or Terminal in this folder:
   ```bash
   streamlit run streamlit_app.py
   ```
   *(Or double-click [`run_streamlit.bat`](file:///c:/Users/Arogya%20Mary/Downloads/cse7thsem/cie3/run_streamlit.bat))*

2. The application will open automatically in your browser at:
   `http://localhost:8501`
