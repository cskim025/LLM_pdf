from safety_llm_pipeline import demo_pipeline
import numpy as np

normal = np.load("historical_normal.npy")  # shape (n_samples, n_features)
demo_pipeline("./safety_pdfs", normal)
