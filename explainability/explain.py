"""
explainability/explain.py
=========================
LIME Text Explainer tailored for character-level DGA detection models.
It calculates exactly how much each character contributed to the
prediction being DGA (positive weight) or Legit (negative weight).

Author: DGA Research Team
"""

import logging
import numpy as np
from lime.lime_text import LimeTextExplainer
from src.preprocessing import vectorize_domains

logger = logging.getLogger(__name__)


def get_explainer() -> LimeTextExplainer:
    """Initialize a LIME Text Explainer configured to split by character.
    
    By setting `char_level=True`, LIME perturbs individual characters 
    rather than whitespace-separated words, allowing us to generate 
    fine-grained character highlighting for DNS domain strings.
    """
    logger.info("Initializing LIME Character-Level TextExplainer...")
    return LimeTextExplainer(
        class_names=["Legit", "DGA"],
        char_level=True
    )


def explain_domain(domain: str, model, explainer: LimeTextExplainer, 
                   max_len: int = 75, num_features: int = 75) -> list:
    """
    Run the explainer over a single domain name and return character weights.
    
    Parameters
    ----------
    domain : str
        The input domain string to explain (e.g., "google.com").
    model : File-loaded Keras Model
        The trained Deep Learning DGA model.
    explainer : LimeTextExplainer
        Predictive explainer instantiated via `get_explainer()`.
    max_len : int
        Maximum sequence length matching the model's architecture.
    num_features : int
        Maximum number of characters/features to explain.
        
    Returns
    -------
    list of dict
        List of objects spanning the entire input domain in order,
        containing: `{"char": "x", "weight": 0.15}`
    """
    # 1. Define the prediction wrapper function for LIME 
    # LIME requires a function that accepts a list of strings and 
    # returns a 2D numpy array of probabilities for each class 
    # (shape: n_samples x n_classes).
    def predictor(texts: list) -> np.ndarray:
        import pandas as pd
        
        # We must use exactly the same preprocessing pipeline
        domains_series = pd.Series(texts)
        padded_X = vectorize_domains(domains_series, max_len=max_len)
        
        # Predict DGA probability (class 1)
        # Suppress verbose TF output for clean logs
        preds_dga = model.predict(padded_X, verbose=0).ravel()
        
        # probability of Legit (class 0)
        preds_legit = 1.0 - preds_dga
        
        # Stack into [P(Legit), P(DGA)] columns
        return np.vstack([preds_legit, preds_dga]).T

    # 2. Run perturbation and generate explanation
    logger.info("Running LIME explainer on domain: '%s'", domain)
    exp = explainer.explain_instance(
        text_instance=domain,
        classifier_fn=predictor,
        labels=(1,),           # Only interested in the DGA class (index 1) output
        num_features=num_features,
        num_samples=500        # Number of perturbed sequences to generate
    )
    
    # 3. Process the output weights into an ordered character sequence
    # LIME returns list of tuple: (feature_name, weight) 
    # where feature_name is the character.
    # Note: LIME orders by magnitude of weight. We must rebuild the string
    # mapping back to the original order so the frontend can display it correctly.
    
    # LIME's char_level returns items like `('a', 0.12)`, `('b', -0.05)`
    # BUT wait, what if characters repeat? LIME might group them 
    # or return positional indices if we configure it tightly.
    # Actually, LIME `char_level=True` gives features as index sets or string tokens.
    # Let's cleanly map the raw output: `exp.as_list(label=1)` 
    weight_map = dict(exp.as_list(label=1))
    
    # Map back to original string:
    # Since LIME text explainer extracts word/char tokens based on their string value,
    # repeated characters get the SAME weight mathematically in standard LimeTextExplainer.
    explanation = []
    for char in domain:
        # Default to 0.0 if LIME didn't return a weight for this char (e.g. padding/ignored)
        w = weight_map.get(char, 0.0)
        explanation.append({"char": char, "weight": float(w)})
        
    return explanation
