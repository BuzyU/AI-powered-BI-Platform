
import pandas as pd
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.layers.l2_classification.content_classifier import classify_content, ContentType

def test_classifier():
    # 1. Sales Data
    sales_df = pd.DataFrame({
        'transaction_id': range(10),
        'revenue': [100.0] * 10,
        'customer_email': ['a@b.com'] * 10,
        'product_sku': ['SKU-123'] * 10
    })
    
    print("\n--- Testing Sales Data ---")
    result = classify_content(sales_df, "sales_jan_2024.csv")
    print(f"Content Type: {result['content_type']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Description: {result['description']}")
    
    # 2. ML Training Data
    ml_df = pd.DataFrame({
        'feature_1': range(10),
        'feature_2': range(10),
        'target_class': [0, 1] * 5
    })
    
    print("\n--- Testing ML Data ---")
    result = classify_content(ml_df, "training_dataset.csv")
    print(f"Content Type: {result['content_type']}")
    print(f"Confidence: {result['confidence']}")
    
    # 3. Survey Data
    survey_df = pd.DataFrame({
        'respondent_id': range(10),
        'q1_satisfaction': [5] * 10,
        'q2_feedback': ['Great' for _ in range(10)]
    })
    
    print("\n--- Testing Survey Data ---")
    result = classify_content(survey_df, "survey_results.csv")
    print(f"Content Type: {result['content_type']}")

if __name__ == "__main__":
    test_classifier()
