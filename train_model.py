"""
train_model.py
One-shot script to:
  1. Generate dataset (if missing)
  2. Train ATS score model
  3. Save model + vectorizer to /models

Run: python train_model.py
"""

import os
import sys
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from generate_dataset import generate_dataset
from src.ats_score import train_ats_model


def main():
    dataset_path = "data/dataset.csv"
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 1. Generate dataset if not present
    if not os.path.exists(dataset_path):
        print("📦 Generating synthetic dataset...")
        df = generate_dataset(400)
        df.to_csv(dataset_path, index=False)
        print(f"✅ Dataset saved → {dataset_path}  ({len(df)} rows)")
    else:
        df = pd.read_csv(dataset_path)
        print(f"📂 Loaded existing dataset → {len(df)} rows")

    # 2. Train model
    print("\n🤖 Training ATS model...")
    model, vectorizer, metrics = train_ats_model(df)

    print("\n📊 Training complete!")
    print(f"   Mean Absolute Error : {metrics['mae']:.2f}")
    print(f"   R² Score            : {metrics['r2']:.3f}")
    print("\n🚀 Ready to launch: streamlit run app/app.py")


if __name__ == "__main__":
    main()
