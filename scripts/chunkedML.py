print("Subprocess started", flush=True)

# Import packages / libraries
import xarray as xr
import pickle
import xgboost as xgb
import argparse
from xgboost import XGBRegressor
import sys
import os
import numpy as np
import json

print("Modules imported", flush=True)

# print("Python path:", sys.executable)
print("XGBoost version:", xgb.__version__, flush=True)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

parser = argparse.ArgumentParser(description="Train an XGB model on xarray data.")
parser.add_argument("--X", required=True, help="Path to X_chunk.npy")
parser.add_argument("--y", required=True, help="Path to y_chunk.npy")
parser.add_argument("--model-in", default=None, help="Path to existing model (optional)")
parser.add_argument("--model-out", required=True, help="Path to save updated model")
parser.add_argument("--n-estimators", type=int, default=50, help="Number of trees to add")
parser.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
parser.add_argument("--feature_names", required=True, help="Names of Predictor vars passed to understand feature importance")
args = parser.parse_args()

print("Attempting to read data from chunks", flush=True)

# Load data
X = np.load(args.X)
y = np.load(args.y)

if X.shape[0] != y.shape[0]:
    raise ValueError(f"Sample mismatch: X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

print("Chunks read in", flush=True)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("Attempting to establish the model", flush=True)

if args.model_in is None:
    # First chunk: initialize new model
    regressor = XGBRegressor(tree_method = "hist", device = "cuda", n_estimators=args.n_estimators)

else:
    # Warm start from existing model
    with open(args.model_in, 'rb') as f:
        model = pickle.load(f)
    regressor = model['model']
    regressor.set_params(n_estimators=regressor.n_estimators + args.n_estimators)

print("Attempting to train the model", flush=True)

print("Checking training from chunk: X : " + str(np.shape(X)) , flush=True)
print("Checking training from chunk:  : " + str(np.shape(y)) , flush=True)

if args.feature_names:
    with open(args.feature_names, "r") as f:
        feature_names = json.load(f)

if args.model_in is None:
    # Train
    regressor.fit(X, y)

else:
    # Train
    regressor.fit(X, y)

print("Model trained", flush=True)

model_bundle = {
    "model": regressor,
    "feature_names": feature_names
}

# Save updated model
with open(args.model_out, 'wb') as f:
    pickle.dump(model_bundle, f)

print("Model saved", flush=True)

