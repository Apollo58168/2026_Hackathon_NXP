import kagglehub

# Download latest version
path = kagglehub.model_download("intel/midas/tfLite/v2-1-small-lite")

print("Path to model files:", path)
