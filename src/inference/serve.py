import io
import os
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from PIL import Image
from src.dataset import get_transforms
from src.model import get_model

app = FastAPI(title="CIFAR-10 Image Classification API")

# Global variables for model and device
model: nn.Module | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms(train=False)

# CIFAR-10 Class Labels
CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
)

def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform = get_inference_transform()

def load_model_checkpoint():
    global model
    checkpoint_path = Path(os.getenv("CHECKPOINT_PATH", "checkpoints/classifier_v1.pt"))
    
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        # Load model architecture and weights
        model = get_model(architecture="resnet18", num_classes=10)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(device)
        model.eval()
        print(f"Successfully loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    load_model_checkpoint()


@app.get("/health")
def health_check():
    """
    Returns 200 OK if model is loaded and ready for inference.
    """
    if model is None:
        # Try reloading in case checkpoint appeared after container start
        if not load_model_checkpoint():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model checkpoint not loaded"
            )
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accepts an image file and returns predicted class probabilities.
    """
    if model is None:
        if not load_model_checkpoint():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded"
            )

    try:
        # Read and preprocess input image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor_image = transform(pil_image).unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(tensor_image)
            probabilities = torch.softmax(outputs, dim=1)[0]

        # Structure prediction output
        prob_dict = {
            CLASSES[i]: round(probabilities[i].item(), 4)
            for i in range(len(CLASSES))
        }
        top_pred_idx = torch.argmax(probabilities).item()

        return {
            "filename": image.filename,
            "prediction": CLASSES[top_pred_idx],
            "confidence": round(probabilities[top_pred_idx].item(), 4),
            "probabilities": prob_dict,
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image processing error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)