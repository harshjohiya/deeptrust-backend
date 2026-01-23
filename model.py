import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import config

class DeepfakeDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.transform = self._get_transform()
        self.load_model()
    
    def _get_transform(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    
    def load_model(self):
        """Load the trained EfficientNet model"""
        try:
            # Create model architecture
            self.model = timm.create_model(
                config.MODEL_NAME,
                pretrained=True,  # Use pretrained as fallback
                num_classes=config.NUM_CLASSES
            )
            
            # Load trained weights if they exist
            if config.MODEL_PATH.exists():
                try:
                    checkpoint = torch.load(
                        config.MODEL_PATH,
                        map_location=self.device
                    )
                    self.model.load_state_dict(checkpoint["model_state_dict"])
                    print(f"✅ Model loaded from {config.MODEL_PATH}")
                except Exception as load_err:
                    print(f"⚠️ Error loading model weights: {load_err}")
                    print("   Using pretrained model instead")
            else:
                print(f"⚠️ Warning: Model weights not found at {config.MODEL_PATH}")
                print("   Using pretrained model. Upload your trained model for better accuracy.")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input"""
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image_tensor: torch.Tensor) -> dict:
        """Make prediction on preprocessed image"""
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_class = predicted.item()
            conf_score = confidence.item() * 100
            
            return {
                "prediction": config.CLASS_NAMES[pred_class],
                "confidence": round(conf_score, 2),
                "probabilities": {
                    "fake": round(probabilities[0][0].item() * 100, 2),
                    "real": round(probabilities[0][1].item() * 100, 2)
                }
            }
    
    def predict_from_file(self, image_path: str) -> dict:
        """Complete prediction pipeline from file"""
        image_tensor = self.preprocess_image(image_path)
        return self.predict(image_tensor)

# Global instance
detector = None

def get_detector() -> DeepfakeDetector:
    """Get or create detector instance"""
    global detector
    if detector is None:
        detector = DeepfakeDetector()
    return detector
