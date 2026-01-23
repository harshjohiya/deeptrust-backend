import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
import cv2
from pathlib import Path
import config

class GradCAMExplainer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Target the last convolutional layer
        self.target_layers = [model.conv_head]
        self.cam = GradCAM(model=model, target_layers=self.target_layers)
        
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    
    def generate_heatmap(self, image_path: str, output_path: str) -> dict:
        """Generate Grad-CAM heatmap for the given image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                pred_class = outputs.argmax(dim=1).item()
                confidence = torch.nn.functional.softmax(outputs, dim=1)[0][pred_class].item()
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = self.cam(
                input_tensor=input_tensor,
                targets=targets
            )[0]
            
            # Prepare image for overlay
            img_array = np.array(image.resize(config.IMAGE_SIZE))
            img_normalized = img_array.astype(np.float32) / 255.0
            
            # Create visualization
            visualization = show_cam_on_image(
                img_normalized,
                grayscale_cam,
                use_rgb=True
            )
            
            # Save heatmap
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(visualization).save(output_path)
            
            return {
                "success": True,
                "heatmap_path": str(output_path),
                "prediction": config.CLASS_NAMES[pred_class],
                "confidence": round(confidence * 100, 2)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_heatmap_from_tensor(self, image_tensor: torch.Tensor, original_image: np.ndarray) -> np.ndarray:
        """Generate heatmap from tensor (for video frames)"""
        try:
            # Get prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                pred_class = outputs.argmax(dim=1).item()
            
            # Generate Grad-CAM
            targets = [ClassifierOutputTarget(pred_class)]
            grayscale_cam = self.cam(
                input_tensor=image_tensor,
                targets=targets
            )[0]
            
            # Resize original image to match model input
            img_resized = cv2.resize(original_image, config.IMAGE_SIZE)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Create visualization
            visualization = show_cam_on_image(
                img_normalized,
                grayscale_cam,
                use_rgb=True
            )
            
            return visualization
            
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            return original_image
