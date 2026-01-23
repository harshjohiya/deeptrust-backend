import cv2
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
import config
from typing import List, Dict, Optional
import base64
from io import BytesIO
from PIL import Image

class VideoProcessor:
    def __init__(self):
        # Initialize MediaPipe face detector with error handling
        self.face_detector = None
        self.mp_face = None
        
        try:
            import mediapipe as mp
            self.mp_face = mp.solutions.face_detection
            self.face_detector = self.mp_face.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            print("✅ MediaPipe face detection initialized successfully")
        except (AttributeError, ImportError) as e:
            print(f"⚠️ MediaPipe initialization failed: {e}")
            print("⚠️ Video processing will work without face detection (using center crop)")
        
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.MEAN, std=config.STD)
        ])
    
    def sample_frames(self, video_path: str, num_frames: int = None) -> tuple[List[np.ndarray], List[float]]:
        """Extract evenly spaced frames from video"""
        if num_frames is None:
            num_frames = config.VIDEO_SAMPLE_FRAMES
            
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            cap.release()
            return [], []
        
        # Sample frame indices
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int) \
            if total_frames > num_frames else range(total_frames)
        
        frames = []
        timestamps = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                # Calculate timestamp
                timestamp = idx / fps if fps > 0 else 0
                timestamps.append(timestamp)
        
        cap.release()
        return frames, timestamps
    
    def extract_face_mediapipe(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract and crop face from image using MediaPipe"""
        if self.face_detector is None:
            return None
            
        try:
            # Process with MediaPipe (expects RGB)
            results = self.face_detector.process(image)
            
            if not results.detections:
                return None
            
            # Get first detection
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w, _ = image.shape
            
            # Convert relative coordinates to absolute
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop face
            face = image[y1:y2, x1:x2]
            
            if face.size == 0:
                return None
            
            return face
        except Exception as e:
            print(f"⚠️ MediaPipe face extraction failed: {e}")
            return None
    
    def extract_face_center_crop(self, image: np.ndarray) -> np.ndarray:
        """Fallback: Extract center crop from image"""
        h, w, _ = image.shape
        
        # Use center square crop
        size = min(h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        y2 = y1 + size
        x2 = x1 + size
        
        face = image[y1:y2, x1:x2]
        return face
    
    def extract_face(self, image: np.ndarray) -> np.ndarray:
        """Extract and crop face from image (with fallback)"""
        # Try MediaPipe first
        face = self.extract_face_mediapipe(image)
        
        # Fallback to center crop if MediaPipe fails or unavailable
        if face is None:
            face = self.extract_face_center_crop(image)
        
        # Resize to model input size
        face_resized = cv2.resize(face, config.IMAGE_SIZE)
        return face_resized
    
    def process_frame_for_model(self, frame: np.ndarray) -> torch.Tensor:
        """Process frame for model input"""
        # Extract face
        face = self.extract_face(frame)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(face)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        return tensor
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', image_bgr)
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    
    def process_video(self, video_path: str, model, explainer=None) -> Dict:
        """Process entire video and return frame-by-frame analysis"""
        try:
            # Extract frames
            frames, timestamps = self.sample_frames(video_path, config.MAX_FRAMES)
            
            if not frames:
                return {
                    "success": False,
                    "error": "Could not extract frames from video"
                }
            
            frame_results = []
            predictions = []
            confidences = []
            
            for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                # Process frame
                face_crop = self.extract_face(frame)
                
                # Prepare for model
                pil_image = Image.fromarray(face_crop)
                tensor = self.transform(pil_image).unsqueeze(0)
                
                # Move to CPU (Hugging Face Spaces uses CPU)
                tensor = tensor.cpu()
                
                # Predict
                with torch.no_grad():
                    outputs = model(tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    pred_class = predicted.item()
                    conf_score = confidence.item() * 100
                    
                    predictions.append(pred_class)
                    confidences.append(conf_score)
                
                # Generate thumbnail with optional heatmap
                thumbnail = face_crop
                if explainer:
                    try:
                        thumbnail = explainer.generate_heatmap_from_tensor(tensor, face_crop)
                    except Exception as e:
                        print(f"⚠️ Error generating heatmap for frame {idx}: {e}")
                
                # Convert timestamp to readable format
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                time_str = f"{minutes}:{seconds:02d}"
                
                # Determine verdict
                verdict = "FAKE" if pred_class == 0 else "REAL"
                if 45 <= conf_score <= 65:
                    verdict = "UNCERTAIN"
                
                frame_results.append({
                    "frameNumber": idx + 1,
                    "timestamp": time_str,
                    "verdict": verdict,
                    "confidence": round(conf_score, 2),
                    "thumbnail": self.image_to_base64(thumbnail)
                })
            
            # Calculate overall verdict
            fake_count = sum(1 for p in predictions if p == 0)
            real_count = sum(1 for p in predictions if p == 1)
            avg_confidence = sum(confidences) / len(confidences)
            
            if fake_count > real_count:
                final_verdict = "FAKE"
                explanation = f"Analysis of {len(frames)} frames detected manipulation in {fake_count} frames. Inconsistencies in facial features and temporal artifacts suggest synthetic content."
            elif real_count > fake_count:
                final_verdict = "REAL"
                explanation = f"Analysis of {len(frames)} frames shows consistent authentic features in {real_count} frames. No significant manipulation artifacts detected."
            else:
                final_verdict = "UNCERTAIN"
                explanation = f"Analysis inconclusive. Equal distribution of authentic and synthetic indicators across {len(frames)} frames."
            
            return {
                "success": True,
                "verdict": final_verdict,
                "confidence": round(avg_confidence, 2),
                "explanation": explanation,
                "frames": frame_results,
                "total_frames": len(frames)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }