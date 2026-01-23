# DeepTrust Backend

FastAPI backend for deepfake detection using EfficientNet-B0 with Grad-CAM explainability.

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Add Your Trained Model**
   - Place your trained model file at: `backend/models/best_efficientnet_b0.pth`
   - The model should be an EfficientNet-B0 trained on deepfake detection

3. **Run the Server**
```bash
cd backend
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```
GET /
GET /health
```

### Image Analysis
```
POST /api/analyze/image
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
  "success": true,
  "verdict": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": 87.5,
  "explanation": "...",
  "probabilities": {
    "fake": 87.5,
    "real": 12.5
  },
  "heatmap_url": "/results/xxx_heatmap.jpg",
  "file_id": "uuid"
}
```

### Video Analysis
```
POST /api/analyze/video
Content-Type: multipart/form-data
Body: file (video file)

Response:
{
  "success": true,
  "verdict": "FAKE" | "REAL" | "UNCERTAIN",
  "confidence": 85.3,
  "explanation": "...",
  "frames": [
    {
      "frameNumber": 1,
      "timestamp": "0:02",
      "verdict": "FAKE",
      "confidence": 82.5,
      "thumbnail": "data:image/jpeg;base64,..."
    },
    ...
  ],
  "total_frames": 6,
  "file_id": "uuid"
}
```

## Directory Structure

```
backend/
├── app.py                 # Main FastAPI application
├── config.py             # Configuration settings
├── model.py              # Model loading and inference
├── gradcam.py            # Grad-CAM explainability
├── video_processor.py    # Video frame extraction and processing
├── requirements.txt      # Python dependencies
├── models/               # Model weights directory
│   └── best_efficientnet_b0.pth
├── uploads/              # Temporary upload storage
├── temp/                 # Temporary processing files
└── results/              # Generated heatmaps and results
```

## Features

- ✅ Image deepfake detection
- ✅ Video frame-by-frame analysis
- ✅ Grad-CAM explainability heatmaps
- ✅ Face extraction using MediaPipe
- ✅ CORS enabled for frontend integration
- ✅ GPU acceleration support (CUDA)
- ✅ Confidence scoring and verdict classification

## Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224
- **Classes**: 2 (fake, real)
- **Preprocessing**: ImageNet normalization
- **Face Detection**: MediaPipe Face Detection

## Notes

- The server runs on `http://localhost:8000`
- Heatmap images are served at `/results/{file_id}_heatmap.jpg`
- Temporary files are automatically cleaned up
- Video processing samples 6 frames evenly distributed
