from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import aiofiles
from pathlib import Path
import uuid
from datetime import datetime

import config
from model import get_detector
from gradcam import GradCAMExplainer
from video_processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(
    title="DeepTrust API",
    description="Deepfake Detection API with Explainability",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving heatmaps
app.mount("/results", StaticFiles(directory=str(config.RESULTS_DIR)), name="results")

# Global instances (lazy-loaded)
detector = None
gradcam_explainer = None
video_processor = None


# ✅ LAZY LOAD SERVICES (CRITICAL FIX)
def get_services():
    global detector, gradcam_explainer, video_processor

    if detector is None:
        detector = get_detector()

    if gradcam_explainer is None:
        gradcam_explainer = GradCAMExplainer(detector.model, detector.device)

    if video_processor is None:
        video_processor = VideoProcessor()


@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "DeepTrust API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    get_services()  # ✅ ENSURE MODELS LOADED

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_path = config.TEMP_DIR / f"{file_id}{file_extension}"
    heatmap_path = config.RESULTS_DIR / f"{file_id}_heatmap.jpg"

    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        result = detector.predict_from_file(str(temp_path))

        heatmap_result = gradcam_explainer.generate_heatmap(
            str(temp_path),
            str(heatmap_path)
        )

        prediction = result["prediction"]
        confidence = result["confidence"]

        if prediction == "fake":
            verdict = "FAKE"
            explanation = (
                f"High confidence ({confidence}%) deepfake detected."
                if confidence > 80 else
                f"Moderate confidence ({confidence}%) deepfake detected."
            )
        else:
            verdict = "REAL"
            explanation = (
                f"High confidence ({confidence}%) authentic content."
                if confidence > 80 else
                f"Moderate confidence ({confidence}%) authentic content."
            )

        if 45 <= confidence <= 65:
            verdict = "UNCERTAIN"
            explanation = (
                f"Inconclusive result ({confidence}% confidence). "
                "Manual verification recommended."
            )

        response = {
            "success": True,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "probabilities": result["probabilities"],
            "heatmap_url": f"/results/{file_id}_heatmap.jpg" if heatmap_result["success"] else None,
            "file_id": file_id
        }

        temp_path.unlink(missing_ok=True)
        return JSONResponse(content=response)

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        heatmap_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    get_services()  # ✅ ENSURE MODELS LOADED

    if not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="File must be a video")

    file_id = str(uuid.uuid4())
    file_extension = Path(file.filename).suffix
    temp_path = config.TEMP_DIR / f"{file_id}{file_extension}"

    try:
        async with aiofiles.open(temp_path, 'wb') as f:
            content = await file.read()
            await f.write(content)

        result = video_processor.process_video(
            str(temp_path),
            detector.model,
            gradcam_explainer
        )

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Video processing failed"))

        response = {
            "success": True,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "explanation": result["explanation"],
            "frames": result["frames"],
            "total_frames": result["total_frames"],
            "file_id": file_id
        }

        temp_path.unlink(missing_ok=True)
        return JSONResponse(content=response)

    except Exception as e:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.delete("/api/cleanup/{file_id}")
async def cleanup_files(file_id: str):
    try:
        heatmap_path = config.RESULTS_DIR / f"{file_id}_heatmap.jpg"
        heatmap_path.unlink(missing_ok=True)
        return {"success": True, "message": "Files cleaned up"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=False
    )
