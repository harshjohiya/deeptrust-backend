"""
Test script for DeepTrust API
Run this after starting the backend server to verify it's working correctly
"""

import requests
import sys
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_image_upload():
    """Test image upload endpoint"""
    print("\nTesting image upload endpoint...")
    
    # You need to provide a test image
    test_image = Path("test_image.jpg")
    
    if not test_image.exists():
        print("⚠️  No test image found. Create a 'test_image.jpg' to test this endpoint.")
        return None
    
    try:
        with open(test_image, 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(f"{API_URL}/api/analyze/image", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image analysis successful!")
            print(f"   Verdict: {data.get('verdict')}")
            print(f"   Confidence: {data.get('confidence')}%")
            print(f"   Heatmap URL: {data.get('heatmap_url')}")
            return True
        else:
            print(f"❌ Image analysis failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Image upload error: {e}")
        return False

def test_video_upload():
    """Test video upload endpoint"""
    print("\nTesting video upload endpoint...")
    
    # You need to provide a test video
    test_video = Path("test_video.mp4")
    
    if not test_video.exists():
        print("⚠️  No test video found. Create a 'test_video.mp4' to test this endpoint.")
        return None
    
    try:
        with open(test_video, 'rb') as f:
            files = {'file': ('test.mp4', f, 'video/mp4')}
            response = requests.post(f"{API_URL}/api/analyze/video", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Video analysis successful!")
            print(f"   Verdict: {data.get('verdict')}")
            print(f"   Confidence: {data.get('confidence')}%")
            print(f"   Frames analyzed: {data.get('total_frames')}")
            return True
        else:
            print(f"❌ Video analysis failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Video upload error: {e}")
        return False

def main():
    print("=" * 50)
    print("DeepTrust API Test Suite")
    print("=" * 50)
    
    # Test health check
    health_ok = test_health_check()
    
    if not health_ok:
        print("\n❌ Server is not running or not healthy!")
        print("   Make sure to start the backend server first:")
        print("   cd backend && python app.py")
        sys.exit(1)
    
    # Test image endpoint
    test_image_upload()
    
    # Test video endpoint
    test_video_upload()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()
