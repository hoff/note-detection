#!/bin/bash
# Quick setup script for note-detection project

set -e  # Exit on error

echo "🎵 Setting up Note Detection Environment"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "✅ Python version: $python_version"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv magenta_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source magenta_env/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Download model if it doesn't exist
MODEL_FILE="onsets_frames_wavinput_no_offset_uni.tflite"
if [ ! -f "$MODEL_FILE" ]; then
    echo "⬇️  Downloading TensorFlow Lite model..."
    curl -O https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/$MODEL_FILE
    echo "✅ Model downloaded: $(ls -lh $MODEL_FILE | awk '{print $5}') "
else
    echo "✅ Model already exists: $MODEL_FILE"
fi

# Check if MP3 directory exists
if [ ! -d "mp3" ]; then
    echo "📁 Creating mp3 directory for test audio files..."
    mkdir mp3
    echo "   Place your MP3 files in the ./mp3/ directory for testing"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "  source magenta_env/bin/activate"
echo "  python realtime_web.py --model_path $MODEL_FILE --web_output"
echo ""
echo "Then visit: http://localhost:5000"
echo ""
echo "For web demos, start a web server:"
echo "  python -m http.server 8000"
echo "Then visit: http://localhost:8000/"