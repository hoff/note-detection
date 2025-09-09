# Setup Instructions

## Prerequisites

- Python 3.8+ (tested with Python 3.13)
- macOS, Linux, or Windows
- Audio input device (microphone) for real-time detection

## Virtual Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd note-detection
```

### 2. Create Virtual Environment
```bash
python -m venv magenta_env
```

### 3. Activate Virtual Environment

**macOS/Linux:**
```bash
source magenta_env/bin/activate
```

**Windows:**
```bash
magenta_env\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Required Models

The project requires TensorFlow Lite models for note detection. Download the required model:

```bash
# Download the main TFLite model (76MB)
curl -O https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_no_offset_uni.tflite

# Optional: Download alternative model (103MB)
curl -O https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite
```

## Running the Applications

### Python Real-time Detection
```bash
source magenta_env/bin/activate  # Activate environment
python realtime_web.py --model_path onsets_frames_wavinput_no_offset_uni.tflite --web_output
```

Access the web interface at: http://localhost:5000

### Static Web Demos (TensorFlow.js)
Start a local web server:
```bash
# Python 3
python -m http.server 8000

# Or Python 2
python -m SimpleHTTPServer 8000
```

Then access:
- http://localhost:8000/tensorflowjs_note_detection.html - Full TensorFlow.js demo
- http://localhost:8000/simple_tensorflowjs_demo.html - Simple TensorFlow.js demo  
- http://localhost:8000/mp3_tensorflowjs_demo.html - MP3 file processing demo
- http://localhost:8000/converted_model_demo.html - Converted model demo

## System Dependencies

### macOS
```bash
# Install PortAudio for PyAudio
brew install portaudio

# If you get compilation errors during pip install
export CPPFLAGS=-I$(brew --prefix portaudio)/include
export LDFLAGS=-L$(brew --prefix portaudio)/lib
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install portaudio19-dev python3-pyaudio
```

### Windows
PyAudio wheels are available for Windows, but if you encounter issues:
```bash
# Install Visual Studio Build Tools or
# Use conda instead of pip for PyAudio
conda install pyaudio
```

## Troubleshooting

### Common Issues

**1. PyAudio Installation Fails**
```bash
# macOS: Install PortAudio first
brew install portaudio

# Ubuntu/Debian: Install system dependencies
sudo apt install portaudio19-dev

# Windows: Try conda instead
conda install pyaudio
```

**2. TensorFlow Warnings**
TensorFlow may show warnings about GPU support or optimizations. These are generally safe to ignore for CPU usage.

**3. Audio Device Not Found**
Ensure your microphone is connected and permissions are granted. On macOS, check System Preferences > Security & Privacy > Microphone.

**4. Port Already in Use**
If port 5000 or 8000 is busy:
```bash
# For Python app
python realtime_web.py --model_path onsets_frames_wavinput_no_offset_uni.tflite --web_output --port 5001

# For web server
python -m http.server 8001
```

### Performance Tips

- **GPU Acceleration**: TensorFlow will automatically use GPU if available (CUDA on NVIDIA, Metal on Apple Silicon)
- **CPU Optimization**: Close unnecessary applications for better real-time performance
- **Browser Performance**: Chrome/Edge typically perform better than Firefox for TensorFlow.js

## Development

### Project Structure
```
note-detection/
├── magenta_env/              # Virtual environment
├── mp3/                      # Audio files for testing
├── scripts/                  # Conversion and utility scripts
├── tfjs_model/              # TensorFlow.js model files
├── saved_model/             # TensorFlow SavedModel format
├── *.html                   # Web demo applications
├── realtime_web.py          # Main Python application
├── tflite_model.py          # TensorFlow Lite model wrapper
├── requirements.txt         # Python dependencies
└── *.tflite                 # TensorFlow Lite model files
```

### Adding New Features
1. Activate the virtual environment
2. Make changes to Python files
3. Test with both Python and web versions
4. Update requirements.txt if adding new dependencies

## Model Information

- **Input**: Raw audio at 16kHz sample rate
- **Output**: 88-key piano note predictions (A0 to C8)
- **Architecture**: Onsets & Frames transcription model
- **Training**: MAESTRO dataset (classical piano performances)
- **Performance**: ~8-10 FPS on CPU, higher with GPU acceleration