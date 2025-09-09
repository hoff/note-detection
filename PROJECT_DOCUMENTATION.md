# Real-Time Note Detection Project

## Project Overview

This project provides real-time polyphonic musical note detection using Google's
Magenta TensorFlow models, with a beautiful waterfall visualization displaying
detected notes in a browser interface.

## Project Genesis & Development History

### Initial Goal

The project started with a simple objective: implement local real-time note
detection using Magenta's onsets_frames_transcription models to experiment with
musical AI.

### Evolution Timeline

1. **Basic Setup Phase**:
   - Cloned Magenta repository for model access
   - Set up Python virtual environment with TensorFlow dependencies
   - Downloaded TensorFlow Lite model
     (`onsets_frames_wavinput_no_offset_uni.tflite`)
   - Got basic terminal-based note detection working

2. **Web Integration Phase**:
   - Created WebSocket streaming from Python backend to browser
   - Built initial HTML interface with piano-key visualization
   - Implemented real-time data streaming with JSON protocol

3. **Performance Optimization Phase**:
   - Identified and fixed 1 FPS bottleneck (was batching time slices)
   - Optimized to process individual time slices for ~8-10 FPS updates
   - Added comprehensive debugging and monitoring

4. **UI Evolution Phase**:
   - Removed piano glow effects for cleaner bar chart visualization
   - Implemented threshold-based filtering to match clean terminal output
   - Added comprehensive settings panel with real-time controls

5. **Advanced Visualization Phase**:
   - Replaced static bars with canvas-based waterfall animation
   - Implemented smooth scrolling with fade trails
   - Added double-buffered rendering for smooth 60fps performance

## Architecture Overview

### System Components

```
┌─────────────────┐    WebSocket    ┌──────────────────┐
│   Python Backend│◄───────────────►│ Browser Frontend │
│                 │   ws://8765      │                  │
│ ┌─────────────┐ │                  │ ┌──────────────┐ │
│ │ Audio Input │ │                  │ │Canvas Renderer│ │
│ └─────────────┘ │                  │ └──────────────┘ │
│ ┌─────────────┐ │                  │ ┌──────────────┐ │
│ │TensorFlow   │ │                  │ │Settings Panel│ │
│ │Lite Model   │ │                  │ └──────────────┘ │
│ └─────────────┘ │                  │ ┌──────────────┐ │
│ ┌─────────────┐ │                  │ │ Note History │ │
│ │4x Worker    │ │                  │ └──────────────┘ │
│ │Processes    │ │                  └──────────────────┘
│ └─────────────┘ │
└─────────────────┘
```

### Python Backend (`realtime_web.py`)

**Core Components:**

- **Audio Recorder**: Captures microphone input at 16kHz sample rate
- **TensorFlow Lite Model**: Processes 2048-sample frames with 75% overlap
- **Worker Pool**: 4 parallel processes for real-time inference
- **WebSocket Server**: Streams results to browser at 8765

**Data Flow:**

1. Audio captured in 128ms frames (2048 samples @ 16kHz)
2. Frames processed by TensorFlow Lite model (onsets_frames_transcription)
3. Model outputs frame/onset/velocity probabilities for 88 piano keys
4. Thresholding applied: `frame_prob > 0.3 OR onset_prob > 0.5`
5. Results streamed via WebSocket as JSON

**Key Files:**

- `realtime_web.py`: Main application with WebSocket integration
- `audio_recorder.py`: Audio capture and preprocessing
- `tflite_model.py`: TensorFlow Lite model wrapper
- `start_web_demo.py`: Simple launcher script

### Frontend (`index.html`)

**Core Components:**

- **WebSocket Client**: Receives real-time note data
- **Canvas Renderer**: 60fps waterfall visualization with double-buffering
- **Settings Panel**: Real-time control of thresholds and visualization
- **Performance Monitoring**: FPS counters and connection status

**Visualization Features:**

- **Waterfall Effect**: Notes appear at bottom, scroll upward while fading
- **88-Key Coverage**: Full piano range (A0-C8, MIDI notes 21-108)
- **Color Coding**: Each note has distinct color (A=blue, C=cyan, etc.)
- **Real-time Controls**: Adjustable thresholds, fade speed, noise floor

### Data Protocol

**WebSocket Message Format:**

```javascript
{
  "timestamp": 1693938200.123,        // Unix timestamp
  "notes": [...],                     // Detected notes (above threshold)
  "all_notes": [...],                 // All 88 notes with values
  "serial": 42,                       // Frame sequence number
  "time_slice": 0,                    // Time slice within frame
  "total_notes_detected": 3,          // Count of detected notes
  "has_notes": true,                  // Boolean flag
  "frame_info": {                     // Threshold settings
    "frame_threshold": 0.3,
    "onset_threshold": 0.5
  }
}
```

**Individual Note Object:**

```javascript
{
  "note": "C",                        // Note name: A, A#, B, C, C#, D, D#, E, F, F#, G, G#
  "octave": 4,                        // Octave number (0-7 for 88-key piano)
  "midi_note": 60,                    // MIDI note number (21-108)
  "velocity": 0.85,                   // Velocity/strength (0.0-1.0)
  "frame_prob": 0.42,                 // Frame detection probability
  "onset_prob": 0.67,                 // Onset detection probability  
  "offset_prob": 0.12,                // Offset detection probability
  "is_onset": true,                   // Above onset threshold
  "is_frame": false,                  // Above frame threshold
  "strength": 0.85                    // Max of frame_prob and velocity
}
```

## Magenta Repository Integration

### Current Structure

The project includes the entire Magenta repository as a subdirectory:

```
note-detection/
├── magenta/                    # Full Magenta repository (entire clone)
│   ├── magenta/models/onsets_frames_transcription/
│   ├── setup.py
│   └── ... (thousands of files)
├── magenta_env/               # Python virtual environment
│   ├── bin/activate
│   ├── lib/python3.13/
│   └── ...
├── realtime_web.py            # Main application
├── start_web_demo.py          # Simple launcher script
├── audio_recorder.py          # Audio utilities (copied from Magenta)
├── tflite_model.py           # Model wrapper (copied from Magenta)
├── index.html                # Web interface with waterfall visualization
└── onsets_frames_wavinput_no_offset_uni.tflite  # TensorFlow Lite model (76MB)
```



## Available Models & Alternatives

### Current Model

**`onsets_frames_wavinput_no_offset_uni.tflite`**

- **Type**: Unidirectional LSTM (most efficient for real-time)
- **Input**: Raw audio waveform (16kHz mono)
- **Output**: 88-key piano transcription
- **Latency**: ~20-50ms per frame
- **Size**: ~76MB

### Alternative Models Available

The following alternative models have been downloaded and are available in the project root:

1. **`onsets_frames_wavinput.tflite`** (Original)
   - Bidirectional LSTM (higher accuracy, more latency)
   - Better for offline processing
   - ~100MB
   - Downloaded successfully (~103MB) using: `curl -O https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite`

2. **`onsets_frames_wavinput_no_offset.tflite`**
   - Similar to current but with offset detection
   - Slightly larger and slower
   - Download using `curl -O https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_no_offset.tflite` returned a small index file (255 bytes), indicating the exact file may not exist at that path. The current model (`onsets_frames_wavinput_no_offset_uni.tflite`) is the unidirectional variant. If a bidirectional version with offset is needed, further investigation or model conversion may be required.

3. **Spectrogram-based Models**
   - Process mel spectrograms instead of raw audio
   - Different preprocessing pipeline required

### Model Download Sources

```bash
# From Magenta's releases
wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_no_offset_uni.tflite

# Alternative models
wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput.tflite
```

## Porting to Angular Application

### Architecture Adaptation for Angular

#### 1. Backend Service Integration

```typescript
// Create Angular service
@Injectable({ providedIn: "root" })
export class NoteDetectionService {
  private socket?: WebSocket;
  private noteData$ = new BehaviorSubject<NoteData | null>(null);

  connect(): Observable<NoteData> {
    this.socket = new WebSocket("ws://localhost:8765");
    // ... WebSocket handling
    return this.noteData$.asObservable();
  }
}
```

#### 2. Canvas Component

```typescript
@Component({
  selector: "app-note-waterfall",
  template: '<canvas #canvas width="1200" height="400"></canvas>',
})
export class NoteWaterfallComponent {
  @ViewChild("canvas")
  canvasRef!: ElementRef<HTMLCanvasElement>;
  private ctx!: CanvasRenderingContext2D;

  // Convert vanilla JS animation loop to Angular
  ngOnInit() {
    this.noteDetectionService.connect().subscribe((data) => {
      this.updateNoteValues(data);
    });
    this.startAnimation();
  }
}
```

#### 3. Settings with Angular Signals

```typescript
// Convert settings to Angular signals
export class SettingsService {
  frameThreshold = signal(0.3);
  onsetThreshold = signal(0.5);
  fadeSpeed = signal(0.02);

  // Computed values
  thresholdSettings = computed(() => ({
    frame: this.frameThreshold(),
    onset: this.onsetThreshold(),
  }));
}
```

### Dependency Extraction Strategy

#### Option 1: Minimal Extraction

Keep only essential files:

```
src/
├── services/
│   ├── note-detection.service.ts
│   └── websocket.service.ts
├── components/
│   ├── note-waterfall/
│   └── settings-panel/
└── python-backend/
    ├── realtime_web.py
    ├── audio_recorder.py
    ├── tflite_model.py
    └── model.tflite
```

#### Option 2: Complete Decoupling

Replace Magenta dependencies:

```typescript
// Custom audio processing
class AudioProcessor {
  // Implement microphone access with Web Audio API
  // Or keep Python backend as microservice
}

// Custom model interface
class TensorFlowLiteService {
  // Wrap TensorFlow.js or keep Python backend
}
```

### Migration Checklist

#### Phase 1: Backend Preservation

- [ ] Copy Python backend files to Angular project
- [ ] Extract minimal Magenta dependencies (3-4 files)
- [ ] Test standalone Python execution
- [ ] Document Python virtual environment setup

#### Phase 2: Frontend Migration

- [ ] Create Angular service for WebSocket communication
- [ ] Convert canvas animation to Angular component
- [ ] Migrate settings panel to Angular forms with signals
- [ ] Implement performance monitoring

#### Phase 3: Integration Testing

- [ ] Test WebSocket connectivity
- [ ] Verify real-time performance
- [ ] Test all settings controls
- [ ] Cross-browser compatibility

#### Phase 4: Optional Enhancements

- [ ] Replace Python backend with TensorFlow.js (if needed)
- [ ] Add recording/playback features
- [ ] Implement MIDI export
- [ ] Add chord recognition

### Environment Setup & Development

#### Setting Up the Python Environment

**1. Create Virtual Environment (Python 3.11+ recommended):**

```bash
# Create virtual environment  
python3 -m venv magenta_env

# Activate it
source magenta_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

**2. Install Dependencies:**

```bash
# Core ML dependencies
pip install tensorflow librosa numpy scipy

# Audio dependencies  
pip install pyaudio soundfile

# Web dependencies
pip install websockets asyncio-mqtt

# Utility dependencies
pip install absl-py colorama attr

# System audio dependencies (macOS)
brew install portaudio libsamplerate cmake
```

**3. Download Model:**

```bash
# Download the TensorFlow Lite model (76MB)
wget https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/tflite/onsets_frames_wavinput_no_offset_uni.tflite
```

#### Development Workflow

**Current Setup:**

```bash
# Activate Python environment
source magenta_env/bin/activate

# Run the complete demo (starts HTTP server + note detection)
python start_web_demo.py

# OR run just the note detection backend
python realtime_web.py --model_path onsets_frames_wavinput_no_offset_uni.tflite --web_output
```

**For Angular Integration:**

```bash
# Python backend (terminal 1)
source magenta_env/bin/activate  
python realtime_web.py --model_path onsets_frames_wavinput_no_offset_uni.tflite --web_output

# Angular frontend (terminal 2)
ng serve
```

#### Production Options

1. **Dual Service**: Python microservice + Angular app
2. **Docker Container**: Both services in container
3. **TensorFlow.js**: Full client-side processing (future)

## Performance Characteristics

### Latency Breakdown

- **Audio Buffer**: ~100-200ms (depends on audio device)
- **Frame Processing**: 128ms frame size at 16kHz
- **Model Inference**: ~20-50ms per frame
- **Network Latency**: <5ms (local WebSocket)
- **Rendering**: 16ms (60fps canvas)
- **Total Latency**: ~150-400ms end-to-end

### Resource Usage

- **CPU**: 15-30% (4 worker processes)
- **Memory**: ~200MB (Python + model)
- **Network**: ~50KB/s (WebSocket data)
- **GPU**: None (CPU-only TensorFlow Lite)

## Future Enhancement Opportunities

### Technical Improvements

1. **TensorFlow.js Migration**: Eliminate Python dependency
2. **Web Audio API**: Direct browser audio access
3. **WASM Optimization**: Faster client-side processing
4. **WebRTC**: Lower-latency audio streaming

### Feature Extensions

1. **MIDI Export**: Save detected performances
2. **Chord Recognition**: Identify chord progressions
3. **Multi-instrument Models**: Guitar, drums, etc.
4. **Real-time Effects**: Audio processing integration

### UI/UX Enhancements

1. **Multiple Visualization Modes**: Piano roll, spectrum, etc.
2. **Recording/Playback**: Capture and review sessions
3. **Social Features**: Share performances
4. **Educational Tools**: Music learning applications

This documentation should provide a comprehensive understanding of the project
architecture and clear guidance for integration into your Angular application.
The modular design makes it relatively straightforward to extract the core
functionality while preserving the real-time performance characteristics.

#### on killing zombie processes

1. Found processes using port 8765: lsof -ti:8765 found PIDs 3095 and 53695
2. Killed them: kill -9 3095 53695
3. Verified port is free: lsof -ti:8765 now returns nothing (port available)
4. Cleaned up background processes from our testing

For future reference, here are the commands to handle this:

Quick one-liner to kill all processes on port 8765: 

kill -9 $(lsof -ti:8765)

Or step by step:

# Find what's using the port

lsof -i:8765

# Kill specific processes

kill -9 <PID1> <PID2>

# Or kill all on that port

lsof -ti:8765 | xargs kill -9
