# Secretary

A real-time speech-to-text application with WebRTC audio streaming and advanced audio processing capabilities.

## Features

- Real-time audio streaming using WebRTC
- Advanced audio processing pipeline
  - High-pass filtering
  - Dynamic range compression
  - Noise gating
  - Audio level normalization
- Speech recognition using Google's Speech-to-Text API
- Debug audio recording for troubleshooting
- WebSocket-based communication

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
uvicorn app.main:app --reload
```

4. Open your browser and navigate to `http://localhost:8000`

## Project Structure

```
secretary/
├── app/
│   ├── main.py           # FastAPI application
│   └── static/
│       └── index.html    # Frontend interface
├── debug_audio/          # Debug audio recordings
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Audio Processing

The application implements several audio processing techniques to improve speech recognition:

1. High-pass filter (80Hz) to reduce low-frequency noise
2. Dynamic range compression (4:1 ratio) for consistent levels
3. Noise gate with smooth transitions
4. Real-time audio level monitoring
5. Automatic gain control

## Debugging

Audio files are saved in the `debug_audio` directory with timestamps:
- `raw_*.wav`: Raw audio before processing
- `normalized_*.wav`: Audio after normalization
- `success_*.wav`: Audio that was successfully recognized

## License

MIT License 