from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.models import ChatRequest, ChatResponse
from app.services.chat import get_chat_response
import os
import tempfile
import subprocess
import speech_recognition as sr
import json
import asyncio
from typing import List, Dict
import logging
import re
import datetime
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCIceCandidate
from aiortc.sdp import candidate_from_sdp
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaStreamError
import av
import traceback
import numpy as np
import wave
import time
from pydub import AudioSegment

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create debug directories
DEBUG_AUDIO_DIR = "debug_audio"
DEBUG_METADATA_DIR = os.path.join(DEBUG_AUDIO_DIR, "metadata")
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)
os.makedirs(DEBUG_METADATA_DIR, exist_ok=True)

def save_debug_audio(audio_data: np.ndarray, sample_rate: int = 16000, prefix: str = "audio", metadata: dict = None) -> str:
    """Save audio data to a WAV file for debugging with metadata."""
    try:
        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{DEBUG_AUDIO_DIR}/{prefix}_{timestamp}.wav"
        metadata_filename = f"{DEBUG_METADATA_DIR}/{prefix}_{timestamp}.json"
        
        # Ensure audio data is in the correct format
        if audio_data.dtype != np.int16:
            # Normalize to int16 range
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = (audio_data / max_val * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)
        
        # Calculate audio statistics
        audio_stats = {
            "min": float(audio_data.min()),
            "max": float(audio_data.max()),
            "mean": float(audio_data.mean()),
            "std": float(audio_data.std()),
            "duration": len(audio_data) / sample_rate,
            "sample_rate": sample_rate,
            "timestamp": timestamp,
            "filename": filename
        }
        
        # Add any additional metadata
        if metadata:
            audio_stats.update(metadata)
        
        # Save audio statistics
        with open(metadata_filename, 'w') as f:
            json.dump(audio_stats, f, indent=2)
        
        # Log audio statistics
        logger.info(f"Saving audio file {prefix}_{timestamp}.wav with stats: {json.dumps(audio_stats, indent=2)}")
        
        # Save as WAV file
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        logger.info(f"Saved debug audio to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving debug audio: {e}")
        return ""

app = FastAPI(
    title="Secretary - Voice Assistant",
    description="A modern voice assistant built with FastAPI and WebRTC",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Store conversation history and WebRTC connections
conversation_history: Dict[str, List[dict]] = {}
peer_connections: Dict[str, RTCPeerConnection] = {}

def parse_ice_candidate(candidate_str: str) -> dict:
    """Parse ICE candidate string into components."""
    parts = candidate_str.split()
    if len(parts) < 8:
        raise ValueError(f"Invalid ICE candidate string: {candidate_str}")
    
    # Extract basic components
    foundation = parts[0].split(':')[1]  # Remove 'candidate:' prefix
    component = int(parts[1])
    protocol = parts[2].lower()
    priority = int(parts[3])
    ip = parts[4]
    port = int(parts[5])
    type = parts[7]  # 'typ' is at index 6, actual type is at 7
    
    # Parse additional attributes
    related_address = None
    related_port = None
    for i in range(8, len(parts), 2):
        if i + 1 < len(parts):
            if parts[i] == 'raddr':
                related_address = parts[i + 1]
            elif parts[i] == 'rport':
                related_port = int(parts[i + 1])
    
    return {
        'foundation': foundation,
        'component': component,
        'protocol': protocol,
        'priority': priority,
        'ip': ip,
        'port': port,
        'type': type,
        'related_address': related_address,
        'related_port': related_port
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint that processes messages and returns AI responses.
    """
    try:
        response = await get_chat_response(request.messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.post("/listen")
async def listen():
    """
    Endpoint that listens to the microphone, converts speech to text, and returns the AI's response as an audio file.
    """
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
        text = recognizer.recognize_google(audio)
        print(text)
        request = ChatRequest(messages=[{"role": "user", "content": text}])
        response = await get_chat_response(request.messages)
        print(response)
        response_text = response.message.content
        with tempfile.NamedTemporaryFile(suffix=".aiff", delete=False) as temp_file:
            temp_filename = temp_file.name
        subprocess.run(["say", "-v", "Alex", "-o", temp_filename, response_text])
        return FileResponse(temp_filename, media_type="audio/aiff", filename="response.aiff")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def process_audio(audio_data: np.ndarray, sample_rate: int = 16000) -> str:
    """Process audio data and return transcribed text."""
    try:
        # Calculate audio statistics from the original audio
        audio_level = np.abs(audio_data).mean()
        
        # Save debug audio with metadata
        metadata = {
            "audio_level": float(audio_level),
            "is_silence": bool(audio_level < 0.005),
            "processing_timestamp": datetime.datetime.now().isoformat(),
            "original_sample_rate": sample_rate
        }
        
        # Save the ORIGINAL audio for debugging
        save_debug_audio(audio_data, sample_rate, "original", metadata)
        
        # Check audio levels - using a more lenient threshold
        if audio_level < 0.005:
            logger.warning(f"Audio level too low ({audio_level:.2f}), skipping recognition")
            return ""

        # --- RESAMPLING STEP ---
        # Create a pydub AudioSegment to make resampling easy
        try:
            original_segment = AudioSegment(
                data=audio_data.tobytes(),
                sample_width=2,  # int16 = 2 bytes
                frame_rate=sample_rate,
                channels=1
            )
            # Resample to 16kHz, which is ideal for most STT engines
            resampled_segment = original_segment.set_frame_rate(16000)
            resampled_audio_array = np.array(resampled_segment.get_array_of_samples())
            
            # Save the RESAMPLED audio for debugging
            save_debug_audio(resampled_audio_array, 16000, "resampled", metadata)

        except Exception as e:
            logger.error(f"Error during audio resampling: {e}")
            # Fallback to original audio if resampling fails
            resampled_segment = original_segment
            
        # --- SPEECH RECOGNITION on RESAMPLED audio ---
        logger.info("Attempting speech recognition on resampled audio...")
        recognizer = sr.Recognizer()

        # Create an AudioData object for the speech recognition library
        audio_data_for_sr = sr.AudioData(
            resampled_segment.raw_data,
            sample_rate=16000, # Use the resampled rate
            sample_width=2
        )
        
        try:
            text = recognizer.recognize_google(audio_data_for_sr)
            logger.info(f"Recognized text: {text}")
            
            # Save successful recognition with text
            metadata["recognized_text"] = text
            save_debug_audio(np.array(resampled_segment.get_array_of_samples()), 16000, "recognized", metadata)
            
            return text
        except sr.UnknownValueError:
            logger.error(f"Speech recognition could not understand audio - Audio levels: min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean()}, std={audio_data.std()}")
            return ""
        except sr.RequestError as e:
            logger.error(f"Could not request results from speech recognition service: {e}")
            return ""
            
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return ""

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for continuous conversation.
    Maintains conversation history and provides real-time responses.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    pc = None
    audio_track = None
    
    async def consume_audio_track(track):
        """A task to consume the audio track and perform VAD."""
        # VAD parameters
        VAD_THRESHOLD = 300  # Energy threshold for speech detection (based on int16 samples)
        VAD_SILENCE_FRAMES = 25  # Frames of silence to end an utterance (e.g., 25 * 20ms = 500ms)
        VAD_MIN_SPEECH_FRAMES = 5  # Min speech frames to be considered an utterance

        is_speaking = False
        silent_frames_count = 0
        audio_buffer = bytearray()
        speech_frames_count = 0

        logger.info("Audio consumer task started.")
        while True:
            try:
                frame = await track.recv()
            except MediaStreamError:
                logger.info("Audio track ended.")
                return
            
            # Convert audio frame to numpy array. This is compatible with older aiortc/PyAV.
            audio_samples = frame.to_ndarray()

            # If stereo, take one channel (though client should send mono).
            if audio_samples.ndim > 1 and audio_samples.shape[0] > 1:
                audio_samples = audio_samples[0, :]
            
            audio_samples = audio_samples.flatten()

            # Ensure data is int16, which is what VAD and process_audio expect.
            if audio_samples.dtype != np.int16:
                # Common case from WebRTC is float, in [-1.0, 1.0] range.
                audio_samples = (audio_samples * 32767).astype(np.int16)
            
            # Simple VAD based on Root Mean Square (RMS)
            rms = np.sqrt(np.mean(np.square(audio_samples, dtype=np.float64)))

            if is_speaking:
                logger.info(f"Still speaking, rms: {rms}")
                # We are in a speech segment
                audio_buffer.extend(audio_samples.tobytes())
                speech_frames_count += 1
                if rms < VAD_THRESHOLD:
                    silent_frames_count += 1
                    if silent_frames_count > VAD_SILENCE_FRAMES:
                        # End of utterance
                        logger.info(f"End of speech detected. Processing {len(audio_buffer)} bytes.")
                        if speech_frames_count > VAD_MIN_SPEECH_FRAMES:
                            # Process the buffered audio
                            logger.info(f"Processing {len(audio_buffer)} bytes. frame.sample_rate: {frame.sample_rate}")
                            full_audio_data = np.frombuffer(audio_buffer, dtype=np.int16)
                            text = await process_audio(full_audio_data, sample_rate=frame.sample_rate)
                            logger.info(f"Processed audio data: {text}")
                            
                            if text:
                                await websocket.send_json({"type": "transcription", "text": text})
                        # Reset for next utterance
                        is_speaking = False
                        audio_buffer.clear()
                        speech_frames_count = 0
                else:
                    # Still speaking, reset silence counter
                    silent_frames_count = 0
            else:
                # We are in a silence segment
                if rms > VAD_THRESHOLD:
                    # Start of utterance
                    logger.info("Start of speech detected.")
                    is_speaking = True
                    silent_frames_count = 0
                    speech_frames_count = 1
                    audio_buffer.extend(audio_samples.tobytes())

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "offer":
                logger.info(f"Received WebRTC offer from client {id(websocket)}")
                pc = RTCPeerConnection()
                
                @pc.on("track")
                async def on_track(track):
                    nonlocal audio_track
                    if track.kind == "audio":
                        logger.info(f"Received audio track from client {id(websocket)}")
                        audio_track = track
                        # Start a task to consume the audio track
                        asyncio.create_task(consume_audio_track(track))
                
                # Set the remote description
                await pc.setRemoteDescription(RTCSessionDescription(
                    sdp=message["sdp"],
                    type=message["type"]
                ))
                
                # Create and send answer
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                
                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp
                })
                
            elif message["type"] == "candidate":
                if pc and pc.remoteDescription and message.get("candidate"):
                    try:
                        candidate_obj = message.get("candidate", {})
                        candidate_sdp = candidate_obj.get("candidate")

                        if not candidate_sdp:
                            logger.warning("Received and ignoring an empty ICE candidate. This is normal.")
                            continue
                        
                        logger.info(f"Received ICE candidate from client {id(websocket)}")

                        # Use the official aiortc SDP parser to create the candidate.
                        candidate = candidate_from_sdp(candidate_sdp)
                        
                        # The parser doesn't know about sdpMid or sdpMLineIndex, so add them manually.
                        candidate.sdpMid = candidate_obj.get("sdpMid")
                        candidate.sdpMLineIndex = candidate_obj.get("sdpMLineIndex")

                        await pc.addIceCandidate(candidate)
                    except Exception as e:
                        logger.error(f"Error adding ICE candidate: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "error": f"Failed to add ICE candidate: {str(e)}"
                        })
                    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for client {id(websocket)}")
    except Exception as e:
        logger.error(f"Error in WebSocket connection: {e}")
        await websocket.send_json({
            "type": "error",
            "error": str(e)
        })
    finally:
        if pc:
            await pc.close()

@app.get("/")
async def read_root():
    """
    Serve the main HTML page.
    """
    return FileResponse("app/static/index.html") 