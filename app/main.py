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
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
import av
import traceback
import numpy as np
import wave
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create debug directory for audio files
DEBUG_AUDIO_DIR = "debug_audio"
os.makedirs(DEBUG_AUDIO_DIR, exist_ok=True)

def save_debug_audio(audio_data: np.ndarray, sample_rate: int = 16000, prefix: str = "audio") -> str:
    """Save audio data to a WAV file for debugging."""
    # Create debug directory if it doesn't exist
    os.makedirs("debug_audio", exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"debug_audio/{prefix}_{timestamp}.wav"
    
    # Ensure audio data is in the correct format
    if audio_data.dtype != np.int16:
        # Normalize to int16 range
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = (audio_data / max_val * 32767).astype(np.int16)
        else:
            audio_data = audio_data.astype(np.int16)
    
    # Log audio statistics before saving
    logger.info(f"Saving audio file {prefix}_{timestamp}.wav with stats: min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean():.2f}, std={audio_data.std():.2f}")
    
    # Save as WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())
    
    logger.info(f"Saved debug audio to: {filename}")
    return filename

app = FastAPI(
    title="Conversational AI Bot",
    description="A modern conversational AI bot built with FastAPI and Ollama",
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

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for continuous conversation.
    Maintains conversation history and provides real-time responses.
    """
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    # Initialize conversation history for this client
    client_id = str(id(websocket))
    conversation_history[client_id] = []
    
    try:
        while True:
            try:
                # Wait for WebSocket messages
                message = await websocket.receive_text()
                logger.debug(f"Received WebSocket message: {message[:200]}...")  # Log first 200 chars
                data = json.loads(message)
                logger.debug(f"Parsed WebSocket data: {json.dumps(data, indent=2)}")
                
                if data["type"] == "offer":
                    logger.info(f"Received WebRTC offer from client {client_id}")
                    # Create peer connection
                    pc = RTCPeerConnection()
                    peer_connections[client_id] = pc
                    
                    # Set up audio handling
                    @pc.on("track")
                    async def on_track(track):
                        if track.kind == "audio":
                            logger.info(f"Received audio track from client {client_id}")
                            
                            # Create a temporary file for recording
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                                recorder = MediaRecorder(temp_file.name)
                                recorder.addTrack(track)
                                await recorder.start()
                                
                                # Process audio frames
                                audio_frames = []
                                last_process_time = time.time()
                                min_buffer_duration = 1.0  # Minimum buffer duration in seconds
                                process_interval = 0.5     # Process every 500ms
                                max_retries = 3
                                retry_count = 0
                                
                                while True:
                                    try:
                                        frame = await track.recv()
                                        retry_count = 0  # Reset retry count on successful frame
                                        
                                        # Convert frame to numpy array
                                        audio_data = np.frombuffer(frame.planes[0], dtype=np.int16)
                                        audio_data = audio_data.reshape(1, -1)  # Reshape to (1, samples)
                                        
                                        # Log audio frame details
                                        logger.info(f"Received audio frame: shape={audio_data.shape}, dtype={audio_data.dtype}, min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean():.2f}, std={audio_data.std():.2f}")
                                        
                                        # Check audio levels - match client's noise gate threshold
                                        mean_abs = np.mean(np.abs(audio_data))
                                        if mean_abs < 0.1 * 32767:  # Convert client's 0.1 threshold to int16 range
                                            logger.warning(f"Audio level too low ({mean_abs:.2f}), skipping recognition")
                                            continue
                                        
                                        # Normalize audio data with dynamic gain control
                                        max_val = np.max(np.abs(audio_data))
                                        if max_val > 0:
                                            # Calculate gain to match client's compressor settings
                                            target_peak = 0.8 * 32767  # 80% of int16 max
                                            gain = min(target_peak / max_val, 4.0)  # Match client's compression ratio
                                            audio_data = (audio_data * gain).astype(np.int16)
                                            logger.info(f"Applied gain: {gain:.2f}x, new levels - min={audio_data.min()}, max={audio_data.max()}, mean={audio_data.mean():.2f}, std={audio_data.std():.2f}")
                                        
                                        audio_frames.append(audio_data)
                                        
                                        # Process accumulated frames
                                        current_time = time.time()
                                        if (current_time - last_process_time >= process_interval and 
                                            len(audio_frames) * audio_data.shape[1] / 16000 >= min_buffer_duration):
                                            
                                            # Combine frames
                                            combined_audio = np.concatenate(audio_frames, axis=1)
                                            logger.info(f"Combined audio shape: {combined_audio.shape}, dtype: {combined_audio.dtype}, min={combined_audio.min()}, max={combined_audio.max()}, mean={combined_audio.mean():.2f}, std={combined_audio.std():.2f}")
                                            
                                            # Save raw audio for debugging
                                            raw_file = save_debug_audio(combined_audio, prefix="raw")
                                            
                                            # Save normalized audio
                                            normalized_file = save_debug_audio(combined_audio, prefix="normalized")
                                            
                                            # Attempt speech recognition
                                            logger.info("Attempting speech recognition...")
                                            try:
                                                # Convert to bytes for recognition
                                                audio_bytes = combined_audio.tobytes()
                                                
                                                # Perform speech recognition
                                                result = recognizer.recognize_google(audio_bytes, language="en-US")
                                                
                                                if result:
                                                    logger.info(f"Recognized text: {result}")
                                                    # Save successful recognition audio
                                                    success_file = save_debug_audio(combined_audio, prefix="success")
                                                    
                                                    # Send result to client
                                                    await websocket.send_json({
                                                        "type": "transcription",
                                                        "text": result
                                                    })
                                            except sr.UnknownValueError:
                                                logger.error(f"Speech recognition could not understand audio - Audio levels: min={combined_audio.min()}, max={combined_audio.max()}, mean={combined_audio.mean():.2f}, std={combined_audio.std():.2f}")
                                            except Exception as e:
                                                logger.error(f"Error during speech recognition: {str(e)}")
                                                
                                            # Clear processed frames
                                            audio_frames = []
                                            last_process_time = current_time
                                            
                                    except MediaStreamError as e:
                                        retry_count += 1
                                        logger.warning(f"MediaStreamError occurred (attempt {retry_count}/{max_retries}): {str(e)}")
                                        
                                        if retry_count >= max_retries:
                                            logger.error("Max retries reached for MediaStreamError, ending track processing")
                                            break
                                        
                                        await asyncio.sleep(0.1)  # Small delay before retry
                                        continue
                                        
                                    except Exception as e:
                                        logger.error(f"Error receiving audio frame: {str(e)}")
                                        break
                                
                                await recorder.stop()
                                os.unlink(temp_file.name)
                    
                    # Set the remote description
                    offer = RTCSessionDescription(
                        sdp=data["sdp"]['sdp'],
                        type=data["type"]
                    )
                    await pc.setRemoteDescription(offer)
                    
                    # Create and send answer
                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)
                    
                    try:
                        answer_message = {
                            "type": "answer",
                            "sdp": pc.localDescription.sdp,
                            "type": pc.localDescription.type
                        }
                        logger.debug(f"Sending WebRTC answer: {json.dumps(answer_message, indent=2)}")
                        await websocket.send_json(answer_message)
                    except WebSocketDisconnect:
                        logger.info("WebSocket disconnected during answer")
                        return
                    
                elif data["type"] == "ice-candidate":
                    if client_id in peer_connections:
                        pc = peer_connections[client_id]
                        try:
                            # Parse the ICE candidate string
                            candidate_str = data["candidate"]["candidate"]
                            sdp_mid = data["candidate"]["sdpMid"]
                            sdp_mline_index = data["candidate"]["sdpMLineIndex"]
                            
                            logger.debug(f"Processing ICE candidate: {candidate_str}")
                            logger.debug(f"sdpMid: {sdp_mid}, sdpMLineIndex: {sdp_mline_index}")
                            
                            # Parse ICE candidate components
                            candidate_parts = parse_ice_candidate(candidate_str)
                            
                            # Create ICE candidate
                            candidate = aiortc.RTCIceCandidate(
                                component=candidate_parts['component'],
                                foundation=candidate_parts['foundation'],
                                priority=candidate_parts['priority'],
                                protocol=candidate_parts['protocol'],
                                relatedAddress=candidate_parts['related_address'],
                                relatedPort=candidate_parts['related_port'],
                                sdpMid=sdp_mid,
                                sdpMLineIndex=sdp_mline_index,
                                tcpType=None,
                                type=candidate_parts['type'],
                                ip=candidate_parts['ip'],
                                port=candidate_parts['port']
                            )
                            await pc.addIceCandidate(candidate)
                            logger.info(f"Added ICE candidate for client {client_id}")
                        except Exception as e:
                            logger.error(f"Error adding ICE candidate: {str(e)}")
                            logger.error(f"Stack trace: {traceback.format_exc()}")
                            logger.error(f"Candidate data: {json.dumps(data['candidate'], indent=2)}")
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for client {client_id}")
                break
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                logger.error(f"Stack trace: {traceback.format_exc()}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "content": str(e)
                    })
                except WebSocketDisconnect:
                    break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket connection closed for client {client_id}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket handler: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
    finally:
        # Clean up WebRTC connection
        if client_id in peer_connections:
            pc = peer_connections[client_id]
            await pc.close()
            del peer_connections[client_id]
        # Clean up conversation history
        if client_id in conversation_history:
            del conversation_history[client_id]

@app.get("/")
async def read_root():
    """
    Serve the main HTML page.
    """
    return FileResponse("app/static/index.html") 