// Global variables
let ws = null;
let peerConnection = null;
let isConnected = false;
let audioContext = null;
let mediaStream = null;
let isRecording = false;

// Constants
const SAMPLE_RATE = 48000;
const FRAME_SIZE = 1920;
const SILENCE_THRESHOLD = 0.01;
const MIN_AUDIO_DURATION = 0.5; // seconds
const MAX_AUDIO_DURATION = 10; // seconds

// Audio processing variables
let audioBuffer = [];
let recordingStartTime = null;
let silenceStartTime = null;

async function connectWebSocket() {
    if (ws) {
        ws.close();
    }
    
    ws = new WebSocket(`ws://${window.location.host}/ws/chat`);
    
    ws.onopen = async () => {
        console.log('WebSocket connected');
        isConnected = true;
        
        // Set up WebRTC after WebSocket is connected
        const success = await setupWebRTC();
        if (!success) {
            console.error('Failed to set up WebRTC');
            ws.close();
        }
    };
    
    ws.onmessage = async (event) => {
        const message = JSON.parse(event.data);
        
        switch (message.type) {
            case 'answer':
                if (peerConnection) {
                    await peerConnection.setRemoteDescription(new RTCSessionDescription({
                        type: 'answer',
                        sdp: message.sdp
                    }));
                }
                break;
                
            case 'candidate':
                if (peerConnection && peerConnection.remoteDescription) {
                    try {
                        await peerConnection.addIceCandidate(new RTCIceCandidate(message.candidate));
                    } catch (error) {
                        console.error('Error adding ICE candidate:', error);
                    }
                }
                break;
                
            case 'transcription':
                if (message.text) {
                    console.log('Transcription:', message.text);
                    const event = new CustomEvent('transcription', { detail: message.text });
                    document.dispatchEvent(event);
                }
                break;
                
            case 'error':
                console.error('Server error:', message.error);
                break;
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        isConnected = false;
        if (peerConnection) {
            peerConnection.close();
            peerConnection = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 3000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

// Initialize connection when the page loads
document.addEventListener('DOMContentLoaded', () => {
    connectWebSocket();
});

async function setupWebRTC() {
    try {
        // Get user media with specific constraints
        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
                channelCount: 1,
                sampleRate: SAMPLE_RATE
            },
            video: false
        });
        
        // Create and configure peer connection
        peerConnection = new RTCPeerConnection({
            iceServers: [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        });
        
        // Add audio track to peer connection
        const audioTrack = mediaStream.getAudioTracks()[0];
        if (!audioTrack) {
            throw new Error('No audio track found in media stream');
        }
        
        // Add the track to the peer connection
        peerConnection.addTrack(audioTrack, mediaStream);
        
        // Create and send offer
        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        
        // Send the offer to the server
        ws.send(JSON.stringify({
            type: 'offer',
            sdp: peerConnection.localDescription.sdp
        }));
        
        // Set up event handlers
        peerConnection.onicecandidate = (event) => {
            // event.candidate can be an object with an empty candidate string ""
            // to signal the end of a generation of candidates. We should not send these.
            if (event.candidate && event.candidate.candidate) {
                ws.send(JSON.stringify({
                    type: 'candidate',
                    candidate: {
                        candidate: event.candidate.candidate,
                        sdpMid: event.candidate.sdpMid,
                        sdpMLineIndex: event.candidate.sdpMLineIndex,
                    }
                }));
            }
        };
        
        peerConnection.onconnectionstatechange = () => {
            console.log('Connection state:', peerConnection.connectionState);
            if (peerConnection.connectionState === 'failed') {
                console.error('WebRTC connection failed');
                // Attempt to restart the connection
                restartWebRTC();
            }
        };
        
        return true;
    } catch (error) {
        console.error('Error setting up WebRTC:', error);
        if (error.name === 'NotAllowedError') {
            alert('Please allow microphone access to use this feature.');
        } else {
            alert('Error setting up audio: ' + error.message);
        }
        return false;
    }
}

// Function to restart WebRTC connection
async function restartWebRTC() {
    if (peerConnection) {
        peerConnection.close();
        peerConnection = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    await setupWebRTC();
} 