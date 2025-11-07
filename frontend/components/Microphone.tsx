
import React, { useState, useRef } from 'react';
import './Microphone.css';
import { API_BASE_URL } from '@/constants';

interface MicrophoneProps {
  onTranscriptionSuccess: (transcription: string) => void;
}

const Microphone: React.FC<MicrophoneProps> = ({ onTranscriptionSuccess }) => {
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const handleToggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const startRecording = () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
      .then(stream => {
        streamRef.current = stream;
        mediaRecorder.current = new MediaRecorder(stream);
        mediaRecorder.current.ondataavailable = (event) => {
          audioChunks.current.push(event.data);
        };
        mediaRecorder.current.onstop = () => {
          const audioBlob = new Blob(audioChunks.current, { type: 'audio/wav' });
          sendAudioToServer(audioBlob);
          audioChunks.current = [];
        };

        const sendAudioToServer = async (audioBlob: Blob) => {
          const formData = new FormData();
          formData.append('audio', audioBlob);

          try {
            const response = await fetch(`${API_BASE_URL}/transcribe`, {
              method: 'POST',
              body: formData,
            });

            if (response.ok) {
              const data = await response.json();
              onTranscriptionSuccess(data.transcription);
            } else {
              console.error('Error transcribing audio:', response.statusText);
            }
          } catch (error) {
            console.error('Error sending audio to server:', error);
          }
        };
        mediaRecorder.current.start();
        setIsRecording(true);
      })
      .catch(error => {
        console.error('Error accessing microphone:', error);
      });
  };

  const stopRecording = () => {
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
      setIsRecording(false);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
      }
    }
  };

  return (
    <div className="microphone-container text-gray-500 hover:text-gray-700">
      <button onClick={handleToggleRecording} className={`microphone-button ${isRecording ? 'recording' : ''}`}>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="22"></line>
        </svg>
      </button>
    </div>
  );
};

export default Microphone;
