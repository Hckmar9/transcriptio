import time
from transformers import pipeline
import librosa
import numpy as np
import os
from app.services.audio_processing import preprocess_audio

class TranscriptionService:
    def __init__(self):
        print("Initializing TranscriptionService...")
        
        try:
            import transformers
            print(f"Using transformers version: {transformers.__version__}")
            
            try:
                print("Attempting to load whisper-tiny model...")
                self.model = pipeline(
                    task="automatic-speech-recognition", 
                    model="openai/whisper-tiny",
                    device=-1
                )
                print("Successfully loaded whisper-tiny model on CPU")
            except Exception as e:
                print(f"Error loading whisper-tiny: {str(e)}")
                
                print("Trying alternative model loading method...")
                from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
                
                processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
                model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny")
                
                self.model = pipeline(
                    task="automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=-1
                )
                print("Successfully loaded model using alternative method")
        
        except Exception as e:
            import traceback
            print(f"CRITICAL ERROR initializing transcription service: {str(e)}")
            print(traceback.format_exc())
            
            print("Creating fallback dummy model")
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        def dummy_model(*args, **kwargs):
            error_msg = "Transcription model failed to load. Please check server logs."
            print(f"Dummy model called with args: {args}, kwargs: {kwargs}")
            return {"text": error_msg, "segments": []}
        
        return dummy_model
    
    async def transcribe(self, audio_file_path, language=None, task="transcribe"):
        """
        Transcribe audio file using Whisper model
        """
        start_time = time.time()
        
        try:
            # Librosa
            audio_array = preprocess_audio(audio_file_path)
            
            # For longer audio (>30s), we need to enable timestamps
            audio_length_seconds = len(audio_array) / 16000  # 16kHz sample rate
            print(f"Audio length: {audio_length_seconds:.2f} seconds")
            
            options = {}
            
            # For longer audio (>30s), return_timestamps must be True
            if audio_length_seconds > 30:
                options["return_timestamps"] = True
                print("Enabling return_timestamps for long audio")
            
            print(f"Running model with options: {options}")
            result = self.model(audio_array, **options)
            
            if isinstance(result, dict):
                text = result.get("text", "")
                chunks = result.get("chunks", [])
            else:
                text = result if isinstance(result, str) else str(result)
                chunks = []
            
            processing_time = time.time() - start_time
            
            segments = []
            if chunks:
                for i, chunk in enumerate(chunks):
                    if isinstance(chunk, dict):
                        segments.append({
                            "start": chunk.get("timestamp", (0, 0))[0],
                            "end": chunk.get("timestamp", (0, 0))[1],
                            "text": chunk.get("text", "")
                        })
                    elif isinstance(chunk, tuple) and len(chunk) == 2:
                        segment_text, timestamps = chunk
                        start, end = timestamps if isinstance(timestamps, tuple) else (0, 0)
                        segments.append({
                            "start": start,
                            "end": end,
                            "text": segment_text
                        })
            
            return {
                "text": text,
                "segments": segments,
                "language": language or "en",
                "processing_time": processing_time
            }
            
        except Exception as e:
            import traceback
            print(f"Transcription error: {str(e)}")
            print(traceback.format_exc())
            
            raise Exception(f"Failed to transcribe audio: {str(e)}")

transcription_service = TranscriptionService()