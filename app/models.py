from pydantic import BaseModel
from typing import Optional, List

# Define a Pydantic model for the transcription request and response
class TranscriptionRequest(BaseModel):
    audio_file_path: str
    language: Optional[str] = None
    task: Optional[str] = "transcribe"

class TranscriptionResponse(BaseModel):
    text: str
    segments: Optional[List[dict]] = []
    language: str
    processing_time: float