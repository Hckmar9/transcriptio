from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import tempfile
import os
import shutil
from app.models import TranscriptionRequest, TranscriptionResponse
from app.services.transcription import transcription_service

router = APIRouter()

@router.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = None,
    task: str = "transcribe",
    background_tasks: BackgroundTasks = None
):
    temp_file = None
    try:
        print(f"Received transcription request: language={language}, task={task}, filename={file.filename}")
        
        # Creates a temporary file to store uploaded audio
        # It doesn't support multiple audio files at the same time... for now...
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        shutil.copyfileobj(file.file, temp_file)
        temp_file.close()
        
        print(f"Saved uploaded file to {temp_file.name}")
        
        result = await transcription_service.transcribe(
            temp_file.name,
            language=language,
            task=task
        )
        
        if background_tasks:
            background_tasks.add_task(os.unlink, temp_file.name)
        
        return result
    
    except Exception as e:
        import traceback
        print(f"Error in transcribe_audio endpoint: {str(e)}") #If audio mixes languages, it might throw an error
        print(traceback.format_exc())
        
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)
        
        raise HTTPException(
            status_code=500, 
            detail=f"Transcription failed: {str(e)}. Please check server logs for more details."
        )

@router.get("/health/")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}