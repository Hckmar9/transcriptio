import gradio as gr
import requests
import tempfile
import os
import json
import traceback
import time
import soundfile as sf

API_URL = "http://localhost:8000/transcribe/"

def transcribe_audio(audio, language=None, task="transcribe"):
    start_time = time.time()
    temp_path = None
    
    try:
        if audio is None:
            return "Please upload an audio file or record audio"
            
        print(f"Received audio for transcription: type={type(audio)}")
        
        debug_info = [
            "==== DEBUG INFORMATION ====",
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Language requested: {language}",
            f"Task: {task}"
        ]
        
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"gradio_audio_{int(time.time())}.wav")
        
        if isinstance(audio, tuple):
            sample_rate, audio_data = audio
            debug_info.append(f"Audio from microphone: {len(audio_data)} samples at {sample_rate}Hz")
            
            sf.write(temp_path, audio_data, sample_rate)
            debug_info.append(f"Saved to temporary file: {temp_path}")
            
        elif isinstance(audio, str):
            temp_path = audio
            debug_info.append(f"Using provided file path: {temp_path}")
            
            if not os.path.exists(temp_path):
                return f"Error: File does not exist at {temp_path}"
                
            file_size = os.path.getsize(temp_path)
            debug_info.append(f"File size: {file_size} bytes")
            
        else:
            return f"Error: Unexpected audio type: {type(audio)}"
        
        try:
            with open(temp_path, 'rb') as f:
                f.read(1024)
                f.seek(0)
        except Exception as e:
            return f"Error: Could not read audio file: {str(e)}"
        
        # API request
        debug_info.append("Preparing API request...")
        files = {"file": open(temp_path, 'rb')}
        params = {}
        if language:
            params["language"] = language
        if task:
            params["task"] = task
            
        debug_info.append(f"Request parameters: {params}")
        
        debug_info.append("Sending request to API...")
        try:
            response = requests.post(API_URL, files=files, params=params, timeout=90)
            debug_info.append(f"API response status: {response.status_code}")
            
            if response.status_code != 200:
                debug_info.append(f"Error response: {response.text[:500]}")
                return f"Error: API returned status {response.status_code}\n\n" + "\n".join(debug_info)
                
            result = response.json()
            debug_info.append("Successfully parsed JSON response")
            
            output_lines = []
            output_lines.append(f"Transcription: {result.get('text', 'No text returned')}")
            output_lines.append("")
            output_lines.append(f"Detected Language: {result.get('language', 'unknown')}")
            output_lines.append(f"Processing Time: {result.get('processing_time', time.time() - start_time):.2f} seconds")
            output_lines.append("")
            
            segments = result.get('segments', [])
            if segments:
                output_lines.append("Segments:")
                for i, segment in enumerate(segments):
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '')
                    # This line of code is part of the function transcribe_audio, it processes the audio file and returns the transcription. 
                    output_lines.append(f"{i+1}. [{start:.2f}s - {end:.2f}s]: {text}")
            else:
                output_lines.append("Note: Segment timestamps not available with current model version.") # I added this bc I had an issue with a whispers version
            debug_info.append("Processing completed successfully")
            
            return "\n".join(output_lines)
            
        except requests.exceptions.Timeout: #sometimes the API takes too long to respond
            return "Error: API request timed out after 90 seconds. The server might be overloaded or the audio file might be too large."
            
        except requests.exceptions.ConnectionError:
            return f"Error: Could not connect to API at {API_URL}. Make sure the server is running."
            
        except Exception as e:
            debug_info.append(f"Exception during API request: {str(e)}")
            debug_info.append(traceback.format_exc())
            return f"Error: {str(e)}\n\n" + "\n".join(debug_info)
            
    except Exception as e:
        return f"Error: {str(e)}\n\n{traceback.format_exc()}"
        
    finally:
        if isinstance(audio, tuple) and temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"Removed temporary file: {temp_path}")
            except Exception as e:
                print(f"Error removing temporary file: {str(e)}")

# Gradio interface
with gr.Blocks(title="Transcriptio") as app:
    gr.Markdown("# Audio transcription service")
    gr.Markdown("Upload an audio file or record audio to get a transcription using Whisper.")
    
    with gr.Tab("Upload Audio"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                with gr.Row():
                    # language input is not working, I'm still trying to figure it out... [wip]
                    language_input = gr.Dropdown(
                        choices=["", "en", "fr", "de", "es", "it", "ja", "zh", "ru", "pt"], 
                        value="", 
                        label="Language (optional)"
                    )
                    task_input = gr.Radio(
                        choices=["transcribe"], 
                        value="transcribe", 
                        label="Task"
                    )
                submit_btn = gr.Button("Transcribe")
            
            with gr.Column():
                output_text = gr.Textbox(label="Transcription Result", lines=10)
    
    with gr.Tab("Record Audio"):
        with gr.Row():
            with gr.Column():
                audio_record = gr.Audio(sources=["microphone"], type="numpy", label="Record Audio")
                with gr.Row():
                    # language input is not working, I'm still trying to figure it out... [wip]
                    language_record = gr.Dropdown(
                        choices=["", "en", "fr", "de", "es", "it", "ja", "zh", "ru", "pt"], 
                        value="", 
                        label="Language (optional)"
                    )
                    task_record = gr.Radio(
                        choices=["transcribe"], 
                        value="transcribe", 
                        label="Task"
                    )
                record_btn = gr.Button("Transcribe Recording")
            
            with gr.Column():
                record_output = gr.Textbox(label="Transcription Result", lines=10)
    
    submit_btn.click(
        transcribe_audio,
        inputs=[audio_input, language_input, task_input],
        outputs=[output_text]
    )
    
    record_btn.click(
        transcribe_audio,
        inputs=[audio_record, language_record, task_record],
        outputs=[record_output]
    )
    
    gr.Markdown("### About")
    gr.Markdown("""
    This application uses:
    - FastAPI for the backend API
    - Hugging Face's Whisper model for transcription
    - Librosa for audio processing
    - Gradio for this user interface
    """)

if __name__ == "__main__":
    app.launch(share=True)