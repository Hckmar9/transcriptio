import librosa
import numpy as np
import soundfile as sf
import tempfile
import os

def preprocess_audio(file_path, target_sr=16000):
    """
    Preprocess audio using Librosa with extensive error handling
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (Whisper expects 16kHz)
    
    Returns:
        numpy array of audio data
    """
    try:
        print(f"Starting audio preprocessing for file: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found at {file_path}")
        
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size} bytes")
        
        if file_size == 0:
            raise ValueError("Audio file is empty (0 bytes)")
        
        try:
            print("Loading audio with librosa...")
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            print(f"Successfully loaded audio: {len(audio)} samples at {sr}Hz")
        except Exception as e:
            print(f"Librosa load failed: {str(e)}")
            
            print("Trying soundfile as fallback...")
            import soundfile as sf
            audio, sr = sf.read(file_path)
            
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            
            print(f"Loaded with soundfile: {len(audio)} samples at {sr}Hz")
        
        # Resample to 16kHz if needed (Whisper requires 16kHz)
        if sr != target_sr:
            print(f"Resampling from {sr}Hz to {target_sr}Hz...")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            print(f"Resampled audio length: {len(audio)} samples")
        
        print("Normalizing audio...")
        audio = librosa.util.normalize(audio)
        
        if np.isnan(audio).any() or np.isinf(audio).any():
            print("WARNING: Audio contains NaN or Inf values. Replacing with zeros.")
            audio = np.nan_to_num(audio)
        
        if len(audio) < target_sr * 0.1:
            print("WARNING: Audio is very short, might be insufficient for transcription")
        
        return audio
    
    except Exception as e:
        import traceback
        print(f"Error in audio preprocessing: {str(e)}")
        print(traceback.format_exc())
        raise

def save_processed_audio(audio_array, output_path, sr=16000):
    """
    Save processed audio array to file
    
    Args:
        audio_array: Numpy array of audio data
        output_path: Path to save the file
        sr: Sample rate
    """
    sf.write(output_path, audio_array, sr)
    
    return output_path

def extract_audio_features(audio_array, sr=16000):
    """
    Extract audio features for potential additional analysis
    
    Args:
        audio_array: Numpy array of audio data
        sr: Sample rate
    
    Returns:
        dict of audio features
    """
    duration = librosa.get_duration(y=audio_array, sr=sr)
    
    mfccs = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=13)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sr)
    
    return {
        "duration": duration,
        "mfccs": mfccs.mean(axis=1).tolist(),
        "spectral_centroid": spectral_centroid.mean()
    }