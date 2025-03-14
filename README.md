# Transcriptio app

Welcome to the audio transcription and processing app. This project is a FastAPI application with a Gradio frontend interface, designed to provide seamless audio transcription and processing services.

### 🚨 This is a work in progress, pending to work on:

- Choose language to either upload audio or record it

## Features

- **FastAPI Backend**: A robust and scalable backend built with FastAPI, provides a better performance API endpoints.
- **Gradio Frontend**: A basic and interactive user interface by Gradio, allowing to easily interact with the application.
- **Audio Transcription**: Utilizes Whisper for accurate and efficient audio transcription services.
- **Audio Processing**: Librosa for advanced audio preprocessing capabilities.

## Project Structure

- **app/**:

  - `main.py`: Entry point for the FastAPI application.
  - `models.py`: Pydantic models for request and response validation.
  - **services/**: Here's the core service logic.
    - `transcription.py`: Handles audio transcription.
    - `audio_processing.py`: Manages audio preprocessing tasks.
  - **api/**: Defines the API endpoints.
    - `endpoints.py`: Contains the API routes.

- **frontend/**: The frontend interface.

  - `app.py`: Sets up the Gradio interface.

- `requirements.txt`: Required dependencies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Hckmar9/transcriptio.git
   cd transcriptio
   ```

2. Install the dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:

   ```bash
   python3 -m app.main
   ```

2. Start the Gradio frontend:
   ```bash
   python3 -m frontend.app
   ```

## Contributing

Contributions are always welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

Nerdily made 👩‍💻 by Hckmar
