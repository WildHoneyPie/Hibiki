# Real-time Text Emotion Detection

This project uses LangChain and OpenAI's GPT-4 to perform real-time emotion detection on text input. The application provides a simple web interface where users can input text and receive emotion analysis.

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Features

- Real-time emotion detection using GPT-4
- Web interface for easy interaction
- Detailed emotion analysis with confidence scores
- Support for multiple emotions in a single text

## Requirements

- Python 3.8+
- OpenAI API key
- Internet connection 