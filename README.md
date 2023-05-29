# Voice Controlled AI Assistant

This project is a voice-controlled AI assistant that can process user input through speech and respond with synthesized speech. It utilizes OpenAI's GPT model for generating responses and Eleven Labs' text-to-speech service for voice synthesis.

## How it works

The AI assistant is composed of two main Python scripts: `speech_assistant.py` and `query_analysis.py`.

### speech_assistant.py

This script is responsible for the following tasks:

1. Recording the user's voice input.
2. Transcribing the voice input using OpenAI's transcription service.
3. Passing the transcribed text to the AI model for generating a response.
4. Passing the AI-generated response to Eleven Labs' text-to-speech service for voice synthesis.
5. Playing the synthesized speech back to the user.
6. Handling code snippets in the AI-generated responses, if any.

The script uses multiple threads to handle tasks such as playing audio, displaying a spinning icon, and handling the AI-generated tokens. The main thread manages the voice input and output process.

### query_analysis.py

This script preprocesses user queries by determining if the AI assistant needs to access the user's clipboard or a local file to answer the query. It uses separate language models to make these determinations.

The script contains several functions:

1. `load_clip_chain`: Initializes an LLMChain with GPT-3.5-turbo to determine if the user's clipboard is needed.
2. `load_file_chains`: Initializes two LLMChains with GPT-4 to determine if a local file is needed and to extract the filename.
3. `find_file`: Searches for a specified file within a search path.
4. `prep_all_inputs`: Takes a query, runs it through the chains, and appends the clipboard content or local file content to the query if necessary. It also returns a dictionary of 'edits' that indicates if the clipboard or file access was needed.

By providing the necessary context from the clipboard or local files (if required) before passing it to the main AI assistant, this script helps in preprocessing the user's query.

## Usage

To use the voice-controlled AI assistant, run the `speech_assistant.py` script. The assistant will listen for your voice input, transcribe it, and generate a response based on the input. The response will be synthesized into speech and played back to you.

Make sure to set up the required API keys and dependencies before running the script.

## Dependencies

- OpenAI GPT models
- Eleven Labs text-to-speech service
- PyAudio
- Pyperclip
- Keyboard
- Colorama