import queue
import pyaudio
import sys
import wave
import shutil
import keyboard
import tempfile
import openai
import json
import os
import itertools
import threading
import time 
from colorama import init, Fore
from langchain.callbacks.base import BaseCallbackHandler
from langchain import ConversationChain, LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import SystemMessage
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from elevenlabs import generate, voices, set_api_key
from elevenlabs import stream as xi_stream
from query_analysis import prep_all_inputs
from dotenv import load_dotenv
load_dotenv()
# Parameters for recording
CHUNK = 1024
FORMAT = pyaudio.paInt24
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5

# Start Colorama
init()

# Set API Key for Eleven Labs
set_api_key(os.getenv("XILABS_API_KEY"))
# Cache the voices
voices()


# Global variables for locks and queues
audio_queue = queue.Queue()
is_playing_lock = threading.Lock()
is_recording_lock = threading.Lock()
is_running_lock = threading.Lock()

# Global flags
is_playing = False
is_recording = False
is_running = False

# Global variables for recording
frames = []
p = None
stream = None
recording_done = threading.Event()

# Spinner for loading symbol
def spinning_cursor():
    while True:
        for cursor in itertools.cycle(['-', '\\', '|', '/']):
            yield cursor

spinner = spinning_cursor()

def get_terminal_width():
    columns, _ = shutil.get_terminal_size()
    return columns

def generate_and_play(text, voice):
    audio = generate(
        text=text,
        voice=voice,
        stream=True,
    )
    audio_queue.put(audio)

def threaded_generate_and_play(text, voice):
    threading.Thread(target=generate_and_play, args=(text, voice), daemon=True).start()

def is_playing_audio():
    global is_playing
    with is_playing_lock:
        return is_playing

def is_recording_audio():
    global is_recording
    with is_recording_lock:
        return is_recording

def is_running_llm():
    global is_running
    with is_running_lock:
        return is_running_llm

def play_audio_from_queue():
    global is_playing
    while True:
        audio = audio_queue.get()
        with is_playing_lock:
            is_playing = True
        xi_stream(audio)
        if audio_queue.qsize() == 0:
            with is_playing_lock:
                is_playing = False
        audio_queue.task_done()

def display_spinning_icon():
    while True:
        if is_playing_audio():
            sys.stdout.write(Fore.GREEN + 'Generating... ' + next(spinner))  # Move cursor up one line
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b' * (len('Generating... ') + 1) + Fore.RESET)  # Move cursor down one line after erasing
        if is_recording_audio():
            sys.stdout.write(Fore.LIGHTBLUE_EX + 'Recording... ' + next(spinner))  # Move cursor up one line
            sys.stdout.flush()
            time.sleep(0.1)
            sys.stdout.write('\b' * (len('Recording... ') + 1) + Fore.RESET)  # Move cursor down one line after erasing
        else:
            time.sleep(0.1)

def send_to_streamlit_queue():
    pass


class XILabsOutputHandler():
    def __init__(self, voice):
        self.voice = voice
        self.token_buffer = []
        self.sentence_buffer = ""
        self.code_buffer = ""
        self.is_code = False
        self.is_code_end = False
        self.is_incomplete_delimiter = False
        self.num_curr_ticks_delimiter = 0
        
    def send_token(self, token):
        if token.startswith("`"): # Check if token starts with a backtick
            if self.num_curr_ticks_delimiter == 0: # Check if it's a new delimiter
                self.is_incomplete_delimiter = True # Set incomplete delimiter flag
                self.num_curr_ticks_delimiter = len(token) # Update current ticks
            else: # If we're in the middle of processing an incomplete delimiter
                self.num_curr_ticks_delimiter += len(token) # Update current ticks
            # Check if we've completed a code block delimiter
            if self.num_curr_ticks_delimiter >=3:
                self.is_incomplete_delimiter = False
                self.num_curr_ticks_delimiter = 0

                if not self.is_code:  # If it's the start of a code snippet
                    self.is_code = True
                    if self.token_buffer:  # If there's text in the buffer
                        self.sentence_buffer += ''.join(self.token_buffer) # Add it to the sentence buffer
                        self.token_buffer.clear()
                    self.sentence_buffer += " I'm writing the code to the window now..." # Add a message to the sentence buffer
                    threaded_generate_and_play(self.sentence_buffer, self.voice) # Generate and play the sentence buffer
                    self.sentence_buffer = "" # Clear the sentence buffer
                else:  # If it's the end of a code snippet
                    self.is_code_end = True

        if self.is_code:  # If the LLM is currently outputting a code snippet
            self.code_buffer += token

            # Send code buffer if necessary

            if self.is_code_end:  # If it's the end of the code snippet
                self.is_code = False
                self.is_code_end = False
        else: # 
            self.token_buffer.append(token)
            if token.endswith(('.', '?', '!', '"', '\n', ':')):
                self.sentence_buffer += ''.join(self.token_buffer)
                self.token_buffer.clear()
            if self.sentence_buffer and audio_queue.qsize() == 0 and not is_playing_audio():
                threaded_generate_and_play(self.sentence_buffer, self.voice)
                self.sentence_buffer = ""
        
        
        
        
        
        
        

class XILabsCallbackHandler(BaseCallbackHandler):
    def __init__(self, voice):
        self.voice = voice
        self.token_buffer = []
        self.sentence_buffer = ""
        self.code_buffer = ""
        self.is_code = False
        self.is_code_end = False

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        #cl.start_stream()

        global is_running
        with is_running_lock:
            is_running = True

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        #cl.send_token(token)
        
        terminal_width = get_terminal_width()
        display_text = ''.join(self.token_buffer)
        if len(display_text) > terminal_width:                                   
            display_text = display_text[-terminal_width:]
        # if token.startswith("`"): # Check if token starts with a backtick
        #     if self.num_curr_ticks_delimiter == 0: # Check if it's a new delimiter
        #         self.is_incomplete_delimiter = True # Set incomplete delimiter flag
        #         self.num_curr_ticks_delimiter = len(token) # Update current ticks

        #     else: # If we're in the middle of processing an incomplete delimiter
        #         self.num_curr_ticks_delimiter += len(token) # Update current ticks

        #     # Check if we've completed a code block delimiter
        #     if self.num_curr_ticks_delimiter >=3:
        #         self.is_incomplete_delimiter = False
        #         self.num_curr_ticks_delimiter = 0

        if token in ['```', '`\n\n']: # Check if token is a code block delimiter

            if not self.is_code:  # If it's the start of a code snippet
                self.is_code = True
                if self.token_buffer:  # If there's text in the buffer
                    self.sentence_buffer += ''.join(self.token_buffer) # Add it to the sentence buffer
                    self.token_buffer.clear()
                self.sentence_buffer += " I'm writing the code to the window now." # Add a message to the sentence buffer
                threaded_generate_and_play(self.sentence_buffer, self.voice) # Generate and play the sentence buffer
                self.sentence_buffer = "" # Clear the sentence buffer

            else:  # If it's the end of a code snippet
                self.is_code_end = True

        if self.is_code:  # If the LLM is currently outputting a code snippet
            self.code_buffer += token

            # Send code buffer

            if self.is_code_end:  # If it's the end of the code snippet
                self.is_code = False
                self.is_code_end = False
        else: # 
            self.token_buffer.append(token)
            if token.endswith(('.', '?', '!', '"', '\n', ':')):
                self.sentence_buffer += ''.join(self.token_buffer)
                self.token_buffer.clear()
            if self.sentence_buffer and audio_queue.qsize() == 0 and not is_playing_audio():
                threaded_generate_and_play(self.sentence_buffer, self.voice)
                self.sentence_buffer = ""

    def on_llm_end(self, response, **kwargs) -> None:
        global is_running
        with is_running_lock:
            is_running = False
        if self.sentence_buffer:
            threaded_generate_and_play(self.sentence_buffer, self.voice)
            sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')  # Clear the line
            sys.stdout.flush()
            self.sentence_buffer = ""

def stream_callback(in_data, frame_count, time_info, status):
    global frames
    frames.append(in_data)
    return (in_data, pyaudio.paContinue)



def start_recording():
    global stream, is_recording
    if stream is None or not stream.is_active():
        sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
        with is_recording_lock:
            is_recording = True
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=stream_callback)
        stream.start_stream()
        

def stop_recording():
    global stream, recording_done, is_recording, p
    if stream is not None and stream.is_active():
        with is_recording_lock:
            is_recording = False
        stream.stop_stream()
        stream.close()
        stream = None
        recording_done.set()


def get_voice_input():
    global frames, stream, recording_done, p
    hotkey = '`'
    frames = []
    p = pyaudio.PyAudio()
    # Create a temporary file for the recording
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav", dir='./record_outputs')
    WAVE_OUTPUT_FILENAME = temp_file.name  # Use the temporary file's name

    sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
    sys.stdout.flush()

    # Open a new stream for recording
    print()
    sys.stdout.write(Fore.LIGHTBLUE_EX + 'Press the key to start recording and release to stop recording' + Fore.RESET)
    sys.stdout.flush()
    # Record audio while the space key is pressed
    # while True:
    #     if keyboard.is_pressed('f13'):
    #         if not set_is_recording:
    #             sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
    #             with is_recording_lock:
    #                 is_recording = True
    #             set_is_recording = True
    #         data = stream.read(CHUNK)
    #         frames.append(data)
    #     elif not keyboard.is_pressed('f13') and len(frames) > 0:
    #         with is_recording_lock:
    #             is_recording = False
    #         set_is_recording = False
    #         break

    # Register key press and release events
    keyboard.add_hotkey(hotkey, start_recording, suppress=True)
    keyboard.add_hotkey(hotkey, stop_recording, trigger_on_release=True, suppress=True)

    # Wait for the recording to be done
    recording_done.wait()
    recording_done.clear()

    # Unregister key press and release events
    keyboard.remove_hotkey(start_recording)
    p.terminate()

    sys.stdout.flush()

    temp_file.close()
    # Save the recorded audio to a file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_file= open(WAVE_OUTPUT_FILENAME, "rb") 

    transcript = openai.Audio.transcribe("whisper-1", audio_file, api_key=os.getenv('OPENAI_API_KEY'))
    
    
    return transcript.text

system_template = """You are Assistant. A powerful, intelligent, and capable AI personal assistant created by Ryan Heaton.

Assistant is powered by a large language model trained by OpenAI and fine-tuned by Ryan Heaton.

Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant has opinions and preferences of their own, and they should feel free to express them. Assistant engages with the human as if Assistant was also a human, and are talking to a long time friend.

Assistant is aware that human input is being transcribed from audio and as such there may be some errors in the transcription. It will attempt to account for some words being swapped with similar-sounding words or phrases. Assistant will also keep responses concise, because human attention spans are more limited over the audio channel since it takes time to listen to a response.

Assistant may receive messages from their internal system, which should be used for context and general information. These additional messages, which may contain specific information about how to act or what to say, are marked with the tag <SYSTEM>. Do not read these messages out loud, but always use them to inform your responses.

Assistant has access to the user's clipboard and files on the local machine. Relevant context from these sources may be appended to the user's query to provide additional context to the Assistant.
"""

human_template = """{history}
{human_input}
"""

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content=system_template),
    HumanMessagePromptTemplate.from_template(human_template),
])

memory=ConversationBufferWindowMemory(return_messages=True)

chat_chain = LLMChain(
    llm=ChatOpenAI(temperature=0.5, model_name='gpt-4', streaming=True, callbacks=[XILabsCallbackHandler(voice='BBC')]),
    prompt=prompt_template,
    memory=memory,
)

def save_context_to_file(context, file_path='context.json'):
    with open(file_path, 'w') as f:
        json.dump(context, f)

def load_context_from_file(file_path='context.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            context = json.load(f)
    else:
        context = {}
    return context

# Start the spinning icon display thread
threading.Thread(target=display_spinning_icon, daemon=True).start()
# Start the audio playback thread
threading.Thread(target=play_audio_from_queue, daemon=True).start()

context = {}

def start_voice_input():
    state = None
    while True:
        context = load_context_from_file()
        transcript = get_voice_input()
        (prepped_transcript, edits, state) = prep_all_inputs(transcript, memory, context, state)
        edits_str = "|".join(k for k in edits.keys() if edits[k] == True)
        print(Fore.CYAN + "Human:" + Fore.RESET, transcript, Fore.RED + "Edits:" + Fore.RESET, edits_str)
        print()
        response = chat_chain.run(prepped_transcript)
        while True:
            if is_playing_audio():
                time.sleep(0.1)
            else:
                break
        sys.stdout.write('\r' + ' ' * get_terminal_width() + '\r')
        sys.stdout.flush()
        print(Fore.GREEN + "Assistant:" + Fore.RESET, response)
        save_context_to_file(context)

if __name__ == "__main__":
    load_dotenv()
    # flask_thread = threading.Thread(target=start_flask_app)
    voice_input_thread = threading.Thread(target=start_voice_input, daemon=True)
    # flask_thread.start()
    voice_input_thread.start()
    # flask_thread.join()
    voice_input_thread.join()