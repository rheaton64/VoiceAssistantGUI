import keyboard
import pyperclip
import threading
import os
import time
from queue import Queue
from elevenlabs import generate, stream, set_api_key, voices
from dotenv import load_dotenv
load_dotenv()

clipboard_queue = Queue()

def add_to_queue():
    clipboard_content = pyperclip.paste()
    clipboard_queue.put(clipboard_content)
    print(f'Added to queue: {clipboard_content}')

def play_audio_from_queue():
    while True:
        if clipboard_queue.qsize() == 0:
            time.sleep(0.1)
            continue
        audio = clipboard_queue.get()
        print(f'Playing audio: {audio}')
        stream(generate(
            text=audio,
            voice='Josh',
            stream=True
        ))
        clipboard_queue.task_done()

if __name__ == '__main__':
    set_api_key(os.getenv('XILABS_API_KEY'))
    voices()
    print('Listening for hotkey')
    threading.Thread(target=play_audio_from_queue, daemon=True).start()
    hotkey = 'ctrl+shift+a'
    keyboard.add_hotkey(hotkey, add_to_queue, suppress=True)
    keyboard.wait()
