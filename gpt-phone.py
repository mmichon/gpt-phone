#!/usr/bin/python3

import platform
import logging
from colorlog import ColoredFormatter

OS = platform.system()
if OS == "Linux":
    from gpiozero import Button

import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import speech_recognition as sr
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from pydub import AudioSegment
import pydub.playback
import os, sys, contextlib # for ALSA lib errors -- https://stackoverflow.com/questions/7088672/pyaudio-working-but-spits-out-error-messages-each-time


# Flag to enable or disable speech output
SPEAK = True

# Flag to skip the dialing process for testing purposes
SKIP_DIALING = False

# Voice ID for the operator
OPERATOR_VOICE_ID = "qHR09fcvu6SoDtFzqFvm" # Ryan (articulate, friendly, conversational)
# Greeting message from the operator
OPERATOR_GREETING = "Please dial a single digit to proceed. For a directory, please dial 0."
# Message for wrong number
WRONG_NUMBER = "That number is disconnected. Please hang up and try again."

# Speech recognition configuration
LISTEN_TIMEOUT = 5     # Time to wait for speech before giving up
PHRASE_TIMEOUT = 2    # Space between recordings for sepating phrases
# SPEECH_SPEED = 2#.5

# API clients
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_KEY")
ELEVENLABS_MODEL_ID = "eleven_turbo_v2_5"
# GCP_CRED_FILE = "gcp-creds.json"
CHATGPT_MODEL = "gpt-3.5-turbo"

# Hardware configuration
# DEVICE_ID = None # speaker device
DYNAMIC_ENERGY_THRESHOLD = True
HOOK_GPIO = 14
DIAL_GPIO = 15


# Set up logging
logging.root.setLevel(logging.DEBUG)
formatter = ColoredFormatter(
    "%(log_color)s%(asctime)s %(reset)s%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    reset=True,
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'red',
    },
    secondary_log_colors={
        'message': {
            'asctime': 'grey',
        }
    }
)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

openai_client = OpenAI()

r = sr.Recognizer()

if OS == "Linux":
    hook = Button(HOOK_GPIO, pull_up=False)
    dial = Button(DIAL_GPIO, pull_up=False)

@contextlib.contextmanager
def ignoreStderr(): # quiets pyaudio/jackd errors on Linux
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    sys.stderr.flush()
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)

def sendchat(prompt, system_role):
    """
    Sends a chat prompt to the OpenAI API and returns the response.

    Args:
        prompt (str): The prompt to send to the API.
        system_role (str): The system role to send to the API.

    Returns:
        str: The response from the API.
    """
    completion = openai_client.chat.completions.create(
        model=CHATGPT_MODEL,
        messages=[
            {"role": "system", "content": system_role},
            {"role": "user", "content": prompt},
        ],
    )

    response = completion.choices[0].message.content.strip()

    return response

class Phone:
    def __init__(self):
        """
        Initializes the Phone object and loads the ElevenLabs client.
        """
        logger.info("Initializing phone object and speech API")
        load_dotenv()

        self.client = ElevenLabs(
            api_key=ELEVENLABS_API_KEY,
        )

    def speak(self, voice, text):
        """
        Converts text to speech and plays the audio.

        Args:
            text (str): The text to convert to speech.
            voice (str): The voice ID to use for the speech.
        """
        logger.info("Speaking '%s'", text)
        speech_start = time.time()
        if SPEAK:
            try:
                logger.info("Generating audio from text")
                audio_stream = self.client.text_to_speech.convert_as_stream(
                    text=text,
                    voice_id=voice,
                    model_id=ELEVENLABS_MODEL_ID,
                )
            except Exception as e:
                logger.error("An error occurred in speech generation: %s", e)
                return None

            logger.debug("T2S took %s seconds", str(time.time() - speech_start))
            stream(audio_stream)

    def speak_directory(self, roles):
        """
        Speaks the directory for the given list of roles.

        Args:
            roles (list of PhoneRole): The list of roles to speak.
        """
        prompt = ""
        for i, role in enumerate(roles):
            if role is not None:
                prompt = prompt + (f"For {role.name}, dial {i}. ")
        self.speak(OPERATOR_VOICE_ID, prompt)

    def read_dial(self):
        """
        Reads the dialed digit from the rotary dial.

        Returns:
            int: The digit that was dialed.
        """
        digit = -1

        while True:
            dial.wait_for_inactive()
            digit = digit + 1
            time.sleep(.01)
            start_time = time.time()
            dial.wait_for_active(1)
            if time.time() - start_time > 0.1:
                break

        if digit == 10:
            digit = 0

        return digit

    def answer_phone(self, role):
        """
        Answers the phone and processes the call.

        Args:
            role (PhoneRole): The role that was dialed.
        """
        logger.info("Determining ambient noise level")
        with ignoreStderr(): # kills stderr
            with sr.Microphone() as source:
                r.dynamic_energy_threshold = DYNAMIC_ENERGY_THRESHOLD
                r.adjust_for_ambient_noise(source, 2)  # listen for a few seconds to calibrate the energy threshold for ambient noise levels
        logger.debug("Energy threshold: %s"  % r.energy_threshold)

        transcription = []  # reset memory at each call

        self.speak(role.voice_id, role.greeting)

        while OS != "Linux" or hook.value == 1:  # if on Mac or back on hook, return
            try:
                logger.info("Listening...")
                with ignoreStderr(): # kills stderr
                    with sr.Microphone() as source:
                        audio = r.listen(source, timeout=LISTEN_TIMEOUT, phrase_time_limit=PHRASE_TIMEOUT)

                logger.info("Analyzing heard audio")
                audio_start = time.time()

                try:
                    # text = r.recognize_google_cloud(audio, GCP_CRED_FILE)
                    text = r.recognize_openai(audio, model = "whisper-1")
                    logger.info("Transcribed speech: '%s'", text)
                    logger.debug("Recognition took %s seconds", str(time.time() - audio_start))

                    if text:
                        gpt_start = time.time()
                        logger.info("Sending speech to ChatGPT")

                        transcription.append({"role": "user", "content": text})

                        prompt = "\n".join([message["content"] for message in transcription])
                        # logger.info("About to send prompt:\n%s", prompt)
                        chat_response = sendchat(prompt, role.system_role)

                        logger.debug("GPT took %s seconds", str(time.time() - gpt_start))

                        transcription.append({"role": "assistant", "content": chat_response})

                        self.speak(role.voice_id, chat_response)
                    else:
                        logger.warning("Couldn't transcribe audio")
                        self.speak(role.voice_id, "What was that?")

                except sr.RequestError:
                    # API was unreachable or unresponsive
                    logger.error("Speech recognition API unavailable")

                except sr.UnknownValueError:
                    logger.warning("Unable to recognize speech")
                    self.speak(role.voice_id, "What was that?")

            except sr.WaitTimeoutError:
                logger.warning("Timed out waiting for speech")
                if (hook.value == 0): # Caller hung up
                    logger.info("Caller hung up")
                    return
                else:
                    self.speak(role.voice_id, "You still there?")

            except KeyboardInterrupt:
                break

            except SystemExit:
                break


def main():
    """
    Main function to start the phone and wait for off hook events.
    """
    logger.info("Starting up. OS is %s.", OS)

    roles = [None, role_elf, None, role_fred, None, None, role_devil, role_god, role_mike, None]

    phone = Phone()
    logger.info("Initialized phone")

    while True:
        if SKIP_DIALING or OS != "Linux":
            phone.answer_phone(role_mike) # Answer by default as Mike
            break

        logger.info("Waiting for hook event...")
        if hook.value == 0:
            hook.wait_for_press()

        logger.info("Someone picked up the phone")

        time.sleep(2) # Delay to make sure the operator is heard in time
        phone.speak(OPERATOR_VOICE_ID, OPERATOR_GREETING)

        dial.wait_for_press(timeout=10)
        if dial.value == 1:
            digit = phone.read_dial()
            logger.info("Decoded digit: %s", digit)

            # if digit is 0, speak the directory
            if digit == 0 or SKIP_DIALING == True:
                phone.speak_directory(roles)
                continue

            if 1 <= digit < len(roles) and roles[digit] is not None:
                role = roles[digit]
                time.sleep(1)

                if SKIP_DIALING == False and SPEAK == True:
                    dialtone_sound = AudioSegment.from_mp3(role.dialtone_file)
                    pydub.playback.play(dialtone_sound)

                phone.answer_phone(role)
            else:
                phone.speak(OPERATOR_VOICE_ID, WRONG_NUMBER)
                logger.info("Waiting for hangup...")
                hook.wait_for_inactive()

if __name__ == "__main__":
    required_env_vars = ['ELEVENLABS_KEY', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_env_vars if var not in os.environ]

    if missing_vars:
        logger.error("Missing environment variables: %s", ", ".join(missing_vars))
        sys.exit(1)

    # Load roles definitions
    exec(compile(open("roles.py", "rb").read(), "roles.py", 'exec'))

    main()