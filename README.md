# GPT Phone

GPT Phone is a Python-based project that simulates a phone call with various characters using OpenAI's GPT-3.5 and ElevenLabs for text-to-speech. The project is designed to run on a Raspberry Pi with a physical phone interface. It also interfaces via GPIO pins to a Bell 304 series telephone's hardware including the rotary dial switch, hook switch, handset speaker, and handset microphone.

## Installation

1. Create a virtual environment:
    ```sh
    python3 -m venv gpt-phone
    source gpt-phone/bin/activate
    ```

2. Install system dependencies:
    ```sh
    sudo apt install portaudio19-dev pipewire-audio-client-libraries
    ```

3. Install Python dependencies:
    ```sh
    pip3 install -r requirements.txt
    ```

## Hardware

1. Connect the hook switch to GPIO14 (or change the pin definition in `gpt-phone.py`) and ground
2. Connect the rotary dial switch to GPIO15 (or change the pin definition in `gpt-phone.py`) and ground
3. Connect your audio device's microphone to the phone's microphone circuit
4. Connect your audio device's speaker to the phone's speaker circuit

## Configuration

- **Personality**: Personalize the system roles, greetings, etc in [roles.py](http://_vscodecontentref_/0) file.
- **Voices**: Set the voice IDs for ElevenLabs in [roles.py](http://_vscodecontentref_/1) array.
- **Dialtone Files**: Optionally, change the dialtone files in [roles.py](http://_vscodecontentref_/2) array.
- **

## Usage

1. Set the environment variables `OPENAI_API_KEY` and `ELEVENLABS_KEY`:
    ```sh
    export OPENAI_API_KEY="your_openai_api_key"
    export ELEVENLABS_KEY="your_elevenlabs_api_key"
    export TEST_DIGIT=1 # Optional. Forces a particular role to answer the phone, for testing purposes.
    ```

2. Run the script:
    ```sh
    python3 gpt-phone.py
    ```

## Systemd Service

To run the GPT Phone as a systemd service, create a service file and enable it:

1. Enable and start the service:
    ```sh
    sudo cp phone.service /etc/systemd/systemd
    sudo systemctl --reload-daemon
    sudo systemctl enable phone.service
    sudo systemctl start phone.service
    loginctl enable-linger # this allows pulseaudio/pipewire to load upon boot
    ```

## License

This project is licensed under the MIT License.