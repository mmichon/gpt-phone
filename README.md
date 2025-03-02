# GPT Phone

GPT Phone is a Python-based project that simulates a phone call with various characters using OpenAI's GPT-3.5 and ElevenLabs for text-to-speech. The project is designed to run on a Raspberry Pi with a physical phone interface.

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
    pip3 install --upgrade SpeechRecognition openai mpv google-cloud-speech elevenlabs pydub libpulse-dev pulseaudio apulse pyaudio build-essential libssl-dev libasound2 wget flac gst-1.0 mpg123 gpiozero dotenv mpv ffmpeg
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

To run the GPT Phone as a systemd service, create a service file:

1. Create [phone.service](http://_vscodecontentref_/3):
    ```sh
    sudo nano /etc/systemd/system/phone.service
    ```

2. Add the following content:
    ```ini
    [Unit]
    Description=phone service
    After=network-online.target

    [Service]
    Type=simple
    User=root
    ExecStart=sh -c "/home/pi/start-phone.sh"
    Restart=on-failure
    Environment=PYTHONUNBUFFERED=1

    [Install]
    WantedBy=multi-user.target
    ```

3. Enable and start the service:
    ```sh
    sudo systemctl enable phone.service
    sudo systemctl start phone.service
    ```

## License

This project is licensed under the MIT License.