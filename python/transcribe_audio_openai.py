import argparse
import datetime
import os
import subprocess
import sys
from io import BytesIO

import openai
from pydub import AudioSegment

"""
This script transcribes an audio file using OpenAI's Whisper API.

Usage:
    python transcribe.py -f <audio_file> [-l <language>] [-s <save_directory>] [-k <api_key>] [-o]

Options:
    -f, --file <audio_file>       Path to the audio file to be transcribed. (Required)
    -l, --language <language>     Language code for transcription. Default: "pl" (Polish).
    -s, --save <save_directory>   Directory to save the transcription file.
                                  - If not provided, the file is saved in the same directory as the audio file.
                                  - If provided as an empty string, it is saved in the current working directory.
    -k, --api-key <api_key>       OpenAI API key for this call. Overrides the environment variable if provided.
    -o, --open                    Opens the final transcription file in the default OS text editor.

Examples:
    Basic transcription:
        python transcribe.py -f "./audio.mp3"

    Specify a language (English):
        python transcribe.py -f "./audio.mp3" -l "en"

    Save the transcription in a specific directory:
        python transcribe.py -f "./audio.mp3" -s "/path/to/save"

    Use a custom OpenAI API key for this call:
        python transcribe.py -f "./audio.mp3" -k "your_api_key_here"

    Automatically open the transcription file after completion:
        python transcribe.py -f "./audio.mp3" -o

Mind that the script requires the OpenAI Python library and the pydub library to work properly as well as the OpenAI API key.
"""


# Set default OpenAI API key to None so that the library uses the environment variable if available.
OPENAI_API_KEY = None

# Maximum allowed file size in bytes (25 MB)
MAX_FILE_SIZE = 26214400

def convert_to_wav(input_file):
    """Converts any audio file to WAV (16kHz mono, 16-bit) using pydub."""
    output_file = os.path.splitext(input_file)[0] + ".wav"
    audio = AudioSegment.from_file(input_file)
    # Explicitly set sample width to 2 bytes (16-bit) for consistency.
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(output_file, format="wav")
    return output_file

def transcribe_audio(file_path, language):
    """Transcribes an audio file using OpenAI Whisper API.

    If the file exceeds the maximum allowed size, it splits the audio into chunks.
    """
    # Convert to WAV if needed
    if not file_path.lower().endswith(".wav"):
        file_path = convert_to_wav(file_path)

    file_size = os.path.getsize(file_path)
    if file_size <= MAX_FILE_SIZE:
        with open(file_path, "rb") as audio_file:
            response = openai.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                language=language
            )
        return response.text
    else:
        print("File size exceeds limit; splitting audio into smaller chunks...")
        return transcribe_audio_chunked(file_path, language)

def transcribe_audio_chunked(file_path, language):
    """Splits the audio into chunks and transcribes each chunk separately."""
    audio = AudioSegment.from_file(file_path)
    # Define chunk duration (in milliseconds). 800 sec is a safe bet.
    chunk_duration_ms = 800 * 1000
    total_duration = len(audio)
    transcript = ""
    num_chunks = (total_duration // chunk_duration_ms) + (1 if total_duration % chunk_duration_ms > 0 else 0)

    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, total_duration)
        audio_chunk = audio[start_ms:end_ms]
        chunk_text = transcribe_audio_chunk(audio_chunk, language)
        transcript += f"\n\n--- Transcription chunk {i+1} ---\n" + chunk_text

    return transcript

def transcribe_audio_chunk(audio_chunk, language):
    """Transcribes a single audio chunk from an AudioSegment object."""
    buffer = BytesIO()
    # Export the chunk to a WAV in-memory file
    audio_chunk.export(buffer, format="wav")
    # Set a name attribute so the API can detect the file type correctly.
    buffer.name = "chunk.wav"
    buffer.seek(0)
    response = openai.audio.transcriptions.create(
        file=buffer,
        model="whisper-1",
        language=language
    )
    return response.text

def save_transcription(text, original_file, save_dir=None):
    """Saves transcription to a timestamped text file with a descriptive name.

    - If save_dir is None: saves in the same directory as the original audio file.
    - If save_dir is an empty string: saves in the current working directory.
    - Otherwise, saves in the provided directory.
    """
    base_name = os.path.splitext(os.path.basename(original_file))[0]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{base_name}_transcription_{timestamp}.txt"

    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(original_file))
    elif save_dir == "":
        save_dir = os.getcwd()

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    return file_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe an audio file using OpenAI API")
    parser.add_argument("-f", "--file", required=True, help="Path to the audio file")
    parser.add_argument("-l", "--language", default="pl", help="Language code (default: pl)")
    parser.add_argument("-s", "--save", nargs="?", default=None,
                        help=("Directory to save the transcription file. If provided as an empty string, "
                              "uses current working directory. If not provided, uses directory of audio file."))
    parser.add_argument("-k", "--api-key", default=None,
                        help="OpenAI API key for this call (overrides environment variable if provided)")
    parser.add_argument("-o", "--open", action="store_true", default=False,
                        help="Open the final file in default OS text editor")

    args = parser.parse_args()

    # Set the API key if provided, otherwise let OpenAI library use the environment variable
    openai.api_key = args.api_key if args.api_key else OPENAI_API_KEY

    transcript = transcribe_audio(args.file, args.language)
    output_file = save_transcription(transcript, args.file, save_dir=args.save)

    print(f"\nTranscription saved to: {output_file}")
    if args.open:
        try:
            if sys.platform.startswith("darwin"):
                subprocess.run(["open", output_file])
            elif os.name == "nt":
                os.startfile(output_file)
            elif os.name == "posix":
                subprocess.run(["xdg-open", output_file])
        except Exception as e:
            print(f"Failed to open the file: {e}")
