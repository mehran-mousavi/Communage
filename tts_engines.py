# import soundcard for playback
import io
from edge_tts import Communicate
import torchaudio
import asyncio  # noqa: F811
import json


class TTSEngine:
    """
    Abstract base class for Text-to-Speech engines.
    """

    def text_to_speech(self, text, lang):
        """
        Converts the given text to speech and returns the audio data and sample rate.

        Args:
            text (str): The text to be converted to speech.

        Returns:
            dict: A dictionary with the following keys:
                "audio" (torch.Tensor): The generated audio data as a PyTorch tensor.
                "samplerate" (int): The sample rate of the generated audio.
        """
        raise NotImplementedError


class EdgeTTSEngine(TTSEngine):
    def __init__(self):
        self._load_voice_mapping()

    def _load_voice_mapping(self):
        with open("resources/settings.json", "r") as settings_file:
            settings = json.load(settings_file)
            self.language_and_voice_mapping = settings.get("edge_tts_voice_mapping", {})

    async def _convert_async(self, text, voice):
        print(text + " *** " + voice)
        communicate = Communicate(text, voice)
        audio_data = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        audio_data.seek(0)
        return audio_data

    def text_to_speech(self, text, lang):
        print("Generating speech from text...")

        if lang:
            print(lang)
            voice = self.language_and_voice_mapping.get(
                lang, {"male": "en-US-SteffanNeural", "female": "en-US-MichelleNeural"}
            )["male"]
            print(voice)
        else:
            print("Voice Not Found...")
            return

        # Set up the event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Run the coroutine in an event loop until it is complete
        audio_data = loop.run_until_complete(self._convert_async(text, voice))
        print("Speech generated successfully...")

        # Load the audio data from the BytesIO object
        audio_tensor, sample_rate = torchaudio.load(audio_data, format="mp3")

        # Resample the audio to 16000 Hz
        # audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)

        return {
            "audio": audio_tensor.squeeze(),
            "samplerate": sample_rate,
        }
