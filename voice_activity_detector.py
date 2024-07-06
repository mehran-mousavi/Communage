import webrtcvad
import collections
import pyaudiowpatch as pyaudio
import wave
import numpy as np
import soundcard as sc
import threading
import signal
import time
from array import array
from struct import pack
from abc import ABC, abstractmethod
from scipy.signal import resample

class DeviceSelectionStrategy(ABC):
    """
    Abstract base class for device selection strategies.
    """
    @abstractmethod
    def select_device(self, pyaudio_instance):
        """
        Selects and returns a dictionary containing device information.
        """
        pass

class DefaultDeviceStrategy(DeviceSelectionStrategy):
    """
    Selects the default audio device.
    """
    def select_device(self, pyaudio_instance):
        """
        Returns a dictionary with information about the default audio device.
        """
        default_device_info = pyaudio_instance.get_default_input_device_info()
        return {
            "index": default_device_info["index"],
            "channels": 1,
            "frame_rate": 16000,
        }

class LoopbackDeviceStrategy(DeviceSelectionStrategy):
    """
    Selects the default loopback device or searches for a suitable one.
    """
    def select_device(self, pyaudio_instance):
        """
        Returns a dictionary with information about the selected loopback device.
        """
        wasapi_info = pyaudio_instance.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = pyaudio_instance.get_device_info_by_index(
            wasapi_info["defaultOutputDevice"]
        )

        if not default_speakers["isLoopbackDevice"]:
            for loopback in pyaudio_instance.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break
            else:
                raise RuntimeError("No suitable loopback device found.")

        return {
            "index": default_speakers["index"],
            "channels": default_speakers["maxInputChannels"],
            "frame_rate": int(default_speakers["defaultSampleRate"]),
        }

class VoiceActivityDetector:
    def __init__(self, device_strategy, rate=16000, chunk_duration_ms=30, padding_duration_ms=1500, vad_level=1):
        self.RATE = rate
        self.CHUNK_DURATION_MS = chunk_duration_ms
        self.CHUNK_SIZE = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        self.NUM_PADDING_CHUNKS = int(padding_duration_ms / chunk_duration_ms)
        self.NUM_WINDOW_CHUNKS = int(400 / chunk_duration_ms)
        self.NUM_WINDOW_CHUNKS_END = self.NUM_WINDOW_CHUNKS * 2
        self.vad = webrtcvad.Vad(vad_level)

        self.pa = pyaudio.PyAudio()
        device_info = device_strategy.select_device(self.pa)
        self.input_channels = device_info["channels"]
        self.input_rate = device_info["frame_rate"]
        
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=self.input_channels,
            rate=self.input_rate,
            input=True,
            input_device_index=device_info["index"],
            start=False,
            frames_per_buffer=self.CHUNK_SIZE,
        )

        self.got_a_sentence = False
        self.leave = False
        signal.signal(signal.SIGINT, self.handle_int)

    def handle_int(self, sig, chunk):
        self.leave = True
        self.got_a_sentence = True

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype("float32")
        if abs_max > 0:
            sound *= 1 / 32768
        return sound.squeeze()

    def play_audio(self, audio_buffer):
        speaker = sc.default_speaker()
        _audio_data_np = self.int2float(np.frombuffer(audio_buffer, np.int16))
        speaker.play(_audio_data_np, samplerate=self.RATE)

    def resample_and_convert_chunk(self, chunk):
        audio_data = np.frombuffer(chunk, dtype=np.int16)
        if self.input_channels > 1:
            audio_data = audio_data.reshape(-1, self.input_channels)
            audio_data = audio_data.mean(axis=1).astype(np.int16)

        if self.input_rate != self.RATE:
            audio_data = resample(audio_data, int(len(audio_data) * self.RATE / self.input_rate)).astype(np.int16)
            return audio_data.tobytes()
        
        return chunk

    def vad_generator(self, on_start_detection, on_end_detection):
        ring_buffer = collections.deque(maxlen=self.NUM_PADDING_CHUNKS)
        triggered = False
        voiced_frames = []
        ring_buffer_flags = [0] * self.NUM_WINDOW_CHUNKS
        ring_buffer_index = 0

        ring_buffer_flags_end = [0] * self.NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end = 0

        self.stream.start_stream()
        while not self.leave:
            chunk = self.stream.read(self.CHUNK_SIZE)
            chunk = self.resample_and_convert_chunk(chunk)
            active = self.vad.is_speech(chunk, self.RATE)

            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index = (ring_buffer_index + 1) % self.NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end = (ring_buffer_index_end + 1) % self.NUM_WINDOW_CHUNKS_END

            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * self.NUM_WINDOW_CHUNKS:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    on_start_detection()
                    ring_buffer.clear()
            else:
                voiced_frames.append(chunk)
                ring_buffer.append(chunk)
                num_unvoiced = self.NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                if num_unvoiced > 0.90 * self.NUM_WINDOW_CHUNKS_END:
                    triggered = False
                    on_end_detection()
                    yield b"".join(voiced_frames)
                    voiced_frames = []

        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    def close(self):
        self.leave = True


def main():
    device_strategy =  LoopbackDeviceStrategy() #DefaultDeviceStrategy()
    vad = VoiceActivityDetector(device_strategy)
    print("Started...")
    
    def on_start():
        print("Started...")
        
    def on_end():
        print("Ended...")
    
    for sentence in vad.vad_generator(on_start, on_end):
        print("Detected sentence")
        threading.Thread(target=vad.play_audio, args=(sentence,)).start()

    vad.close()

if __name__ == "__main__":
    main()
