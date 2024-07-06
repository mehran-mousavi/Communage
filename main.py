import sys
import time
import queue
import numpy as np
import asyncio
from google_api import SentenceTranslator, SpeechRecognizer
from tts_engines import EdgeTTSEngine
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QLabel,
    QWidget,
    QPushButton,
    QTextEdit,
    QComboBox,
    QHBoxLayout,
    QSizePolicy,
    QStackedLayout,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt5.QtGui import QColor, QFont, QTextCursor, QTextOption, QMovie, QIcon
import soundcard as sc
from utils import pre_process_audio, int2float, get_languages, get_gui_dark_stylesheet
from voice_activity_detector import (
    VoiceActivityDetector,
    LoopbackDeviceStrategy,
    DefaultDeviceStrategy,
)


# ----- Abstract Base Classes for Audio Stream Handling and Audio Stream Handlers -----
class AbstractStreamHandler(QThread):
    """
    Abstract base class for handling audio streams (microphone and speaker).
    """

    new_translation_signal = pyqtSignal(str)
    update_waveform_signal = pyqtSignal(bool)
    initialized = False

    def __init__(
        self, vad_audio: VoiceActivityDetector, processor, rate=16000, max_chunk_duration=10
    ):
        super().__init__()
        self.vad_audio = vad_audio
        self.rate = rate
        self.frames = self.vad_audio.vad_generator(self.on_start, self.on_end)

        self.processor = processor
        self.max_chunk_duration = max_chunk_duration

        self.initialized = False
        self.running = True
        
    def on_start(self):
        self.update_waveform_signal.emit(True)
        
    def on_end(self):
        self.update_waveform_signal.emit(False)

    def run(self):
        self.initialized = True  # Set to True after initialization
        self.processor.start()

        try:
            for sentence in self.frames:
                # threading.Thread(target=self.vad_audio .play_audio, args=(sentence,)).start()
                self.process_wav_data(sentence)
                
                # print(sentence)
                
        except Exception as e:
            print("AbstractStreamHandler : " + str(e))
        finally:
            self.stop_processing()

    def process_wav_data(self, wav_data):
        if not wav_data:
            return
        new_sound = np.frombuffer(wav_data, np.int16)
        audio_float32 = int2float(new_sound)
        self.processor.process(audio_float32)

    def stop_processing(self):
        try:
            self.running = False

            if self.processor:
                self.processor.stop()
                self.processor = None

            if self.vad_audio:
                self.vad_audio.close()
                self.vad_audio = None

            self.requestInterruption()
            self.quit()  # Ensure the thread event loop is stopped
        except Exception as e:
            print(e)


class MicrophoneStreamHandler(AbstractStreamHandler):
    """
    Handles the microphone audio stream, including VAD and processing.
    """

    def __init__(self, src_lang, dst_lang, tts_engine):
        device_strategy = DefaultDeviceStrategy()
        vad_audio = VoiceActivityDetector(device_strategy)

        processor = MicrophoneProcessor(
            src_lang, dst_lang, tts_engine, 16000
        )
        super().__init__(vad_audio, processor, 16000)
        processor.new_translation_signal.connect(self.new_translation_signal)


class SpeakerStreamHandler(AbstractStreamHandler):
    """
    Handles the speaker audio stream, including VAD and processing.
    """

    def __init__(self, src_lang, dst_lang):
        device_strategy = LoopbackDeviceStrategy()
        vad_audio = VoiceActivityDetector(device_strategy)

        processor = SpeakerProcessor(src_lang, dst_lang, 16000)
        super().__init__(vad_audio, processor, 16000)
        processor.new_translation_signal.connect(self.new_translation_signal)


# ----- Abstract Base Classes for Processors -----
class AbstractProcessor(QThread):
    """
    Abstract base class for audio processing threads (microphone and speaker).
    """

    new_translation_signal = pyqtSignal(str)

    def __init__(self, src_lang, dst_lang, rate=16000, on_translated=None):
        super().__init__()
        self.queue = queue.Queue()
        self.transcriber = SpeechRecognizer(language=src_lang)
        self.translator = SentenceTranslator(src=src_lang, dst=dst_lang)
        self.running = True
        self.rate = rate
        self.on_translated = on_translated

    def run(self):
        pass

    async def async_run(self):
        while self.running:
            if not self.queue.empty():
                try:
                    audio_data = self.queue.get_nowait()
                    processed_audio = pre_process_audio(audio_data)
                    
                    transcription = await self.transcriber(processed_audio)
                    if transcription and transcription.strip() != "":
                        translation = await self.translator(transcription)
                        self.process_result_of_queue_processing(
                            translation, self.on_translated
                        )
                except Exception as e:
                    await asyncio.sleep(0.1)
                    continue
                    print(f"Error during processing: {e}")
            await asyncio.sleep(0.1)  # Add a short sleep to prevent high CPU usage

    def process(self, audio_data):
        print("On Process...")
        self.queue.put_nowait(audio_data)

    def process_result_of_queue_processing(self, translated_text, action):
        if translated_text and translated_text.strip() != "":
            self.new_translation_signal.emit(translated_text)
            if action:
                action(translated_text)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.queue = None
        self.quit()  # Ensure the thread event loop is stopped


# ----- Audio Processors  -----
class MicrophoneProcessor(AbstractProcessor):
    """
    Processes audio from the microphone, transcribes it,
    translates it, and sends the translation to the TTS engine.
    """

    def __init__(self, src_lang, dst_lang, tts_engine, rate=16000):
        super().__init__(src_lang, dst_lang, rate, self.generate_tts_by_translation)
        self.tts_queue = queue.Queue()
        self.tts_player_worker = PlayerWorker(self.tts_queue)
        self.tts_worker = TTSWorker(
            self.tts_player_worker.queue, tts_engine=tts_engine, lang=dst_lang
        )

    def run(self):
        self.tts_worker.start()
        self.tts_player_worker.start()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.async_run()
        )  # Call async_run from AbstractProcessor
        loop.close()
        self.tts_worker.stop()
        self.tts_player_worker.stop()

    def generate_tts_by_translation(self, translation):
        self.tts_worker.queue.put_nowait(translation)


class SpeakerProcessor(AbstractProcessor):
    """
    Processes audio from the speaker and translates it.
    """

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            self.async_run()
        )  # Call async_run from AbstractProcessor
        loop.close()


# ----- TTS Worker and Player -----

class TTSWorker(QThread):
    """
    Worker thread for generating speech from text using the chosen TTS engine.
    """

    def __init__(self, play_queue, tts_engine, lang):
        super().__init__()
        self.tts_engine = tts_engine
        self.queue = queue.Queue()
        self.play_queue = play_queue
        self.running = True
        self.tts_lang = lang

    def run(self):
        while self.running:
            if not self.queue.empty():
                try:
                    text = self.queue.get_nowait()
                    self.play_queue.put_nowait(
                        self.tts_engine.text_to_speech(text, self.tts_lang)
                    )
                except Exception as e:
                    time.sleep(0.05)
                    continue
                    print(f"Error during TTS generation: {e}")
                # finally:
                #     if self.queue and text:
                #         self.queue.task_done()
            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.queue = None
        self.quit()  # Ensure the thread event loop is stopped


class PlayerWorker(QThread):
    """
    Worker thread for playing generated speech.
    """

    def __init__(self, tts_queue):
        super().__init__()
        self.queue = tts_queue
        self.running = True
        self.speaker = sc.default_speaker()

    def set_speaker(self, speaker):
        self.speaker = speaker

    def run(self):
        while self.running:
            if not self.queue.empty():
                try:
                    tts_result = self.queue.get_nowait()
                    audio_data = tts_result["audio"]
                    samplerate = tts_result["samplerate"]
                    self.speaker.play(audio_data, samplerate=samplerate)
                except Exception as e:
                    time.sleep(0.05)
                    continue
                    print(f"Error during playback: {e}")
                # finally:
                #     if self.queue and tts_result:
                #         self.queue.task_done()

            time.sleep(0.05)

    def stop(self):
        self.running = False
        self.requestInterruption()
        self.queue = None
        self.quit()  # Ensure the thread event loop is stopped


# ----- Application GUI -----

class ApplicationGUI(QMainWindow):
    """
    Main application window with UI elements.
    """

    update_translation_signal = pyqtSignal(str, str)
    update_waveform_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Communage, AI speech-to-speech Simultaneous translator!")
        self.setGeometry(100, 100, 750, 400)
        self.setWindowIcon(QIcon('resources/icon.ico'))
        self.apply_dark_theme()
        self.init_ui()
        self.microphone_thread = None
        self.speaker_thread = None
        self.tts_engine = None

    def apply_dark_theme(self):
        # apply dark theme
        self.setStyleSheet(get_gui_dark_stylesheet())

    def init_ui(self):
        # Create layout and widgets
        layout = QVBoxLayout()
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Translation labels (scrollable text boxes with word wrapping)
        self.speaker_translation_textbox = self.create_translation_textbox("#333333")
        self.microphone_translation_textbox = self.create_translation_textbox("#444444")

        # Add padding to the microphone_translation_textbox
        self.microphone_translation_textbox.setStyleSheet(
            "padding-bottom: 95px; background-color: #444444;"
        )

        # Add the speaker_translation_textbox to the layout
        layout.addWidget(self.speaker_translation_textbox)

        # Create a QStackedLayout to stack the waveform on top of the textbox
        self.stacked_layout = QStackedLayout()
        layout.addLayout(self.stacked_layout)
        self.stacked_layout.setStackingMode(QStackedLayout.StackAll)

        # Add the microphone_translation_textbox to the stacked layout
        self.stacked_layout.addWidget(self.microphone_translation_textbox)

        # Waveform animation widget
        self.waveform_label = QLabel()
        self.movie = QMovie("resources/sound-wave.gif")  # Replace with the path to your GIF
        self.waveform_label.setMovie(self.movie)
        self.movie.start()
        self.waveform_label.setScaledContents(True)  # Scale the GIF to fill the label
        self.waveform_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.waveform_label.setFixedHeight(80)
        self.waveform_label.setFixedWidth(700)
        self.waveform_label.setVisible(False)  # Initially hidden

        # Create a transparent container for the waveform animation
        waveform_container = QWidget()
        waveform_container.setStyleSheet("background-color: transparent;padding:5px;")
        waveform_container_layout = QVBoxLayout()
        waveform_container_layout.setContentsMargins(0, 0, 0, 0)  # Remove any margins
        waveform_container_layout.addWidget(
            self.waveform_label, alignment=Qt.AlignBottom
        )
        waveform_container.setLayout(waveform_container_layout)

        # Add the waveform_container to the stacked layout
        self.stacked_layout.addWidget(waveform_container)

        # Set the index to show the microphone_translation_textbox initially
        self.stacked_layout.setCurrentIndex(0)

        # Start/Stop button
        self.start_stop_button = QPushButton("Start Translation")
        self.start_stop_button.clicked.connect(self.toggle_translation)
        layout.addWidget(self.start_stop_button)

        # Language selection layout
        language_selection_layout = QHBoxLayout()

        # Speaker language selection
        speaker_language_vbox = QVBoxLayout()
        speaker_language_label = QLabel("Guest Language:")
        speaker_language_vbox.addWidget(speaker_language_label, alignment=Qt.AlignLeft)
        self.speaker_language_selection = QComboBox()
        speaker_language_vbox.addWidget(self.speaker_language_selection)
        language_selection_layout.addLayout(speaker_language_vbox)

        # Microphone language selection
        microphone_language_vbox = QVBoxLayout()
        microphone_language_label = QLabel("Your Language:")
        microphone_language_vbox.addWidget(
            microphone_language_label, alignment=Qt.AlignLeft
        )

        self.microphone_language_selection = QComboBox()
        microphone_language_vbox.addWidget(self.microphone_language_selection)
        language_selection_layout.addLayout(microphone_language_vbox)

        # Add the complete language selection layout to the main layout
        layout.addLayout(language_selection_layout)

        # TTS engine selection and Output Device
        engine_device_layout = QHBoxLayout()

        # TTS engine selection
        tts_engine_vbox = QVBoxLayout()
        tts_engine_label = QLabel("TTS Engine:")
        tts_engine_vbox.addWidget(tts_engine_label, alignment=Qt.AlignLeft)
        self.tts_engine_selection = QComboBox()
        self.tts_engine_selection.addItem("Edge TTS", EdgeTTSEngine)
        self.tts_engine_selection.setCurrentText("Edge TTS")
        tts_engine_vbox.addWidget(self.tts_engine_selection)
        engine_device_layout.addLayout(tts_engine_vbox)

        # Output Device selection
        output_device_vbox = QVBoxLayout()
        output_device_label = QLabel("Output Device:")
        output_device_vbox.addWidget(output_device_label, alignment=Qt.AlignLeft)
        self.output_device_selection = QComboBox()

        self.speakers = sc.all_speakers()
        default_speaker = sc.default_speaker()
        default_speaker_index = 0
        for index, speaker in enumerate(self.speakers):
            self.output_device_selection.addItem(speaker.name, speaker.id)
            if speaker.id == default_speaker.id:
                default_speaker_index = index
        self.output_device_selection.setCurrentIndex(default_speaker_index)

        output_device_vbox.addWidget(self.output_device_selection)
        engine_device_layout.addLayout(output_device_vbox)

        # Add the complete engine/device selection layout to the main layout
        layout.addLayout(engine_device_layout)

        # Add languages to the drop-down menus
        languages = get_languages()
        for lang in languages:
            self.speaker_language_selection.addItem(lang["name"], lang["code"])
            self.microphone_language_selection.addItem(lang["name"], lang["code"])

        self.speaker_language_selection.setCurrentText("English (United States)")
        self.microphone_language_selection.setCurrentText("Persian (Iran)")

        # Add languages to the drop-down menus
        languages = get_languages()
        for lang in languages:
            self.speaker_language_selection.addItem(lang["name"], lang["code"])
            self.microphone_language_selection.addItem(lang["name"], lang["code"])

        # Connect signals to slots
        self.update_translation_signal.connect(self.update_translation)
        self.update_waveform_signal.connect(self.change_waveform)
        self.output_device_selection.currentIndexChanged.connect(
            self.on_speaker_changed
        )

    def create_translation_textbox(self, background_color):
        textbox = QTextEdit()
        textbox.setReadOnly(True)
        textbox.setWordWrapMode(QTextOption.WordWrap)
        textbox.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        textbox.setFont(QFont("Segoe UI", 14, QFont.Bold))
        textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        textbox.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {background_color};
            }}
            """
        )
        textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return textbox

    def on_speaker_changed(self, index):
        speaker_id = self.output_device_selection.itemData(index)
        selected_speaker = sc.get_speaker(speaker_id)

        try:
            if (
                self.microphone_thread is not None
                and self.microphone_thread.isRunning()
            ):
                self.microphone_thread.processor.tts_player_worker.set_speaker(
                    selected_speaker
                )
        except Exception as e:
            print(f"Error on changing speaker: {e}")

    @pyqtSlot(str, str)
    def update_translation(self, source, translation):
        textbox = (
            self.microphone_translation_textbox
            if source == "microphone"
            else self.speaker_translation_textbox
        )
        cursor = QTextCursor(textbox.document())
        cursor.select(QTextCursor.Document)
        format = cursor.charFormat()
        format.setForeground(QColor("#ffffff"))
        format.setFontWeight(QFont.Weight.Light)  # Set font weight to 100 (Light)
        format.setFontPointSize(12)
        cursor.setCharFormat(format)
        cursor.setPosition(0)
        textbox.setTextCursor(cursor)
        highlight_color = QColor("#00eb9e")
        format.setForeground(highlight_color)
        format.setFontWeight(QFont.Weight.Bold)  # Set font weight to 100 (Light)
        format.setFontPointSize(14)
        cursor.insertText(translation + "\n\n", format)
        textbox.ensureCursorVisible()
        QApplication.processEvents()

    @pyqtSlot(bool)
    def change_waveform(self, is_speaking):
        if self.waveform_label.isVisible() != is_speaking:
            self.stacked_layout.setCurrentIndex(0 if not is_speaking else 1)
            self.waveform_label.setVisible(is_speaking)
            QApplication.processEvents()

    @pyqtSlot(str)
    def update_microphone_translation(self, text):
        self.update_translation_signal.emit("microphone", text)

    @pyqtSlot(str)
    def update_speaker_translation(self, text):
        self.update_translation_signal.emit("speaker", text)

    def toggle_translation(self):
        if self.start_stop_button.text() == "Start Translation":
            # Disable language and engine selection (Task 2)
            self.speaker_language_selection.setEnabled(False)
            self.microphone_language_selection.setEnabled(False)
            self.tts_engine_selection.setEnabled(False)

            speaker_lang = self.speaker_language_selection.currentData()
            microphone_lang = self.microphone_language_selection.currentData()

            # Get selected TTS engine class (Task 5)
            tts_engine_class = self.tts_engine_selection.currentData()
            self.tts_engine = tts_engine_class()  # Instantiate the engine

            self.microphone_thread = MicrophoneStreamHandler(
                microphone_lang, speaker_lang, self.tts_engine
            )

            self.microphone_thread.new_translation_signal.connect(
                self.update_microphone_translation
            )

            self.microphone_thread.update_waveform_signal.connect(
                self.update_waveform_signal
            )
            self.microphone_thread.start()

            # Set Output Device to user selected Device
            selected_speaker = sc.get_speaker(
                self.output_device_selection.currentData()
            )
            self.microphone_thread.processor.tts_player_worker.set_speaker(
                selected_speaker
            )

            self.speaker_thread = SpeakerStreamHandler(speaker_lang, microphone_lang)
            self.speaker_thread.new_translation_signal.connect(
                self.update_speaker_translation
            )
            self.speaker_thread.start()

            # Initialization check for microphone thread
            if not self.microphone_thread.initialized:
                self.start_stop_button.setEnabled(False)
                self.start_stop_button.setText("Initializing...")
                self.initialization_timer = QTimer(self)
                self.initialization_timer.timeout.connect(self.check_initialization)
                self.initialization_timer.start(100)  # Check every 100 milliseconds
            else:
                self.start_stop_button.setText("Stop Translation")
        else:
            self.start_stop_button.setText("Start Translation")

            # Stop translation logic...
            if self.microphone_thread:
                self.microphone_thread.stop_processing()
                self.microphone_thread = None

            if self.speaker_thread:
                self.speaker_thread.stop_processing()
                self.speaker_thread = None

            # Hide Waveform
            self.waveform_label.setVisible(False)
            self.stacked_layout.setCurrentIndex(0)
            QApplication.processEvents()

            # Enable language and engine selection (Task 2)
            self.speaker_language_selection.setEnabled(True)
            self.microphone_language_selection.setEnabled(True)
            self.tts_engine_selection.setEnabled(True)
            QApplication.processEvents()

    def check_initialization(self):
        # Check if microphone thread is initialized
        if self.microphone_thread.initialized:
            self.initialization_timer.stop()
            self.start_stop_button.setEnabled(True)
            self.start_stop_button.setText("Stop Translation")

    def closeEvent(self, event):
        if self.microphone_thread:
            self.microphone_thread.stop_processing()
            self.microphone_thread = None

        if self.speaker_thread:
            self.speaker_thread.stop_processing()
            self.speaker_thread = None

        event.accept()


# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = ApplicationGUI()
    mainWin.show()
    sys.exit(app.exec_())
