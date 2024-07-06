import httpx
import json
import requests
from google.cloud import speech

class SpeechRecognizer(object):
    def __init__(
        self,
        language="fa-IR",
        rate=8000,
        retries=3,
        api_key="AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw",
        timeout=30,
        error_messages_callback=None,
    ):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries
        self.timeout = timeout
        self.error_messages_callback = error_messages_callback
        self.client = httpx.AsyncClient()

    async def __call__(self, data):
        try:
            for i in range(self.retries):
                url = f"http://www.google.com/speech-api/v2/recognize?client=chromium&lang={self.language}&key={self.api_key}"
                headers = {"Content-Type": "audio/x-flac rate=%d" % self.rate}

                try:
                    resp = await self.client.post(
                        url, data=data, headers=headers, timeout=self.timeout
                    )
                except Exception as e:
                    continue

                for line in resp.content.decode("utf-8").split("\n"):
                    try:
                        line = json.loads(line)
                        line = line["result"][0]["alternative"][0]["transcript"]
                        return line[:1].upper() + line[1:]
                    except:
                        # no result
                        continue

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return


class SpeechRecognizer_V2(object):
    def __init__(
        self,
        language="fa-IR",
        rate=8000,
        retries=3,
        api_key="AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw",
        timeout=30,
        error_messages_callback=None,
    ):
        self.language = language
        self.rate = rate
        self.api_key = api_key
        self.retries = retries
        self.timeout = timeout
        self.error_messages_callback = error_messages_callback
        self.client = speech.SpeechClient(
            client_options={
                "api_key": api_key,
                "quota_project_id": "chromium",
                "api_audience": "https://translate.google.com",
            }
        )

    async def __call__(self, data):
        try:
            for i in range(self.retries):
                try:
                    audio = speech.RecognitionAudio(content=data)
                    config = speech.RecognitionConfig(
                        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
                        sample_rate_hertz=self.rate,
                        audio_channel_count=1,
                        profanity_filter=False,
                        language_code=self.language,
                        enable_automatic_punctuation=True,
                        metadata=speech.RecognitionMetadata(
                            recording_device_name="VoIP",
                            original_mime_type="audio/x-flac",
                            audio_topic="a conference call in the field of Information Technology (IT), Software Engineering, and Computer Science.",
                        ),
                    )
                    response = self.client.recognize(config=config, audio=audio)
                    return response.results[0].alternatives[0].transcript
                except Exception as ex:
                    print("Failed on Recognizer.")
                    print(ex)
                    print(response)
                    break

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return


class SentenceTranslator(object):
    def __init__(self, src, dst, patience=-1, timeout=30, error_messages_callback=None):
        self.src = src
        self.dst = dst
        self.patience = patience
        self.timeout = timeout
        self.error_messages_callback = error_messages_callback
        self.client = httpx.AsyncClient()

    async def __call__(self, sentence):
        try:
            translated_sentence = []
            # handle the special case: empty string.
            if not sentence:
                return None
            translated_sentence = await self.GoogleTranslate(
                sentence, src=self.src, dst=self.dst, timeout=self.timeout
            )
            fail_to_translate = translated_sentence[-1] == "\n"
            patience = self.patience

            while fail_to_translate and patience:
                translated_sentence = await self.GoogleTranslate(
                    translated_sentence,
                    src=self.src,
                    dst=self.dst,
                    timeout=self.timeout,
                ).text
                if translated_sentence[-1] == "\n":
                    if patience == -1:
                        continue
                    patience -= 1
                else:
                    fail_to_translate = False

            return translated_sentence

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return

    async def GoogleTranslate(self, text, src, dst, timeout=30):
        url = "https://translate.googleapis.com/translate_a/"
        params = "single?client=gtx&sl=" + src + "&tl=" + dst + "&dt=t&q=" + text
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://translate.google.com",
        }

        try:

            response = await self.client.get(
                url + params, headers=headers, timeout=self.timeout
            )
            if response.status_code == 200:
                response_json = response.json()[0]
                length = len(response_json)
                translation = ""
                for i in range(length):
                    translation = translation + response_json[i][0]
                return translation
            return

        except requests.exceptions.ConnectionError:
            return

        except KeyboardInterrupt:
            if self.error_messages_callback:
                self.error_messages_callback("Cancelling all tasks")
            else:
                print("Cancelling all tasks")
            return

        except Exception as e:
            if self.error_messages_callback:
                self.error_messages_callback(e)
            else:
                print(e)
            return
