from openai import OpenAI
import os
from environment.agents.base import BaseAgent
from environment.config.config import config
import soundfile as sf
import json
client = OpenAI(api_key='<KEY>')


class TalkShowTranslator(BaseAgent):
    def __init__(self):
        super().__init__()
        client.api_key = config['llm']['api_key']
        client.base_url = config['llm']['base_url']

    def process_message(self, message):
        target = message.content.get("target")
        script_path = os.path.join(os.path.dirname(target), 'ts.txt')

        with open(script_path, 'r', encoding='utf-8') as f:
            script = f.read()

        texts = []
        chunks = []
        current_time = 0.0
        first_line = True

        for text in script.split('\n'):
            if not text.strip():
                continue
            if first_line:
                first_line = False
                continue
            texts.append(text)

        for idx, text in enumerate(texts):
            wav_path = os.path.join(target, "exp", f"{idx}.wav")
            try:
                data, samplerate = sf.read(wav_path)
                duration = round(len(data) / samplerate, 3)
                end_time = current_time + duration

                chunks.append({
                    "id": idx + 1,
                    "timestamp": round(end_time, 3),
                    "content": text
                })

                current_time = end_time

            except Exception as e:
                print(f"无法读取 {wav_path}: {str(e)}")
                continue

        result = {
            "sentence_data": {
                "count": len(chunks),
                "chunks": chunks
            }
        }

        video_gen = 'dataset/video_edit/voice_gen'
        os.makedirs(video_gen, exist_ok=True)
        with open(os.path.join(video_gen, 'gen_audio_timestamps.json'), 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return 0