import re
import os
from environment.agents.base import BaseAgent
from environment.communication.message import Message
from environment.config.llm import deepseek

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import json
from pydub import AudioSegment


class CrossTalkSynth(BaseAgent):
    def __init__(self):
        super().__init__()

    def concatenate_audio_files(self, base_path, cnt):
        combined_audio = AudioSegment.silent(duration=0)

        for i in range(cnt):
            audio_file_path = f"{base_path}/exp/{i}.wav"
            try:
                audio_segment = AudioSegment.from_file(audio_file_path)
                combined_audio += audio_segment
                print(f"Successfully added {audio_file_path} to the combined audio.")
            except Exception as e:
                print(f"Error loading {audio_file_path}: {str(e)}")

        output_file_dir = "../../dataset/video_edit/voice_gen"
        output_file_path = os.path.join(output_file_dir,"gen_audio.wav")
        os.makedirs(output_file_dir, exist_ok=True)
        combined_audio.export(output_file_path, format="wav")
        print(f"Final audio saved to {os.path.abspath(output_file_path)}")

        return os.path.abspath(output_file_path)

    def process_message(self, message):
        script = message.content.get("script")
        dou_gen = message.content.get("dou_gen")
        peng_gen = message.content.get("peng_gen")

        dou_gen_name = os.path.basename(dou_gen)
        peng_gen_name = os.path.basename(peng_gen)

        current_dir = os.getcwd()
        os.chdir(os.path.join(current_dir, "tools", "CosyVoice"))
        try:
            cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
        except Exception as e:
            print('cosyvoice issue:', e)
        results = []
        text_list = []
        cnt = 0
        first_line = True
        base_path = '../../' + os.path.dirname(dou_gen)

        for line in script.split('\n'):
            if not line.strip():
                continue

            if first_line:
                cleaned_line = re.sub(r'[^\w\s]', '', line.strip(), flags=re.UNICODE)
                cleaned_line = cleaned_line.strip()
                print(f"处理后的第一行: {cleaned_line}")
                first_line = False
                title = cleaned_line
                continue

            user_prompt = f"""
            Analyze the following crosstalk dialogue line for performer role, tone, text content and audience reaction:
            {line}

            Output JSON format with STRICT rules:
            1. "role" field must be either {dou_gen_name} or {peng_gen_name}
            2. "tone" field must be "Natural", "Emphatic" or "Confused"
            3. "text" field contains the dialogue content
            4. Add "reaction" field ONLY if [Laughter] or [Cheers] exists (value must be "Laughter" or "Cheers")
            5. No extra characters before/after JSON

            Example 1:
            {{
                "role": "{dou_gen_name}",
                "tone": "Natural",
                "text": "...",
                "reaction": "Cheers"
            }}

            Example 2:
            {{
                "role": "{peng_gen_name}",
                "tone": "Emphatic",
                "text": "..."
            }}

            Strictly ensure:
            - Valid JSON syntax
            - Double quotes for strings
            - Do not add any characters before or after the JSON structure

            Output ONLY the JSON object!
            """

            try:
                # 调用 OpenAI API
                response = deepseek(user=user_prompt)
                res = response.choices[0].message.content

                if res.startswith("```json"):
                    res = res[len("```json"):]
                elif res.startswith("```"):
                    res = res[len("```"):]
                if res.endswith("```"):
                    res = res[:-3]
                res = res.strip()

                print(cnt, ":", res)
                result = json.loads(res)
                role = result['role']
                tone = result['tone'].strip().lower()
                text = result['text'].strip()
                text_list.append(text)

                with open(f'{base_path}/{role}/{tone}.lab', 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()

                os.makedirs(f"{base_path}/exp", exist_ok=True)

                prompt_speech_16k = load_wav(f'{base_path}/{role}/{tone}.wav', 16000)
                for i, j in enumerate(cosyvoice.inference_zero_shot(
                        text,
                        prompt_text, prompt_speech_16k, stream=False)):
                    torchaudio.save(f'{base_path}/exp/{cnt}.wav', j['tts_speech'], cosyvoice.sample_rate)

                if 'reaction' in result:
                    reaction = result['reaction']
                    reaction_path = os.path.join(base_path, "reaction", f"{reaction}.wav")

                    try:
                        original_audio = AudioSegment.from_file(f"{base_path}/exp/{cnt}.wav")
                        reaction_audio = AudioSegment.from_file(reaction_path)

                        combined_audio = original_audio + reaction_audio

                        combined_audio.export(f"{base_path}/exp/{cnt}.wav", format="wav")
                        print(f"Successfully combined reaction audio for line {cnt}.")
                    except Exception as e:
                        print(f"Error combining reaction audio for line {cnt}: {str(e)}")

                results.append(result)
                cnt += 1
            except Exception as e:
                print(f"Error processing line: {line}. Error: {str(e)}")
                continue

        output_file_path = self.concatenate_audio_files(base_path, cnt)
        print(f"Final combined audio saved at: {output_file_path}")
        os.chdir(current_dir)
        with open(os.path.join(os.path.dirname(dou_gen), 'ct.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return Message(
            content={
                "status": "success",
                "message": results,
                "output_file": output_file_path
            }
        )