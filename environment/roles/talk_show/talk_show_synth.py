import os
from environment.agents.base import BaseAgent
from environment.communication.message import Message
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import json
from pydub import AudioSegment
from environment.config.llm import deepseek

class TalkShowSynth(BaseAgent):
    def __init__(self):
        super().__init__()

    def concatenate_audio_files(self, base_path, cnt):
        combined_audio = AudioSegment.silent(duration=0)

        for i in range(cnt):
            audio_file_path = os.path.join(base_path, 'exp', f'{i}.wav')
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
        abs_output_file_path = os.path.abspath(output_file_path)
        print(f"Combined audio saved to {abs_output_file_path}")

        return abs_output_file_path

    def process_message(self, message):
        script = message.content.get("script")
        target = message.content.get('target')

        current_dir = os.getcwd()
        os.chdir(os.path.join(current_dir, "tools", "CosyVoice"))
        try:
            cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=False)
        except Exception as e:
            print('cosyvoice issue:', e)
        cnt = 0
        results = []
        first_line = True
        base_path = '../../' + target
        for line in script.split('\n'):
            if not line.strip():
                continue
            if first_line:
                first_line = False
                continue
            user_prompt = f"""
            Analyze the tone, text content, and atmosphere marker of the following stand-up comedy segment:
            {line}

            Output strictly in JSON format with these rules:
            1. "tone" field must be ONLY "Natural", "Empathetic", "Confused" or "Exclamatory"
            2. "text" field contains the segment's content
            3. Add "reaction" field ONLY if there's atmosphere marker (i.e. [Laughter] or [Cheers]) behind the sentence, value must be "Laughter" or "Cheers"
            4. You should not analyze the tone and atmosphere markers of the segment yourself, but instead strictly rely on whether these markers appear in the segment.
            5. NO extra characters or explanations before/after JSON

            Example 1:
            
            {{
                "tone": "Empathetic",
                "text": "..."
            }}

            Example 2:
            {{
                "tone": "Natural",
                "text": "...",
                "reaction": "Cheers"
            }}

            Ensure the output is strictly in JSON format!
            """

            try:
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
                results.append(result)
                tone = result['tone'].lower()
                text = result['text'].strip()



                with open(os.path.join(base_path, f"{tone}.lab"), 'r', encoding='utf-8') as f:
                    prompt_text = f.read().strip()

                os.makedirs(os.path.join(base_path, 'exp'), exist_ok=True)

                prompt_speech_16k = load_wav(os.path.join(base_path, f"{tone}.wav"), 16000)
                for i, j in enumerate(cosyvoice.inference_zero_shot(
                        text,
                        prompt_text, prompt_speech_16k, stream=False)):
                    torchaudio.save(os.path.join(base_path, 'exp', f"{cnt}.wav"), j['tts_speech'], cosyvoice.sample_rate)

                reaction_path = '../../dataset/talk_show/reaction'

                if 'reaction' in result:
                    reaction = result['reaction'].lower()
                    reaction_path = os.path.join(reaction_path, f"{reaction}.wav")


                    try:
                        original_audio = AudioSegment.from_file(os.path.join(base_path, 'exp', f"{cnt}.wav"))
                        reaction_audio = AudioSegment.from_file(reaction_path)

                        combined_audio = original_audio + reaction_audio

                        combined_audio.export(os.path.join(base_path, 'exp', f"{cnt}.wav"), format="wav")
                        print(f"Successfully combined reaction audio for line {cnt}.")
                    except Exception as e:
                        print(f"Error combining reaction audio for line {cnt}: {str(e)}")

                cnt += 1
            except Exception as e:
                print(f"Error processing line: {line}. Error: {str(e)}")
                continue

        output_file_path = self.concatenate_audio_files(base_path, cnt)
        print(f"Final combined audio saved at: {output_file_path}")
        os.chdir(current_dir)
        with open(os.path.join(os.path.dirname(target), 'ct.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        return Message(
            content={
                "status": "success",
                "output_file": output_file_path
            }
        )