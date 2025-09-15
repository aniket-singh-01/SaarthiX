from pathlib import Path
import os
from environment.agents.base import BaseAgent
from environment.communication.message import Message
from environment.config.llm import claude


class TalkShowAdapter(BaseAgent):
    def __init__(self):
        super().__init__()

    def process_message(self, message):
        reqs = message.content.get("reqs")
        lab_path = message.content.get("lab_path")
        path = Path(lab_path)
        with open(lab_path, 'r', encoding='utf-8') as f:
            cn_script = f.read().strip()
        user_prompt = f"""
        You are a professional English stand-up comedy adaptation specialist. 
        Adapt the following Chinese crosstalk content into an authentic English stand-up comedy format.

        Content to adapt: 
        {cn_script}

        Additional requirements:
        {reqs}

        Format specifications:
        1. Each line must begin with one of these tone markers: [Natural] [Confused] [Empathetic] [Exclamatory]
        2. Add atmosphere cues [Laughter] or [Cheers] at key moments (immediately after dialogue)
        3. Keep each line independent using this structure:
            [Tone marker]...
            [Tone marker]...[Atmosphere cue (if applicable)]

        Important notes:
        - Do NOT include titles, introductions or conclusions
        - Preserve the core humor while localizing cultural references
        - Incorporate linguistic features and rhythm of English stand-up
        - Use atmosphere cues sparingly and only at pivotal moments

        Generate a 3-5 minute performance script following these requirements, ensuring every line has tone markers and key punchlines include atmosphere cues.
        Atmosphere cues (quantity: 4-5)
        Directly output the title and script without any other explanations
        Example output:
        # title
        [Tone marker]...
        [Tone marker]...
        ...
        """

        try:
            response = claude(user=user_prompt)
            res = response.choices[0].message.content

            output_path = os.path.join(path.parent, 'ts.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(res)

            return Message(
                content={
                    "status": "success",
                    "script":  res
                }
            )
        except Exception as e:
            print(f"Error in talk show writing: {str(e)}")
            return Message(
                content={
                    "status": "error",
                    "message": f"Error in talk show writing: {str(e)}"
                }
            )