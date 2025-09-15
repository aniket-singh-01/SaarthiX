from pathlib import Path
import os
from environment.config.llm import claude
from environment.agents.base import BaseAgent
from environment.communication.message import Message


class CrossTalkAdapter(BaseAgent):
    def __init__(self):
        super().__init__()

    def process_message(self, message):
        reqs = message.content.get("reqs")
        lab_path = message.content.get("lab_path")
        dou_gen = message.content.get("dou_gen")
        peng_gen = message.content.get("peng_gen")

        dou_gen_name = os.path.basename(dou_gen)
        peng_gen_name = os.path.basename(peng_gen)

        path = Path(lab_path)
        with open(lab_path, 'r', encoding='utf-8') as f:
            en_script = f.read().strip()

        user_prompt = f"""
        You are a professional crosstalk (xiang sheng) adaptation specialist. Please adapt the following English stand-up comedy material into an authentic traditional Chinese crosstalk dialogue format.

        Material to adapt:  
        {en_script}

        Crosstalk roles:
        - {dou_gen_name}: Comic lead (dou gen), delivers main jokes and drives the narrative
        - {peng_gen_name}: Straight man (peng gen), reacts and plays off the comic lead

        Format Requirements:
        1. Each performer's lines must be on separate lines starting with their name.
        2. Begin each line with one tone marker: [Natural] or [Confused] or [Emphatic]. 
           The same tone should not appear consecutively for more than two lines.
        
        Example:
        [tone] Role name: ... 
        [tone] Role name: ...

        Additional requirements:
        {reqs}

        Guidelines:
        - The first line of the output should be the title of the crosstalk.
        - Preserve core humor while localizing cultural references
        - Incorporate traditional crosstalk speech patterns and rhythm
        - Use common crosstalk phrases and interactive elements

        Output ONLY the adapted title and dialogue without any formatting symbols or explanations.
        """

        try:
            print("Creating crosstalk script... ")
            response = claude(user=user_prompt)
            res = response.choices[0].message.content

            output_path = os.path.join(path.parent, 'ct.txt')
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(res)

            return Message(
                content={
                    "status": "success",
                    "script": res
                }
            )
        except Exception as e:
            print(f"Error in talk show writing: {str(e)}")
            return Message(
                content={
                    "status": "error",
                    "message": f"Error in cross talk writing: {str(e)}"
                }
            )