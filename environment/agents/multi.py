from environment.config.llm import gpt
from environment.agents.cross_talk import gen_cross_talk
from environment.agents.mad_svc import gen_mad_svc
from environment.agents.mad_tts import gen_mad_tts
from environment.agents.talk_show import gen_talk_show
from environment.agents.rhythm_agent import gen_rhy_vid
from environment.agents.comm_agent import gen_comm_vid
from environment.agents.news_agent import gen_news_vid


class MultiAgent:
    def __init__(self, api_key, base_url):
        self.functions = {
            "gen_mad_tts": gen_mad_tts,
            "gen_mad_svc": gen_mad_svc,
            "gen_talk_show": gen_talk_show,
            "gen_cross_talk": gen_cross_talk,
            "gen_rhy_vid": gen_rhy_vid,
            "gen_comm_vid": gen_comm_vid,
            "gen_news_vid": gen_news_vid
        }

    def intent_analysis(self, user_input):
        system_content = """
        You are an AI Function Router specialized in analyzing user requirements and matching them to the most appropriate function. 

        Available Functions:

        1. GEN_MAD_SVC
           - Capabilities: Generate parody audios by changing song lyrics while maintaining the original melody and performing with vocals

        2. GEN_MAD_TTS
           -- Capabilities: Given a character video, creatively adapt/rework the dialogue spoken by the character in the video

        3. GEN_TALK_SHOW
           -- Capabilities: Generate talk show

        4. GEN_CROSS_TALK
           -- Capabilities: Generate cross talk

        5. GEN_RHY_VID
           -- Capabilities: Generate movie edits/rhythm editing video

        6. GEN_NEWS_VID
           -- Capabilities: Generate news summarization

        7. GEN_COMM_VID
           -- Capabilities: Generate novel/book-to-screen adaptation, commentary video

        CRITICAL OUTPUT REQUIREMENTS:
        - You must ONLY input one of the above exact function names
        - DO NOT output any other characters, punctuation, explanation or text
        - DO NOT include quotes, periods, spaces or any other symbols
        - Example valid output: gen_mad_tts"""
        try:
            response = gpt("gpt-4o-mini", system_content, user_input)

            func = response.choices[0].message.content.lower()
            print(func)
            return func
        except Exception as e:
            print(e)

    def execute_function(self, func):

        if func not in self.functions:
            return {
                "status": "error",
                "message": f"Unknown function: {func}"
            }

        try:
            result = self.functions[func]()
            return result
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error executing {func}: {str(e)}"
            }

    def process_request(self):
        try:
            print("\n=== Video Production System ===")
            user_input = input("Please describe the type of video you would like to produce:")

            func = self.intent_analysis(user_input)
            if not func:
                return {
                    "status": "error",
                    "message": "Failed to analyze intent"
                }

            print(f"\nExecuting {func} function...")
            print("-------------------------------------------------------")
            result = self.execute_function(func)

            return result

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing request: {str(e)}"
            }
