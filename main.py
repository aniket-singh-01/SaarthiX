import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools/CosyVoice'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools/CosyVoice/third_party/Matcha-TTS'))

logging.basicConfig(level=logging.WARNING)  # 或 logging.ERROR
logging.getLogger("modelscope").setLevel(logging.WARNING)
from environment.agents.multi import MultiAgent
from environment.config.config import config

def main():
    llm_api_key = config['llm']['api_key']
    llm_base_url = config['llm']['base_url']
    multi_agent = MultiAgent(llm_api_key, llm_base_url)
    result = multi_agent.process_request()

if __name__ == "__main__":
    main()