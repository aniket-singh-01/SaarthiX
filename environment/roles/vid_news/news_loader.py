import json
import os
import math
import time
from typing import Dict, List, Any
import re
import asyncio
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import sys
from environment.config.llm import gpt

# Add the directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import news_editor
import news_editor

class ContentAgent:
    """Agent that creates content based on user ideas."""
    
    def __init__(self, txt_path: str = None, pre_txt_path: str = None):
        self.max_tokens = 15000
        self.txt_path = txt_path
        self.pre_txt_path = pre_txt_path
        self.model = "gpt-4o-mini"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        reraise=True
    )
    async def create_content(self, user_idea):
        """Generate video content incorporating user ideas and reference materials."""

        try:
            # Initialize pipeline without API key
            pipeline = news_editor.Pipeline()
            
            # Get formatted content from news_editor
            formatted_content = await pipeline.process(user_idea, self.txt_path, self.pre_txt_path)
            general_content = formatted_content
        
            scene_translation_prompt = f"""
            Key requirements:
            - Keep the number of "/////" mark unchanged.
            - You CAN ONLY deduce by English visual-scene description.
            - Deduce visual-scene keywords in English, each sections of content deduce some scene keywords (especially proper noun, eg. iphone 16, SWE Arena Benchmark ...).
            - Keep the same number of paragraph separators and spacing.
            - Each scene sections' description don't exceed 1 sentences.
            - Don't directly translate each sentences.


            Content for to process:
            "{general_content}"

            ############################

            Example Input:

            /////\nEmily and Jackson stood together, the ocean breeze ruffling their hair, both soaking up the moment, surrounded by the vastness of the ocean, which reflected their budding love.\n\n/////\nThe leader increased Xiao Wang's business freedom by changing the company's management rules.

            Example Output:

            /////\nA couple standing together on the sunset seaside with hair blown by the wind\n\n/////\nyoung employees within office environment

            """
            
            system_message = "You are a English text-scene description expert"
            response = gpt(model=self.model, system=system_message, user=scene_translation_prompt)
            
            query_content = response.choices[0].message.content
            
            return {"query": query_content, "general": general_content}

        except Exception as e:
            return {"error": f"Error: {e}"}


def count_content_sections(file_path: str) -> int:
    """
    Count the number of sections in the content file marked by '/////'.
    
    Args:
        file_path (str): Path to the content JSON file
        
    Returns:
        int: Number of sections found
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            content = data.get("content_created", "")
            
            # Count sections by splitting on the marker
            sections = content.split("/////")
            
            # Filter out empty sections
            valid_sections = [s.strip() for s in sections if s.strip()]
            
            # Print each section with its number
            print("\nContent Sections Found:")
            for i, section in enumerate(valid_sections, 1):
                print(f"\nSection {i}:")
                print(section.strip()[:100] + "..." if len(section) > 100 else section.strip())
            
            return len(valid_sections)
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return 0
    except json.JSONDecodeError:
        print(f"Error: File {file_path} is not valid JSON")
        return 0
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0


@retry(
    retry=retry_if_exception_type((Exception)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True
)
async def content_main(config=None):
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the Vtube root directory
    vtube_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Define base directory paths
    dataset_dir = os.path.join(vtube_root, 'dataset')
    video_edit_dir = os.path.join(dataset_dir, 'video_edit')
    
    # Ensure required directories exist
    scene_output_dir = os.path.join(video_edit_dir, 'scene_output')
    writing_data_dir = os.path.join(video_edit_dir, 'writing_data')
    os.makedirs(scene_output_dir, exist_ok=True)
    os.makedirs(writing_data_dir, exist_ok=True)
    
    # Define file paths
    content_output_path = os.path.join(scene_output_dir, "video_scene.json")
    txt_path = os.path.join(writing_data_dir, "audio_transcript.txt")
    pre_txt_path = os.path.join(writing_data_dir, "news_present_style.txt")

    if config and 'idea' in config:
        user_idea = config['idea']
        print(f"\nUsing idea from config: {user_idea}")
    else:
        # Always create new content
        print("\n=== CREATING SHORT NEWS CONTENT ===")
        user_idea = input("\nPlease describe your news summarization video idea, and please indicate your word count requirement (250 words around 2 minutes): ")
        
    content_agent = ContentAgent(
        txt_path=txt_path,
        pre_txt_path=pre_txt_path
    )

    content_result = await content_agent.create_content(user_idea)
    
    content_output = {
        "user_idea": user_idea,
        "content_created": content_result.get("general", ""),
        "segment_scene": content_result.get("query", "")
    }
    
    with open(content_output_path, 'w', encoding='utf-8') as f:
        json.dump(content_output, f, indent=2, ensure_ascii=False)
    
    print("\nContent saved to", content_output_path)
    num_sections = count_content_sections(content_output_path)
    print(f"\n{num_sections} Sections have been created")

    return {
        "content_output": content_output,
        "status": "success", 
        "sections": num_sections
    }


