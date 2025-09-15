import json
import os
import sys
from environment.config.llm import gpt

class VideoAdapter:
    def __init__(self):
        # Initialize paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(self.current_dir, '..', '..'))
        
        # Define directory paths
        self.dataset_dir = os.path.join(self.project_root, 'dataset')
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.voice_gen_dir = os.path.join(self.video_edit_dir, 'voice_gen')
        
        # Define file paths
        self.input_file_path = os.path.join(self.voice_gen_dir, 'gen_audio_timestamps.json')
        self.output_file_path = os.path.join(self.scene_output_dir, 'video_scene.json')

    def extract_content_scenes(self):
        """Extract content scenes from audio timestamps, format them, and process with LLM"""
        try:
            # Load the JSON data from the input file
            with open(self.input_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            # Extract content from each chunk
            contents = [chunk["content"] for chunk in data["sentence_data"]["chunks"]]
            
            # Format content with separators
            formatted_content = "/////\n" + "\n\n/////\n".join(contents)
            
            # Process with LLM
            translated_content = self._process_with_llm(formatted_content)
            
            # Create and save the output structure
            self._save_output({"segment_scene": translated_content})
            
            print(f"Successfully processed and saved content to {self.output_file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing content: {str(e)}")
            return False

    def _process_with_llm(self, content):
        
        """Send content to LLM for scene translation using the exact specified prompt"""
        system_prompt = """
        You are a visual scene descriptor. Follow these exact requirements:

        Key requirements:
        - Keep the number of "/////" mark unchanged.
        - You CAN ONLY deduce by English visual-scene description.
        - Deduce visual-scene description in English for each sections.
        - Keep the same number of sentence separators and spacing.
        - Each scene sections' description don't exceed 1 sentences.
        - Don't directly translate each sentences.
        - If the sections contains character name, deduce that character's appearance.
        - Whenever a character is mentioned by name within the sections, the scene description must describe the character's appearance (eg. [Robert Downey Jr.] >>> Robert Downey Jr. a white male with deep brown eyes and a signature goatee.)
        """

        user_prompt = f"""
        Content to process:
        {content}

        Example Input:
        /////\n[Emily] and [Jackson] stood together, the ocean breeze ruffling their hair, both soaking up the moment, surrounded by the vastness of the ocean, which reflected their budding love.\n\n/////\nThe leader increased Xiao Wang's business freedom by changing the company's management rules.

        Example Output:
        /////\nA Red hair girl Emily and brown hair boy Jackson standing together on the sunset seaside with hair blown by the wind\n\n/////\nwhite t-shirt young employees within office environment

        Now process this content following all the rules above:
        {content}
        """

        response = gpt(
            model="gpt-4o-mini",
            system=system_prompt,
            user=user_prompt
        )

        return response.choices[0].message.content

    def _save_output(self, data):
        """Save data to output file with proper formatting"""
        os.makedirs(os.path.dirname(self.output_file_path), exist_ok=True)
        
        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    adapter = VideoAdapter()
    adapter.extract_content_scenes()