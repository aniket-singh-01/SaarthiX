import json
import os
import tenacity
from typing import Dict, List, Any
import math
from environment.config.llm import gpt


class VideoContentExtractionAgent:
    """Agent that extracts video segment content and creates scene-focused narrative summaries."""
    
    def __init__(self, 
                 json_file_path: str, 
                 output_file_path: str = "video_summary.json"):
        """Initialize the agent with file paths."""
        self.json_file_path = json_file_path
        self.output_file_path = output_file_path
    
    def process(self) -> Dict[str, str]:
        """Process the video segments file, extract content and add IDs."""
        # Load data
        print(f"Loading data from {self.json_file_path}")
        with open(self.json_file_path, 'r', encoding='utf-8') as file:
            segments_data = json.load(file)
        
        # Extract all content and add IDs to each Caption
        all_contents = []
        caption_id = 1
        
        for video_key, segments in segments_data.items():
            for segment_id, segment_data in sorted(segments.items(), key=lambda x: int(x[0])):
                if "content" in segment_data:
                    # Get the content
                    content = segment_data["content"]
                    
                    # Add the ID after "Caption:"
                    if content.startswith("Caption:"):
                        content = content.replace("Caption:", f"Video Segments {caption_id}:", 1)
                        caption_id += 1
                    
                    all_contents.append(content)
        
        # Join all contents with double newlines
        final_summary = "\n\n".join(all_contents)
        
        # Save to output file
        output_data = {
            "video_summary": final_summary
        }
        
        with open(self.output_file_path, 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        
        print(f"Content with caption IDs saved to {self.output_file_path}")
        
        return output_data


class StoryboardAgent:
    """Agent that creates a storyboard based on user ideas and grounded video information."""
    
    def __init__(self, audio_json_path: str, rhythm_plot_path: str):
        self.max_tokens = 16000
        self.audio_json_path = audio_json_path
        self.rhythm_plot_path = rhythm_plot_path
        self.model = "gpt-4o-mini"
        # Initialize conversation state tracking
        self.conversation = []
        self.system_message = "You are a creative beat sync video producer who is good at write scenes from ground truth video segments, strictly following the user's requirements."

    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        stop=tenacity.stop_after_attempt(5),
        before_sleep=lambda retry_state: print(f"API call failed. Retrying in {retry_state.next_action.sleep} seconds... (Attempt {retry_state.attempt_number})")
    )
    def _call_gpt_api(self, user_prompt, temperature=0.7):
        """Call API with retry logic, maintaining conversation state."""
        # Store the prompt in our internal conversation tracking
        self.conversation.append({"role": "user", "content": user_prompt})
        
        # Create initial message for this round with system prompt
        initial_message = ""
        if len(self.conversation) > 1:
            # If we have previous conversation, let the model know about it
            assistant_messages = [msg["content"] for msg in self.conversation if msg["role"] == "assistant"]
            if assistant_messages:
                initial_message = "In our previous conversation, you said: " + assistant_messages[-1] + "\n\n"
        
        # Call the API with the combined message
        combined_prompt = initial_message + user_prompt
        response = gpt(model=self.model, system=self.system_message, user=combined_prompt)
        
        # Store the response in our conversation tracker
        assistant_response = response.choices[0].message.content
        self.conversation.append({"role": "assistant", "content": assistant_response})
        
        return response

    def create_storyboard(self, user_idea, video_summary=None):
        """Generate a rhythm-aware storyboard incorporating user ideas and optional video content."""
        # Reset conversation tracking
        self.conversation = []
        
        # First, send the video summary as context if provided
        if video_summary:
            video_context_prompt = f"""
            Here is the visual description of all the video segments from the source video:

            {video_summary}

            Each video segments represents a scene from the ground truth video.
            Later, you'll need to use these video segments content to write storybards. 

            """

            print("\n=== SENDING VIDEO VISUAL CONTENT FOR ANALYSIS ===")
            video_analysis_response = self._call_gpt_api(video_context_prompt)
            print("\nWaiting for the Agent to understand the video:")
            print(video_analysis_response.choices[0].message.content)
            print("\n=== VIDEO CONTENT ANALYSIS COMPLETE ===\n")
        
        # Get sections number from audio analysis
        sections_num = 0  # Default value if not found
        try:
            if os.path.exists(self.audio_json_path):
                with open(self.audio_json_path, 'r', encoding='utf-8') as file:
                    sectionsdata = json.load(file)
                    sections_num = sectionsdata['beat_data']['count']
                    print(f"Found {sections_num} rhythm sections in audio analysis")
            else:
                print(f"Audio JSON file not found at {self.audio_json_path}, using default of {sections_num} sections")
        except Exception as e:
            print(f"Error reading audio JSON file: {e}. Using default of {sections_num} sections")

        # Check if rhythm plot exists
        rhythm_reference = ""
        if os.path.exists(self.rhythm_plot_path):
            rhythm_reference = f"""
            Background Music Visualization with Rhythm Points for Reference:
            [View the rhythm points visualization plot at {self.rhythm_plot_path}]
            - The plot shows the musical intensity and rhythm patterns over time
            - Peaks represent high-energy moments
            - Valleys indicate calmer, quieter segments
            - Use this visualization to guide scene pacing and emotional intensity
            """
        else:
            rhythm_reference = "Note: No rhythm visualization is available. Create scenes with your own rhythm pacing."

        # Create the storyboard using the same conversation
        storyboard_prompt = f"""
        Now, build rhythm-synchronized video storybaords from the ground truth video segments content you just saw, and aligns with user's requirements

        {rhythm_reference}

        Total Scenes Required: {sections_num}
        ###################################

        User's creative requests (high priority):
        "{user_idea}"
        ###################################

        Video Storyboards Guidelines:

        1. Scene Structure:
        - Begin each scene with /////
        - Number scenes from 1 to {sections_num}
        - The video segments can only write from the video segments content given to you before
        - The storyboards you are working on is suitable for beat sync video, which is usually to splice together high-energy clips from the video material.

        2. Visual Requirements:
        - Provide detailed character appearances in every scene (e.g., "Spider-Gwen in white and pink suit with a hood and ballet shoes on a train")
        - No dialogue required
        - Include rich motion and visual descriptions

        3. Rhythm Integration:
        - Match scene intensity with visualization pattern
        - Use peaks for impactful moments

        4. Content Rules:
        - Can not exceed two sentences per scene sections
        - Focus on visual and emotional elements
        - Keep descriptions clear, concise and short
        - Maintain narrative flow between scenes
        - The storybaords should based on previous grounded video segments

        #################################
        Format Output Example (Don't answer anything unrelated, eg. ''' '''):

        /////\nA brass telescope lay forgotten on the windowsill, its lens catching the last golden rays of sunset.\n\n/////\nAutumn leaves spiraled down from maple trees, creating a russet carpet across the silent garden.
        """

        try:
            print("=== CREATING STORYBOARD ===")
            response = self._call_gpt_api(storyboard_prompt, 0.7)
            return response.choices[0].message.content

        except Exception as e:
            return f"Error creating storyboard: {str(e)}"


def story_main(use_video_content=True, user_idea=None):
    """Run the complete pipeline to extract video content and create a storyboard."""

    current_dir = os.getcwd()
    video_edit_dir = os.path.join(current_dir, 'dataset/video_edit')
    
    scene_output_dir = os.path.join(video_edit_dir, 'scene_output')
    music_analysis_dir = os.path.join(video_edit_dir, 'music_analysis')
    workdir = os.path.join(video_edit_dir, 'videosource-workdir')
    
    # Updated file paths
    video_segments_path = os.path.join(workdir, "kv_store_video_segments.json")
    summary_output_path = os.path.join(scene_output_dir, "video_summary.json")
    storyboard_output_path = os.path.join(scene_output_dir, "video_scene.json")
    audio_json_path = os.path.join(music_analysis_dir, "rhythm_points.json")
    rhythm_plot_path = os.path.join(music_analysis_dir, "rhythm_detection.png")
    
    # Variables to hold our data
    summary_results = None
    video_summary = None
    
    # Check if use_video_content is a string (from config) and convert to boolean
    if isinstance(use_video_content, str):
        use_video_content = (use_video_content == "1")
    
    # Process video content if requested
    if use_video_content and os.path.exists(video_segments_path):
        print("\n=== STAGE 1: EXTRACTING AND SUMMARIZING VIDEO CONTENT ===")
        content_agent = VideoContentExtractionAgent(
            json_file_path=video_segments_path,
            output_file_path=summary_output_path
        )
        summary_results = content_agent.process()
        video_summary = summary_results.get("video_summary", "")
    else:
        if use_video_content:
            print(f"Video segments file not found at {video_segments_path}")
        print("Creating storyboard based on your idea only.")
    
    # If user_idea is not provided (should never happen with your new approach)
    if user_idea is None or user_idea == "":
        user_idea = "A creative music video with visual effects"
        print(f"Using default idea: {user_idea}")
    else:
        print(f"Using provided idea: {user_idea}")
    
    # Check if audio analysis files exist
    if not os.path.exists(audio_json_path):
        print(f"Warning: Audio analysis file not found at {audio_json_path}")
        print("The storyboard will be created without rhythm information.")
        print("Run music analysis first to include rhythm data in your storyboard.")
    
    # Create storyboard with rhythm plot reference
    storyboard_agent = StoryboardAgent(
        audio_json_path=audio_json_path,
        rhythm_plot_path=rhythm_plot_path
    )
    storyboard = storyboard_agent.create_storyboard(user_idea, video_summary)
    
    # Save storyboard output
    storyboard_output = {
        "user_idea": user_idea,
        "video_summary": video_summary,
        "segment_scene": storyboard
    }
    
    with open(storyboard_output_path, 'w', encoding='utf-8') as f:
        json.dump(storyboard_output, f, indent=2, ensure_ascii=False)
    
    print("\nStoryboard saved to", storyboard_output_path)
    print("\nStoryboard Preview:")
    print(storyboard[:1000] + "..." if len(storyboard) > 1000 else storyboard)
    
    return {
        "summary_results": summary_results,
        "storyboard_output": storyboard_output
    }
