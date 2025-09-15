import os
import yaml
import sys
import logging
import asyncio
from environment.roles.vid_preloader import Pre_Loader
from environment.roles.vid_news.audio_writer import transcribe_main
from environment.roles.vid_news.news_loader import content_main
from environment.roles.vid_news.voice_maker import voice_main
from environment.roles.vid_news.vid_searcher import video_search_main
from environment.roles.vid_news.vid_editor import main
from environment.roles.vid_news.vid_subtitler import subtitler_main

class NewsAgent:
    def __init__(self, config):
        # Store the config
        self.config = config
        
        # Get the project root path for resolving relative paths
        self.project_root = self._get_project_root()
        
        # Convert relative paths to absolute if needed
        self.idea = self.config["news_agent"]["idea"]
        self.output = self._resolve_path(self.config["news_agent"]["output"])
        
        # Handle video_source_dir - might be optional in the config
        self.video_source_dir = None
        if "video_source_dir" in self.config["news_agent"]:
            self.video_source_dir = self._resolve_path(self.config["news_agent"]["video_source_dir"])
        
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialized with project root: {self.project_root}")
        self.logger.info(f"Commentary idea: {self.idea}")
        self.logger.info(f"Output file: {self.output}")
        if self.video_source_dir:
            self.logger.info(f"Custom video source: {self.video_source_dir}")

    
    def _get_project_root(self):
        """Get the absolute path to the project root directory."""
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the project root
        return os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    def _resolve_path(self, path):
        """Convert relative paths to absolute paths."""
        if not path:
            return None
            
        if os.path.isabs(path):
            return path
        else:
            # For paths starting with 'dataset/' or any other relative path, resolve relative to project root
            return os.path.join(self.project_root, path)

    def preload_video(self):
        """Process videos using the Pre_Loader class."""
        self.logger.info("Starting video processing")
        
        try:
            # Initialize Pre_Loader
            loader = Pre_Loader()
            
            # If custom video source directory is specified in config, update the loader
            if self.video_source_dir:
                self.logger.info(f"Using custom video source directory: {self.video_source_dir}")
                loader.video_source_dir = self.video_source_dir
            
            # Process the videos
            result = loader.preloading_video()
            
            if result:
                self.logger.info("Video processing completed successfully")
            else:
                self.logger.warning("Video processing completed with warnings")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in video processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False

    def process_transcript(self):
        """Process transcription from videos in the source directory."""
        self.logger.info("Starting transcription processing")
        
        try:
            # Create configuration for transcribe_main
            config = {}
            
            # Add the video_source_dir to the config if it's defined
            if self.video_source_dir:
                config["video_source_dir"] = self.video_source_dir
                self.logger.info(f"Using custom video source directory for transcription: {self.video_source_dir}")
            
            # Create the transcriber instance with our config
            transcriber = transcribe_main(config=config)
            
            # Run the transcription process with force_overwrite=True to ensure it always runs
            transcript_results = transcriber.run(force_overwrite=True)
            
            self.logger.info("Transcription processing completed successfully")
            return transcript_results
        except Exception as e:
            self.logger.error(f"Error in transcription processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    
    def process_content(self):
        """Process content generation."""
        self.logger.info("Starting content processing")
        
        try:
            # Create a config dictionary to pass to content_main
            content_config = {
                "idea": self.idea
            }
            
            # Run the async function in the synchronous context
            content_results = asyncio.run(content_main(config=content_config))
            
            self.logger.info("Content processing completed successfully")
            return content_results
        except Exception as e:
            self.logger.error(f"Error in content processing: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None

    def generate_voice(self):
        """Generate voice for the content."""
        self.logger.info("Starting voice generation")
        try:
            # Run voice_main (synchronous function)
            voice_results = voice_main()
            self.logger.info("Voice generation completed successfully")
            return voice_results
        except Exception as e:
            self.logger.error(f"Error in voice generation: {str(e)}")
            raise

    def search_video(self):
        """Search for videos to use in the commentary."""
        self.logger.info("Starting video searching")
        try:
            # Call video_search_main
            search_results = video_search_main()
            self.logger.info("Video searching completed successfully")
            return search_results
        except Exception as e:
            self.logger.error(f"Error in video searching: {str(e)}")
            raise
            
    def process_edit(self):
        """Edit the video with the generated content and voice."""
        self.logger.info(f"Starting video editing.")
        try:
            
            # Call main function from vid_editer with our parameters
            editing_result = main(
                input_path=self.video_source_dir,  # Use custom video source directory if specified
                keep_original_audio=False
            )
            
            self.logger.info(f"Video editing completed successfully. ")
            return editing_result
        except Exception as e:
            self.logger.error(f"Error in video editing: {str(e)}")
            raise

    def process_subtitle(self):
        self.logger.info(f"Starting video subtitle processing with output path: {self.output}")
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output)
            os.makedirs(output_dir, exist_ok=True)
            
            
            # Call subtitler_main with our parameters
            subtitle_result = subtitler_main(
                output_path=self.output
            )
            
            self.logger.info(f"Video subtitle processing completed successfully. Output saved to: {self.output}")
            return subtitle_result
        except Exception as e:
            self.logger.error(f"Error in video subtitle processing: {str(e)}")
            raise
    
    def orchestrator(self):
        """Main orchestration method."""
        try:

            preload_result  = self.preload_video()

            transcript_result = self.process_transcript()
            if not transcript_result:
                self.logger.error("Transcription processing failed but continuing...")
            

            content_result = self.process_content()
            if not content_result:
                self.logger.error("Content processing failed. Stopping the pipeline.")
                return None
            

            voice_result = self.generate_voice()
            

            search_result = self.search_video()
            

            edit_result = self.process_edit()


            subtitle_result = self.process_subtitle()
            
            self.logger.info("All processing completed successfully")
            
            return {
                "transcript_result": transcript_result,
                "content_result": content_result,
                "voice_result": voice_result,
                "search_result": search_result,
                "edit_result": edit_result,
                "process_subtitle": subtitle_result
            }
            
        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def gen_news_vid():
    """Main entry point for the news agent."""
    print("Welcome to the News Generator")
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root by going up two levels
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Construct the absolute path to the config file
    config_path = os.path.join(project_root, 'environment', 'config', 'news_agent.yml')
    
    print(f"Loading config from: {config_path}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    agent = NewsAgent(config)
    return agent.orchestrator()