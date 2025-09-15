import os
import yaml
import sys
import logging
from environment.roles.vid_preloader import Pre_Loader
from environment.roles.vid_rhythm.music_filter import music_main
from environment.roles.vid_rhythm.story_editor import story_main
from environment.roles.vid_rhythm.vid_searcher import video_search_main
from environment.roles.vid_rhythm.vid_editor import main


class RhythmAgent:
    def __init__(self, config):
        # Store the config
        self.config = config
        
        # Get the project root path for resolving relative paths
        self.project_root = self._get_project_root()
        
        # Convert relative paths to absolute if needed
        self.audio = self._resolve_path(self.config["rhythm_agent"]["audio"])
        self.idea = self.config["rhythm_agent"]["idea"]
        self.output = self._resolve_path(self.config["rhythm_agent"]["output"])
        
        # Handle video_source_dir - might be optional in the config
        self.video_source_dir = None
        if "video_source_dir" in self.config["rhythm_agent"]:
            self.video_source_dir = self._resolve_path(self.config["rhythm_agent"]["video_source_dir"])
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info(f"Initialized with project root: {self.project_root}")
        self.logger.info(f"Audio file: {self.audio}")
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
        
    def process_music(self):
        # Pass the audio path to music_main function
        self.logger.info(f"Starting music processing with audio file: {self.audio}")
        
        # Check if audio file exists before proceeding
        if not os.path.exists(self.audio):
            self.logger.error(f"Audio file not found at: {self.audio}")
            return 1
        
        # Pass the absolute path to music_main
        result = music_main(self.audio)
        
        if result == 0:
            self.logger.info("Music processing completed successfully")
        else:
            self.logger.error(f"Music processing failed with code: {result}")
        return result

    def process_story(self):
        # Pass the parameters to story_main function
        self.logger.info(f"Starting story creation with idea: {self.idea}")
        result = story_main(user_idea=self.idea)
        self.logger.info("Story creation completed")
        return result

    def process_video(self):
        self.logger.info("Starting video searching")
        try:
            # Call video_search_main without parameters
            self.search_results = video_search_main()
            self.logger.info("Video searching completed successfully")
            return self.search_results
        except Exception as e:
            self.logger.error(f"Error in video searching: {str(e)}")
            raise
            
    def process_edit(self):
        self.logger.info(f"Starting video editing with output path: {self.output}")
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(self.output)
            os.makedirs(output_dir, exist_ok=True)
            
            # Call main function from vid_editer with our parameters
            editing_result = main(
                input_path=self.video_source_dir,  # Use custom video source directory if specified
                keep_original_audio=False,
                audio_mix_ratio=0.3,
                output_file=self.output,
                audio_file=self.audio  # Pass our custom audio file
            )
            
            self.logger.info(f"Video editing completed successfully. Output saved to: {self.output}")
            return editing_result
        except Exception as e:
            self.logger.error(f"Error in video editing: {str(e)}")
            raise


    def orchestrator(self):
        try:

            preload_result  = self.preload_video()

            music_result = self.process_music()
            if music_result != 0:
                self.logger.error("Music processing failed. Stopping the pipeline.")
                return None
                
            story_result = self.process_story()
            

            search_result = self.process_video()


            edit_result = self.process_edit()
            
            self.logger.info("All processing completed successfully")
            return {
                "preload_result": preload_result,
                "music_result": music_result,
                "story_result": story_result,
                "search_result": search_result,
                "edit_result": edit_result
            }
            
        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None


def gen_rhy_vid():
    print("Welcome to the Rhythm Video Generator")
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root by going up two levels
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Construct the absolute path to the config file
    config_path = os.path.join(project_root, 'environment', 'config', 'rhythm_agent.yml')
    
    print(f"Loading config from: {config_path}")
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at: {config_path}")
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    agent = RhythmAgent(config)
    return agent.orchestrator()


