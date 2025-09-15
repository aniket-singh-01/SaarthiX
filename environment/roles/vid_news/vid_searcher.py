import os
import logging
import warnings
import multiprocessing
import json
import importlib.util
import sys

class Video_Searcher:
    def __init__(self):
        # Configure logging and warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Set up paths
        self._setup_paths()
        
        # Import dependencies after path setup
        self._import_dependencies()
    
    def _setup_paths(self):
        """Set up necessary paths and directories"""
        # Get project root based on file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        
        # Define paths
        self.dataset_dir = os.path.join(project_root, 'dataset')
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.scene_output_path = os.path.join(self.scene_output_dir, 'video_scene.json')
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')
        
        # Add tools directory to path
        tools_dir = os.path.join(project_root, 'tools')
        if tools_dir not in sys.path:
            sys.path.append(tools_dir)
    
    def _import_dependencies(self):
        """Import dependencies that require specific path setup"""
        try:
            from videorag.videoragcontent import VideoRAG, QueryParam
            self.VideoRAG = VideoRAG
            self.QueryParam = QueryParam
        except ImportError as e:
            self.logger.error(f"Failed to import VideoRAG: {e}")
            raise
    
    def process_scene(self):
        """
        Process a scene from JSON and use VideoRAG to search for matching content
        
        Returns:
            The response from VideoRAG query
        """
        try:
            with open(self.scene_output_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                
            # Extract the content
            segment_scene = data.get("segment_scene", "")
            
            if not segment_scene:
                self.logger.warning("Empty segment_scene found in the JSON file")
                
            # Use the content as query
            query = f'''{segment_scene}'''
            
            self.logger.info(f"Using query: {query}")
            
            param = self.QueryParam(mode="videoragcontent")
            # if param.wo_reference = False, VideoRAG will add reference to video clips in the response
            param.wo_reference = True
            
            videoragcontent = self.VideoRAG(
                working_dir=self.working_dir
            )
            
            response = videoragcontent.query(query=query, param=param)
            self.logger.info("VideoRAG query completed successfully")
            
            return response
            
        except FileNotFoundError:
            self.logger.error(f"Error: JSON file not found at {self.scene_output_path}")
            raise
        except json.JSONDecodeError:
            self.logger.error("Error: Invalid JSON format in the file.")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}")
            raise
    
    def run(self):
        """
        Main entry point for Video_Searcher
        """
        self.logger.info("Starting Video_Searcher")
        
        # Initialize multiprocessing with spawn method - use get_context instead
        # Modified to handle the "context already set" error
        
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # If context already set, just use the current context
            self.logger.info("Multiprocessing context already set, using current context")
        
        # Get response from VideoRAG
        response = self.process_scene()
        print(response)
        
        # Run the vid_editer
        #from environment.roles.vid_comm.vid_editer import main as vid_editer_main
        #vid_editer_main()
        
        self.logger.info("Video_Searcher completed")
        return response

def video_search_main():
    """
    Convenience function to create and run a Video_Searcher
    """
    logging.basicConfig(level=logging.INFO)
    searcher = Video_Searcher()
    return searcher.run()

