import os
import logging
import warnings
import multiprocessing
import sys


class Pre_Loader:
    def __init__(self):
        # Setup multiprocessing
        try:
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn')
        except RuntimeError:
            # This handles the case where the start method has already been set
            pass
        
        # Configure logging and warnings
        warnings.filterwarnings("ignore")
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        # Get the current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
        
        # Add the tools directory to the path
        tools_dir = os.path.join(self.project_root, 'tools')
        sys.path.append(tools_dir)
        
        # Initialize directories
        self.initialize_directories()
        
        # Import VideoRAG after directory setup
        from videorag.videoragcontent import VideoRAG, QueryParam
        self.VideoRAG = VideoRAG
        self.QueryParam = QueryParam

    def initialize_directories(self):
        # Define the path to the dataset directory and ensure it exists
        self.dataset_dir = os.path.join(self.project_root, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Define and create video_edit directory
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        os.makedirs(self.video_edit_dir, exist_ok=True)
        
        # Define and create all subdirectories
        subdirectories = [
            'music_analysis',
            'music_data',
            'scene_output',
            'video_source',
            'videosource-workdir',
            'voice_data',
            'voice_gen',
            'writing_data',
            'video_output',
            'face_db'
        ]
        
        # Create all subdirectories
        for subdir in subdirectories:
            dir_path = os.path.join(self.video_edit_dir, subdir)
            os.makedirs(dir_path, exist_ok=True)
        
        # Store directory paths for later use
        self.video_source_dir = os.path.join(self.video_edit_dir, 'video_source')
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')

    def preloading_video(self):
        # Check if there are any videos in the source directory
        if not os.path.exists(self.video_source_dir) or not os.listdir(self.video_source_dir):
            print(f"Warning: No files found in {self.video_source_dir}")
            print("Please add your MP4 video files to this directory before running.")
            return False
        
        # Get all MP4 files from the directory
        video_paths = [os.path.join(self.video_source_dir, f) for f in os.listdir(self.video_source_dir) 
                      if f.endswith('.mp4')]
        
        if not video_paths:
            print("No MP4 video files found. Please add some videos to process.")
            return False
        
        print(f"Found {len(video_paths)} video files to process:")
        for video in video_paths:
            print(f" - {os.path.basename(video)}")

        # Initialize and process videos with VideoRAG
        videoragcontent = self.VideoRAG(working_dir=self.working_dir)
        videoragcontent.insert_video(video_path_list=video_paths)
        return True