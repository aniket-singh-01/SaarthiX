import os
import yaml
import logging
from environment.roles.vid_preloader import Pre_Loader
from environment.communication.message import Message
from environment.roles.loudness_normalizer import LoudnessNormalizer
from environment.roles.mad_svc.mad_svc_analyzer import MadSVCAnalyzer
from environment.roles.mad_svc.mad_svc_annotator import MadSVCAnnotator
from environment.roles.mad_svc.mad_svc_coverist import MadSVCCoverist
from environment.roles.mad_svc.mad_svc_mixer import MadSVCMixer
from environment.roles.mad_svc.mad_svc_single import MadSVCSingle
from environment.roles.mad_svc.mad_svc_subtitle import MadSVCSubtitle
from environment.roles.mad_svc.mad_svc_translator import MadSVCTranslator
from environment.roles.vid_adapter import VideoAdapter
from environment.roles.vid_comm.vid_searcher import video_search_main
from environment.roles.vid_comm.vid_editor import main


class MadSVCAgent:
    def __init__(self, config):
        # Store the config
        self.config = config

        # Get the project root path for resolving relative paths
        self.project_root = self._get_project_root()

        self.output = self._resolve_path(self.config["mad_svc"]["output"])

        # Handle video_source_dir - might be optional in the config
        self.video_source_dir = None
        if "video_source_dir" in self.config["mad_svc"]:
            self.video_source_dir = self._resolve_path(self.config["mad_svc"]["video_source_dir"])

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(logging_handler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(f"Initialized with project root: {self.project_root}")
        self.logger.info(f"Output file: {self.output}")
        if self.video_source_dir:
            self.logger.info(f"Custom video source: {self.video_source_dir}")

        self.midi = config["mad_svc"]["midi"]
        self.lyrics = config["mad_svc"]["lyrics"]
        self.target = config["mad_svc"]["target"]
        self.reqs = config["mad_svc"]["reqs"]
        self.bgm = config["mad_svc"]["bgm"]
        self.normalizer = LoudnessNormalizer()
        self.mixer = MadSVCMixer()
        self.annotator = MadSVCAnnotator()
        self.analyzer = MadSVCAnalyzer()
        self.spliter = MadSVCSpliter()
        self.cover = MadSVCCoverist()
        self.translator = MadSVCTranslator()
        self.subtitle = MadSVCSubtitle()

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

    def adapt_video(self):
        self.logger.info("Starting video adapting")
        try:

            adapter = VideoAdapter()
            adapt_results = adapter.extract_content_scenes()
            self.logger.info("Video adapting completed successfully")
            return adapt_results
        except Exception as e:
            self.logger.error(f"Error in video adapting: {str(e)}")
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
            output_dir = os.path.dirname(self.output)
            os.makedirs(output_dir, exist_ok=True)
            # Call main function from vid_editer with our parameters
            editing_result = main(
                input_path=self.video_source_dir,  # Use custom video source directory if specified
                keep_original_audio=False,
                audio_mix_ratio=0.3,
                output_file=self.output,
            )

            self.logger.info(f"Video editing completed successfully.")
            return editing_result
        except Exception as e:
            self.logger.error(f"Error in video editing: {str(e)}")
            raise

    def orchestrator(self):
        """Main orchestration method."""
        try:
            preload_result = self.preload_video()

            pre_msg = Message(content={"audio_dir": os.path.dirname(self.target)})
            loudness_result = self.normalizer.process_message(pre_msg)
            annotator_msg = Message(content={"midi": self.midi, "lyrics": self.lyrics})
            annotator_result = self.annotator.process_message(annotator_msg)

            analyzer_msg = Message(content={"annotator_result": annotator_result.content, "reqs": self.reqs})
            analyzer_result = self.analyzer.process_message(analyzer_msg)

            single_msg = Message(content={"annotator_result": annotator_result, "analyzer_result": analyzer_result})
            single_result = self.single.process_message(single_msg)

            name = os.path.basename(self.midi)
            pure_name = os.path.splitext(name)[0]
            trans_msg = Message(content={"name": pure_name})
            trans_result = self.translator.process_message(trans_msg)

            new_name = annotator_result.content.get('name') + '_cover'
            cover_msg = Message(content={"source": f"dataset/mad_svc/cover/{new_name}.wav", "target": self.target})
            cover_result = self.cover.process_message(cover_msg)

            mixer_message = Message(content={"bgm": self.bgm, "output_dir": cover_result.content.get('output_dir')})
            mixer_result = self.mixer.process_message(mixer_message)
            print(mixer_result)


            adapt_result = self.adapt_video()

            search_result = self.search_video()

            edit_result = self.process_edit()

            self.logger.info("All processing completed successfully")

            return {
                "status": "success",
                "adapt_result": adapt_result,
                "search_result": search_result,
                "edit_result": edit_result
            }

        except Exception as e:
            self.logger.error(f"Error in orchestration: {str(e)}")
        return None


def gen_mad_svc():
    print("Welcome to the Mad SVC Generator")

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate to the project root by going up two levels
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Construct the absolute path to the config file
    config_path = os.path.join(project_root, 'environment', 'config', 'mad_svc.yml')

    print(f"Loading config from: {config_path}")

    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found at: {config_path}")
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    agent = MadSVCAgent(config)
    return agent.orchestrator()
