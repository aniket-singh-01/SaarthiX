import os
import yaml
import logging
from environment.roles.vid_preloader import Pre_Loader
from environment.communication.message import Message
from environment.roles.loudness_normalizer import LoudnessNormalizer
from environment.roles.resampler import Resampler
from environment.roles.separator import Separator
from environment.roles.talk_show.talk_show_adapter import TalkShowAdapter
from environment.roles.talk_show.talk_show_subtitle import TalkShowSubtitle
from environment.roles.talk_show.talk_show_synth import TalkShowSynth
from environment.roles.talk_show.talk_show_translator import TalkShowTranslator
from environment.roles.transcriber import Transcriber
from environment.roles.vid_adapter import VideoAdapter
from environment.roles.vid_comm.vid_searcher import video_search_main
from environment.roles.vid_comm.vid_editor import main


class TalkShowAgent:
    def __init__(self, config):
        self.config = config

        # Get the project root path for resolving relative paths
        self.project_root = self._get_project_root()

        self.output = self._resolve_path(self.config["talk_show"]["output"])

        # Handle video_source_dir - might be optional in the config
        self.video_source_dir = None
        if "video_source_dir" in self.config["talk_show"]:
            self.video_source_dir = self._resolve_path(self.config["talk_show"]["video_source_dir"])

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

        self.reqs = config["talk_show"]["reqs"]
        self.audio_path = config["talk_show"]["audio_path"]
        self.target = config["talk_show"]["target"]

        self.seperator = Separator()
        self.normalizer = LoudnessNormalizer()
        self.resampler = Resampler()
        self.transcriber = Transcriber()
        self.adapter = TalkShowAdapter()
        self.synth = TalkShowSynth()
        self.translator = TalkShowTranslator()
        self.subtitle = TalkShowSubtitle()

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
        preload_result = self.preload_video()
        pre_msg = Message(content={"audio_dir": os.path.dirname(self.audio_path)})
        separator_msg = self.seperator.process_message(pre_msg)

        normalizer_result = self.normalizer.process_message(pre_msg)
        resampler_result = self.resampler.process_message(pre_msg)
        transcriber_result = self.transcriber.process_message(pre_msg)

        target_msg = Message(content={"audio_dir": self.target})
        transcriber_target_result = self.transcriber.process_message(target_msg)

        lab_path = os.path.splitext(self.audio_path)[0] + '.lab'
        adapter_msg = Message(content={"reqs": self.reqs, "lab_path": lab_path})
        adapter_result = self.adapter.process_message(adapter_msg)

        synth_msg = Message(content={"script": adapter_result.content.get('script'), "target": self.target})
        synth_result = self.synth.process_message(synth_msg)

        # json_path = "dataset/talk_show/ts.json"
        # audio_dir = "dataset/talk_show/guodegang/exp"
        # video_path = "dataset/talk_show/guodegang/final/final.mp4"
        # output_path = "dataset/talk_show/guodegang/final/final_subtitle.mp4"
        # subtitle_msg = Message(content={"video_path": video_path, "output_path": output_path, "audio_dir": audio_dir, "json_path": js})
        # subtitle_result = self.subtitle.process_message(subtitle_msg)

        translator_msg = Message(content={"target": self.target})
        translator_result = self.translator.process_message(translator_msg)
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


def gen_talk_show():
    print("Welcome to the Talk Show Generator")
    with open('environment/config/talk_show.yml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print(config)
    agent = TalkShowAgent(config)
    return agent.orchestrator()
