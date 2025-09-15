import json
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from typing import List, Dict, Tuple
import os
import math
import tempfile
import sys
import re


####if occur XDG_RUNTIME_DI error, use it in terminal >>>>> export XDG_RUNTIME_DIR=/run/user/$(id -u)
class VideoEditor:
    def __init__(self):
        # Get the current file's directory
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Navigate up 2 levels to reach the Vtube root directory
        self.project_root = os.path.abspath(os.path.join(self.current_dir, '..', '..', '..'))
        
        # Define paths
        self.dataset_dir = os.path.join(self.project_root, 'dataset')
        self.video_edit_dir = os.path.join(self.dataset_dir, 'video_edit')
        self.music_analysis_dir = os.path.join(self.video_edit_dir, 'music_analysis')
        self.scene_output_dir = os.path.join(self.video_edit_dir, 'scene_output')
        self.working_dir = os.path.join(self.video_edit_dir, 'videosource-workdir')
        self.voice_gen_dir = os.path.join(self.video_edit_dir, 'voice_gen')
        self.music_data_dir = os.path.join(self.video_edit_dir, 'music_data')
        self.video_output_dir = os.path.join(self.video_edit_dir, 'video_output')
        
        # Set model path to correct location in tools directory
        self.model_path = os.path.join(self.project_root, 'tools', 'MiniCPM-V-2_6-int4')
        print(f"Loading model from: {self.model_path}")
        
        # Load model and tokenizer
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Default video directory
        self.ROOT_VIDEO_DIR = os.path.join(self.video_edit_dir, 'video_source')
        
        # Define standard file paths
        self.beats_file = os.path.join(self.music_analysis_dir, "rhythm_points.json")
        self.storyboard_file = os.path.join(self.scene_output_dir, "video_scene.json")
        # Find the first audio file in music_data directory
        # Find the first file in music_data directory
        self.audio_file = None
        
        # Create necessary directories if they don't exist
        for directory in [self.working_dir, self.scene_output_dir, self.music_analysis_dir, self.music_data_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Load segment data
        segments_path = os.path.join(self.scene_output_dir, 'visual_retrieved_segments.json')
        if os.path.exists(segments_path):
            with open(segments_path, 'r', encoding='utf-8') as f:
                self.video_segments = json.load(f)
                print(f"Loaded video segments from: {segments_path}")
        else:
            print(f"Warning: {segments_path} not found. Using default empty list.")
            self.video_segments = []
            
        kv_store_path = os.path.join(self.working_dir, 'kv_store_video_segments.json')
        if os.path.exists(kv_store_path):
            with open(kv_store_path, 'r', encoding='utf-8') as f:
                self.video_segments_data = json.load(f)
        else:
            print(f"Warning: {kv_store_path} not found. Using default empty dict.")
            self.video_segments_data = {}




    def load_video_timing(self, segment_name: str) -> Tuple[float, float]:
        """Load video timing from segment name"""
        try:
            parts = segment_name.split('_')
            
            # Handle different format cases
            if len(parts) == 3:
                # Format: movie_video_id_section (e.g., "movie_0_105")
                movie = parts[0]
                video_id = parts[1]
                section = parts[2]
                video_key = f"{movie}_{video_id}"
            elif len(parts) == 2:
                # Format: movie_section (e.g., "movie1_105")
                movie = parts[0]
                section = parts[1]
                video_key = movie
            else:
                print(f"Invalid segment name format: {segment_name}")
                return None, None
            
            # Check if data exists and retrieve timing
            if video_key in self.video_segments_data and section in self.video_segments_data[video_key]:
                timing = self.video_segments_data[video_key][section]['time']
                start_time, end_time = map(float, timing.split('-'))
                return start_time, end_time
            else:
                print(f"No timing data found for {segment_name} (key: {video_key}, section: {section})")
                return None, None
        except Exception as e:
            print(f"Error processing timing for {segment_name}: {str(e)}")
            return None, None


    def get_video_path(self, segment_name: str) -> str:
        """Get video file path from segment name"""
        try:
            parts = segment_name.split('_')
            
            if len(parts) == 3:
                # Format: movie_video_id_section 
                movie_name = parts[0]
                video_num = parts[1]
                video_filename = f"{movie_name}_{video_num}.mp4"
            elif len(parts) == 2:
                # Format: movie_section 
                movie_name = parts[0]
                video_filename = f"{movie_name}.mp4"
            else:
                print(f"Invalid segment name format for video path: {segment_name}")
                return None
            
            video_path = os.path.join(self.ROOT_VIDEO_DIR, video_filename)
            if not os.path.exists(video_path):
                print(f"Video file not found at: {video_path}")
            return video_path
        except Exception as e:
            print(f"Error getting video path for {segment_name}: {str(e)}")
            return None


    def extract_frames(self, video: VideoFileClip, start_time: float, end_time: float) -> List[Tuple[float, Image.Image]]:
        """Extract frames including the exact start time"""
        frames = []
        try:
            # Start from exact start_time, then continue with whole seconds
            frames_times = [start_time]  # Include exact start time
            current_time = math.ceil(start_time)  # Round up to next second
            
            while current_time < end_time:
                frames_times.append(current_time)
                current_time += 1
                
            for t in frames_times:
                try:
                    if t >= end_time:
                        break
                    frame = video.get_frame(t)
                    # Convert to RGB if necessary
                    if frame.shape[2] == 4:  # If RGBA
                        frame = Image.fromarray(frame).convert('RGB')
                    else:
                        frame = Image.fromarray(frame)
                    
                    # Resize with consistent dimensions
                    frame = frame.resize((224, 224), Image.Resampling.LANCZOS)
                    frames.append((t, frame))
                except Exception as e:
                    print(f"Error extracting frame at time {t:.3f}: {e}")
                    continue
                    
            return frames
        except Exception as e:
            print(f"Error in frame extraction: {e}")
            return frames


    def process_video(self, beats_file: str, storyboard_file: str, audio_file: str, keep_original_audio: bool = False, audio_mix_ratio: float = 0.3, output_file: str = "output_video.mp4"):
        """Main video processing pipeline"""
        import re  
        import math
        final_clips = []
        final_video = None
        background_audio = None
        
        try:
            # Ensure file paths are absolute
            beats_file = os.path.join(self.working_dir, beats_file) if not os.path.isabs(beats_file) else beats_file
            storyboard_file = os.path.join(self.working_dir, storyboard_file) if not os.path.isabs(storyboard_file) else storyboard_file
            audio_file = os.path.join(self.voice_gen_dir, audio_file) if not os.path.isabs(audio_file) else audio_file
            output_file = os.path.join(self.scene_output_dir, output_file) if not os.path.isabs(output_file) else output_file
            
            print(f"Processing with files:")
            print(f"  Beats: {beats_file}")
            print(f"  Storyboard: {storyboard_file}")
            print(f"  Audio: {audio_file}")
            print(f"  Output: {output_file}")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Load beat timestamps for time periods
            with open(beats_file, 'r', encoding='utf-8') as f:
                beats_data = json.load(f)
            
            # Handle different data formats
            if 'beat_data' in beats_data and 'beats' in beats_data['beat_data']:
                beat_timestamps = [beat['timestamp'] for beat in beats_data['beat_data']['beats']]
            elif 'sentence_data' in beats_data and 'chunks' in beats_data['sentence_data']:
                beat_timestamps = [cut['timestamp'] for cut in beats_data['sentence_data']['chunks']]
            else:
                raise ValueError("Unrecognized timestamp data format")
            
            print(f"Found {len(beat_timestamps)} timestamps")
            print(f"First 5 timestamps: {beat_timestamps[:5]}")
            
            # Create time periods starting from 0
            time_periods = [(0, beat_timestamps[0])] if beat_timestamps else []
            time_periods.extend([(beat_timestamps[i], beat_timestamps[i+1]) 
                            for i in range(len(beat_timestamps)-1)])
            
            print(f"Created {len(time_periods)} time periods")
            
            # Load storyboard
            with open(storyboard_file, 'r', encoding='utf-8') as f:
                storyboard_data = json.load(f)

            storyboard_text = storyboard_data.get('segment_scene', '')


            storyboard_sections = [section.strip() 
                                for section in storyboard_text.split('/////') 
                                if section.strip()]
            
            print(f"Total video segments: {len(self.video_segments)}")
            print(f"Total storyboard sections: {len(storyboard_sections)}")
            
            total_duration = 0
            max_periods = min(len(time_periods), len(storyboard_sections))
            print(f"Will process {max_periods} periods (limited by storyboard sections)")

            # Process each time period
            for j in range(max_periods):
                period_start, period_end = time_periods[j]
                exact_duration = period_end - period_start
                print(f"\nProcessing time period {j+1}: {period_start:.3f}s - {period_end:.3f}s (Duration: {exact_duration:.3f}s)")
                
                if not self.video_segments:
                    print("No video segments available")
                    break
                
                segment_idx = j % len(self.video_segments)
                segment_name = self.video_segments[segment_idx]
                
                try:
                    segment_start, segment_end = self.load_video_timing(segment_name)
                    if segment_start is None or segment_end is None:
                        print(f"Skipping segment {segment_name} - invalid timing")
                        continue
                        
                    print(f"Checking segment: {segment_name}")
                    print(f"Segment range: {segment_start:.3f}s - {segment_end:.3f}s")
                    
                    # Validate segment duration
                    max_start = segment_end - exact_duration
                    if max_start < segment_start:
                        print(f"Segment too short for required duration {exact_duration:.3f}s")
                        continue
                    
                    if j >= len(storyboard_sections):
                        print(f"Not enough storyboard sections for period {j+1}")
                        break
                        
                    current_section = storyboard_sections[j]
                    description = '\n'.join(current_section.split('\n')[1:]).strip()
                    print(f"Using storyboard section {j + 1}")
                    
                    video_path = self.get_video_path(segment_name)
                    if not os.path.exists(video_path):
                        print(f"Video file not found: {video_path}")
                        continue
                    
                    # Load with audio if we're keeping original audio
                    with VideoFileClip(video_path, audio=keep_original_audio) as temp_video:
                        frames_with_times = self.extract_frames(temp_video, segment_start, segment_end)
                    
                    if not frames_with_times:
                        print("No frames extracted")
                        continue
                        
                    print(f"Extracted {len(frames_with_times)} frames for analysis")
                    
                    timestamps = [t for t, _ in frames_with_times]
                    frames = [f for _, f in frames_with_times]
                    
                    prompt = (
                        f"You are analyzing a video segment from {segment_start:.3f}s to {segment_end:.3f}s. "
                        f"You see {len(frames)} consecutive video frames, each frame representing 1 second of video. "
                        f"The required clip duration is {exact_duration:.3f} seconds.\n"
                        f"Find the best sequence matching this description:\n"
                        f"\"{description}\"\n\n"
                        f"Requirements:\n"
                        f"1. You must analyze ALL {len(frames)} frames to find the best consecutive sequence\n"
                        f"2. Choose any starting frame (0-{len(frames)-1}) that allows for a {exact_duration:.3f}s clip\n"
                        f"3. Maximum starting frame should be {len(frames) - math.ceil(exact_duration)} to fit the duration\n"
                        f"4. Return ONLY a single number - the frame number (0-{len(frames)-1}) to start from\n"
                        f"5. Pick the consecutive frames that best align the scene description\n"
                        f"6. Select consecutive frames with high-quality visuals and scene consistency\n"
                    )
                    
                    msgs = [{'role': 'user', 'content': [prompt]}]
                    for frame in frames: 
                        msgs[0]['content'].append(frame)
                    
                    response = self.model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=self.tokenizer
                    ).strip()
                    
                    try:
                        frame_number = int(re.findall(r'\d+', response)[0])
                        max_start_frame = len(frames) - math.ceil(exact_duration)
                        
                        if 0 <= frame_number <= max_start_frame:
                            clip_start = segment_start + frame_number  # Start from frame_number seconds into segment
                            clip_end = clip_start + exact_duration
                            print(f"Selected clip: Starting from frame {frame_number} of {len(frames)-1}")
                            print(f"Precise timing: {clip_start:.3f}s - {clip_end:.3f}s")
                        else:
                            print(f"Frame number {frame_number} would exceed segment bounds")
                            # Use start of segment as fallback
                            clip_start = segment_start
                            clip_end = clip_start + exact_duration
                            print(f"Using segment start: {clip_start:.3f}s - {clip_end:.3f}s")
                    except Exception as e:
                        print(f"Error processing VLM response: {str(e)}")
                        # Use start of segment as fallback
                        clip_start = segment_start
                        clip_end = clip_start + exact_duration
                        print(f"Using segment start: {clip_start:.3f}s - {clip_end:.3f}s")

                    # Create clip with precise timing - now loading with audio based on keep_original_audio
                    clip = VideoFileClip(video_path, audio=keep_original_audio).subclip(clip_start, clip_end)
                    final_clips.append(clip)
                    
                    total_duration += clip.duration
                    print(f"Added clip: Duration = {clip.duration:.3f}s")
                    print(f"Current total duration: {total_duration:.3f}s")
                        
                except Exception as e:
                    print(f"Error processing segment {segment_name}: {e}")
                    continue

            if not final_clips:
                print("No valid clips to process")
                return

            print(f"\nConcatenating {len(final_clips)} clips...")
            final_video = concatenate_videoclips(final_clips, method="compose")
            
            # Audio handling based on the keep_original_audio option
            if keep_original_audio:
                # If keeping original audio, we may mix it with the background music
                print("Loading background music...")
                background_audio = AudioFileClip(audio_file).subclip(0, final_video.duration)
                
                if audio_mix_ratio > 0:
                    print(f"Mixing original audio with background music (ratio: {audio_mix_ratio:.2f})")
                    # Adjust background volume (background is quieter)
                    background_audio = background_audio.volumex(audio_mix_ratio)
                    
                    # Combine original audio with background music
                    mixed_audio = CompositeAudioClip([
                        final_video.audio,  # Original audio
                        background_audio    # Background music at reduced volume
                    ])
                    final_video = final_video.set_audio(mixed_audio)
                else:
                    # Keep only original audio, ignore background music
                    print("Using only original audio (no background music)")
            else:
                # No original audio, just use background music
                print("Adding background music only...")
                background_audio = AudioFileClip(audio_file)
                background_audio = background_audio.subclip(0, final_video.duration)
                final_video = final_video.set_audio(background_audio)
            
            print(f"Writing final video to {output_file}...")
            final_video.write_videofile(
                output_file,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                temp_audiofile=None,
                remove_temp=True,
                verbose=False
            )
            print("Video processing completed!")
            
        except Exception as e:
            print(f"Error in video processing: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            try:
                for clip in final_clips:
                    clip.close()
                if final_video is not None:
                    final_video.close()
                if background_audio is not None:
                    background_audio.close()
            except:
                pass

def main(input_path=None, keep_original_audio=False, audio_mix_ratio=0.3, output_file="rhythm_output_video.mp4", audio_file=None):
    editor = VideoEditor()
    
    # Update the root video directory if provided
    if input_path:
        # Handle the case where input_path might be a string with parentheses
        if isinstance(input_path, str) and '(' in input_path and ')' in input_path:
            # Extract the path from parentheses if needed
            input_path = input_path.strip('()')
        
        editor.ROOT_VIDEO_DIR = input_path
        print(f"Using custom video directory: {editor.ROOT_VIDEO_DIR}")
    
    # Use custom audio file if provided
    if audio_file and os.path.exists(audio_file):
        editor.audio_file = audio_file
        print(f"Using custom audio file: {editor.audio_file}")
    
    # Set output file path in video_output directory
    if not os.path.isabs(output_file):
        output_file = os.path.join(editor.video_output_dir, output_file)
    
    # Verify files exist
    for file_path in [editor.beats_file, editor.storyboard_file, editor.audio_file]:
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
    
    editor.process_video(
        beats_file=editor.beats_file,
        storyboard_file=editor.storyboard_file,
        audio_file=editor.audio_file,
        keep_original_audio=keep_original_audio,
        audio_mix_ratio=audio_mix_ratio,
        output_file=output_file
    )



