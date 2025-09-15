import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import json
import librosa.display
import os
import glob

class MusicAgent:
    def __init__(self, frame_length=2048, hop_length=512):
        """
        Initialize the MusicAgent with default parameters.
        
        Args:
            frame_length (int): Frame length for RMS calculation
            hop_length (int): Hop length for RMS calculation
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.audio_data = None
        self.sr = None
        self.last_analysis = None
    
    def load_audio(self, audio_file):
        """
        Load an audio file for analysis.
        
        Args:
            audio_file (str): Path to the audio file
            
        Returns:
            bool: True if loading was successful
        """
        try:
            self.audio_data, self.sr = librosa.load(audio_file, sr=None)
            self.audio_file_path = audio_file
            self.base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            return True
        except Exception as e:
            print(f"Error loading audio file: {e}")
            return False
    
    def detect_rhythm_points(self, 
                            energy_threshold=0.2, 
                            min_interval=0.2, 
                            smoothing_window=5,
                            mask_ranges=None):
        """
        Detect rhythm points in the loaded audio.
        
        Args:
            energy_threshold (float): Threshold for peak detection
            min_interval (float): Minimum time between detected points (seconds)
            smoothing_window (int): Window size for smoothing the RMS curve
            mask_ranges (list): List of (start, end) tuples for masking detection
            
        Returns:
            dict: Rhythm detection results
        """
        if self.audio_data is None:
            print("No audio loaded. Please load an audio file first.")
            return None
        
        # Calculate RMS energy
        rms = librosa.feature.rms(y=self.audio_data, 
                                 frame_length=self.frame_length, 
                                 hop_length=self.hop_length)[0]
        
        # Normalize RMS
        rms_normalized = rms / np.max(rms)
        
        # Apply smoothing
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            rms_normalized = np.convolve(rms_normalized, kernel, mode='same')
        
        # Find peaks
        min_samples_interval = int(min_interval * self.sr / self.hop_length)
        peaks, _ = find_peaks(rms_normalized, height=energy_threshold, distance=min_samples_interval)
        
        # Convert peaks to timestamps
        timestamps = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
        
        # Apply masking if provided
        if mask_ranges is not None and len(mask_ranges) > 0:
            # Filter out timestamps that fall within masked ranges
            filtered_timestamps = []
            masked_timestamps = []
            
            for ts in timestamps:
                is_masked = False
                for start_time, end_time in mask_ranges:
                    if start_time <= ts <= end_time:
                        is_masked = True
                        masked_timestamps.append(ts)
                        break
                
                if not is_masked:
                    filtered_timestamps.append(ts)
            
            print(f"Masked out {len(masked_timestamps)} rhythm points.")
            timestamps = np.array(filtered_timestamps)
        
        # Create results dictionary
        rhythm_points = []
        for i, timestamp in enumerate(timestamps):
            rhythm_points.append({
                "id": i + 1,
                "timestamp": round(timestamp, 3)
            })
        
        result = {
            "beat_data": {
                "count": len(rhythm_points),
                "beats": rhythm_points
            }
        }
        
        # If masking was applied, add it to the result
        if mask_ranges is not None and len(mask_ranges) > 0:
            result["mask_ranges"] = [{"start": start, "end": end} for start, end in mask_ranges]
        
        # Calculate times for plotting
        times = librosa.frames_to_time(np.arange(len(rms_normalized)), 
                                       sr=self.sr, 
                                       hop_length=self.hop_length)
        
        # Store analysis for later use
        self.last_analysis = {
            "rms_normalized": rms_normalized,
            "times": times,
            "timestamps": timestamps,
            "energy_threshold": energy_threshold,
            "mask_ranges": mask_ranges
        }
        
        return result
    
    def plot_rhythm_detection(self, figsize=(15, 12), show_plot=True, save_path=None, dpi=300):
        """
        Plot the rhythm detection results.
        
        Args:
            figsize (tuple): Figure size
            show_plot (bool): Whether to display the plot
            save_path (str): Path to save the plot
            dpi (int): DPI for the saved plot
            
        Returns:
            bool: True if plotting was successful
        """
        if self.last_analysis is None:
            print("No analysis available. Please run detect_rhythm_points first.")
            return False
        
        rms_normalized = self.last_analysis["rms_normalized"]
        times = self.last_analysis["times"]
        timestamps = self.last_analysis["timestamps"]
        energy_threshold = self.last_analysis["energy_threshold"]
        mask_ranges = self.last_analysis["mask_ranges"]
        
        plt.figure(figsize=figsize)
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(self.audio_data, sr=self.sr, alpha=0.6)
        plt.vlines(timestamps, -1, 1, color='r', linestyle='--', label='Rhythm Points')
        
        # Highlight masked regions if provided
        if mask_ranges is not None:
            for start_time, end_time in mask_ranges:
                plt.axvspan(start_time, end_time, color='gray', alpha=0.3)
        
        plt.title('Waveform with Detected Rhythm Points')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Plot RMS energy
        plt.subplot(3, 1, 2)
        plt.plot(times, rms_normalized, label='RMS Energy')
        plt.vlines(timestamps, 0, 1, color='r', linestyle='--', label='Rhythm Points')
        plt.axhline(y=energy_threshold, color='g', linestyle='-', label=f'Threshold ({energy_threshold})')
        
        # Highlight masked regions if provided
        if mask_ranges is not None:
            for start_time, end_time in mask_ranges:
                label = 'Masked Region' if start_time == mask_ranges[0][0] else ''
                plt.axvspan(start_time, end_time, color='gray', alpha=0.3, label=label)
        
        plt.title('RMS Energy')
        plt.ylabel('Normalized Energy')
        plt.legend()
        
        # Plot spectrogram
        plt.subplot(3, 1, 3)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        librosa.display.specshow(D, sr=self.sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.vlines(timestamps, 0, self.sr/2, color='r', linestyle='--', alpha=0.7)
        
        # Highlight masked regions if provided
        if mask_ranges is not None:
            for start_time, end_time in mask_ranges:
                plt.axvspan(start_time, end_time, color='gray', alpha=0.3)
        
        plt.title('Spectrogram with Rhythm Points')
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return True
    
    def analyze_rhythm_distribution(self, show_plot=True, save_path=None, dpi=300):
        """
        Analyze the distribution of rhythm intervals.
        
        Args:
            show_plot (bool): Whether to display the plot
            save_path (str): Path to save the plot
            dpi (int): DPI for the saved plot
            
        Returns:
            dict: Statistics about the rhythm distribution
        """
        if self.last_analysis is None:
            print("No analysis available. Please run detect_rhythm_points first.")
            return None
        
        timestamps = self.last_analysis["timestamps"]
        
        if len(timestamps) < 2:
            print("Not enough rhythm points to analyze intervals.")
            return None
        
        intervals = np.diff(timestamps)
        
        plt.figure(figsize=(12, 6))
        
        # Histogram of intervals
        plt.subplot(1, 2, 1)
        plt.hist(intervals, bins=20, alpha=0.7)
        plt.axvline(np.mean(intervals), color='r', linestyle='--', 
                    label=f'Mean: {np.mean(intervals):.3f}s')
        plt.axvline(np.median(intervals), color='g', linestyle='-', 
                    label=f'Median: {np.median(intervals):.3f}s')
        plt.title('Histogram of Rhythm Intervals')
        plt.xlabel('Interval (s)')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot the intervals over time
        plt.subplot(1, 2, 2)
        plt.plot(timestamps[:-1], intervals, 'o-')
        plt.axhline(np.mean(intervals), color='r', linestyle='--', label=f'Mean Interval')
        plt.title('Rhythm Intervals Over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Interval (s)')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f"Rhythm distribution plot saved to {save_path}")
        
        # Show plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        # Calculate statistics
        statistics = {
            "mean": float(np.mean(intervals)),
            "median": float(np.median(intervals)),
            "min": float(np.min(intervals)),
            "max": float(np.max(intervals)),
            "std_dev": float(np.std(intervals))
        }
        
        print(f"Interval Statistics:")
        print(f"  Mean: {statistics['mean']:.3f}s")
        print(f"  Median: {statistics['median']:.3f}s")
        print(f"  Min: {statistics['min']:.3f}s")
        print(f"  Max: {statistics['max']:.3f}s")
        print(f"  Std Dev: {statistics['std_dev']:.3f}s")
        
        return statistics
    
    def save_rhythm_points(self, output_file=None):
        """
        Save rhythm points to a JSON file.
        
        Args:
            output_file (str): Path to save the JSON file. If None, uses the base filename.
            
        Returns:
            bool: True if saving was successful
        """
        if self.last_analysis is None:
            print("No analysis available. Please run detect_rhythm_points first.")
            return False
        
        if output_file is None:
            # Create directory if it doesn't exist
            os.makedirs("music_analysis", exist_ok=True)
            output_file = f"music_analysis/{self.base_filename}_rhythm_points.json"
        
        timestamps = self.last_analysis["timestamps"]
        mask_ranges = self.last_analysis["mask_ranges"]
        
        # Create results dictionary
        rhythm_points = []
        for i, timestamp in enumerate(timestamps):
            rhythm_points.append({
                "id": i + 1,
                "timestamp": round(float(timestamp), 3)
            })
        
        result = {
            "beat_data": {
                "count": len(rhythm_points),
                "beats": rhythm_points
            }
        }
        
        # If masking was applied, add it to the result
        if mask_ranges is not None and len(mask_ranges) > 0:
            result["mask_ranges"] = [{"start": start, "end": end} for start, end in mask_ranges]
        
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Rhythm points saved to {output_file}")
        
        return True
    
    def generate_parameter_study(self, output_dir=None, 
                                thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
                                intervals=[0.1, 0.2, 0.3, 0.5],
                                smoothing_windows=[1, 3, 5, 7],
                                mask_ranges=None):
        """
        Generate a parameter study to analyze the effect of different parameters.
        
        Args:
            output_dir (str): Directory to save the results
            thresholds (list): List of threshold values to test
            intervals (list): List of minimum interval values to test
            smoothing_windows (list): List of smoothing window sizes to test
            mask_ranges (list): List of (start, end) tuples for masking detection
            
        Returns:
            list: Summary of parameter study results
        """
        if self.audio_data is None:
            print("No audio loaded. Please load an audio file first.")
            return None
        
        # Create output directory if it doesn't exist
        if output_dir is None:
            output_dir = f"parameter_study_{self.base_filename}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Results summary
        results_summary = []
        
        # Study effect of threshold
        for threshold in thresholds:
            # Use default values for other parameters
            min_interval = 0.2
            smoothing_window = 5
            
            rms = librosa.feature.rms(y=self.audio_data, 
                                     frame_length=self.frame_length, 
                                     hop_length=self.hop_length)[0]
            rms_normalized = rms / np.max(rms)
            
            if smoothing_window > 1:
                kernel = np.ones(smoothing_window) / smoothing_window
                rms_normalized = np.convolve(rms_normalized, kernel, mode='same')
            
            min_samples_interval = int(min_interval * self.sr / self.hop_length)
            peaks, _ = find_peaks(rms_normalized, height=threshold, distance=min_samples_interval)
            timestamps = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
            
            # Apply masking if provided
            if mask_ranges is not None and len(mask_ranges) > 0:
                filtered_timestamps = []
                for ts in timestamps:
                    is_masked = False
                    for start_time, end_time in mask_ranges:
                        if start_time <= ts <= end_time:
                            is_masked = True
                            break
                    if not is_masked:
                        filtered_timestamps.append(ts)
                timestamps = np.array(filtered_timestamps)
            
            times = librosa.frames_to_time(np.arange(len(rms_normalized)), 
                                          sr=self.sr, 
                                          hop_length=self.hop_length)
            
            # Store analysis temporarily for plotting
            temp_analysis = {
                "rms_normalized": rms_normalized,
                "times": times,
                "timestamps": timestamps,
                "energy_threshold": threshold,
                "mask_ranges": mask_ranges
            }
            self.last_analysis = temp_analysis
            
            # Plot and save
            plot_path = os.path.join(output_dir, f"{self.base_filename}_threshold_{threshold:.2f}.png")
            self.plot_rhythm_detection(show_plot=False, save_path=plot_path)
            
            # Add to results
            results_summary.append({
                "parameter": "threshold",
                "value": threshold,
                "rhythm_points": len(timestamps)
            })
        
        # Study effect of minimum interval
        threshold = 0.2  # Use a fixed threshold
        smoothing_window = 5  # Use a fixed smoothing window
        
        rms = librosa.feature.rms(y=self.audio_data, 
                                 frame_length=self.frame_length, 
                                 hop_length=self.hop_length)[0]
        rms_normalized = rms / np.max(rms)
        
        if smoothing_window > 1:
            kernel = np.ones(smoothing_window) / smoothing_window
            rms_normalized = np.convolve(rms_normalized, kernel, mode='same')
        
        times = librosa.frames_to_time(np.arange(len(rms_normalized)), 
                                      sr=self.sr, 
                                      hop_length=self.hop_length)
        
        for min_interval in intervals:
            min_samples_interval = int(min_interval * self.sr / self.hop_length)
            peaks, _ = find_peaks(rms_normalized, height=threshold, distance=min_samples_interval)
            timestamps = librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hop_length)
            
            # Apply masking if provided
            if mask_ranges is not None and len(mask_ranges) > 0:
                filtered_timestamps = []
                for ts in timestamps:
                    is_masked = False
                    for start_time, end_time in mask_ranges:
                        if start_time <= ts <= end_time:
                            is_masked = True
                            break
                    if not is_masked:
                        filtered_timestamps.append(ts)
                timestamps = np.array(filtered_timestamps)
            
            # Store analysis temporarily for plotting
            temp_analysis = {
                "rms_normalized": rms_normalized,
                "times": times,
                "timestamps": timestamps,
                "energy_threshold": threshold,
                "mask_ranges": mask_ranges
            }
            self.last_analysis = temp_analysis
            
            # Plot and save
            plot_path = os.path.join(output_dir, f"{self.base_filename}_interval_{min_interval:.2f}.png")
            self.plot_rhythm_detection(show_plot=False, save_path=plot_path)
            
            # Add to results
            results_summary.append({
                "parameter": "min_interval",
                "value": min_interval,
                "rhythm_points": len(timestamps)
            })
        
        # Save results summary
        with open(os.path.join(output_dir, f"{self.base_filename}_parameter_study.json"), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Parameter study completed. Results saved to {output_dir}")
        return results_summary
    
    @staticmethod
    def load_mask_ranges(mask_file=None):
        """
        Load mask ranges from a JSON file.
        
        Args:
            mask_file (str): Path to the mask file
            
        Returns:
            list: List of (start, end) tuples for masking detection
        """
        if mask_file is None or not os.path.exists(mask_file):
            return None
            
        try:
            with open(mask_file, 'r') as f:
                mask_data = json.load(f)
            
            # Parse mask ranges from the format used in the JSON file
            mask_ranges = []
            if "ranges" in mask_data:
                for range_data in mask_data["ranges"]:
                    start_time = float(range_data.get("start", 0))
                    end_time = float(range_data.get("end", 0))
                    if start_time < end_time:
                        mask_ranges.append((start_time, end_time))
            
            print(f"Loaded {len(mask_ranges)} mask ranges from {mask_file}")
            return mask_ranges
        except Exception as e:
            print(f"Error loading mask file: {e}")
            return None

def music_main(config=None):
    import os
    import glob
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the parent root directory
    parent_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Define paths for music data and analysis
    video_edit_dir = os.path.join(parent_root, 'dataset', 'video_edit')
    music_data_dir = os.path.join(video_edit_dir, 'music_data')
    music_analysis_dir = os.path.join(video_edit_dir, 'music_analysis')
    
    print(f"Using music data from: {music_data_dir}")
    print(f"Saving analysis results to: {music_analysis_dir}")
    
    # Create MusicAgent instance
    agent = MusicAgent()
    
    # Check if a specific audio file is provided in the config
    if config and isinstance(config, str):
        # If config is provided as a string, use it directly as the audio path
        audio_file = os.path.join(parent_root, config)
        print(f"Using audio file from config: {config}")
    else:
        # Otherwise, look for MP3 files in the music_data directory
        mp3_files = glob.glob(os.path.join(music_data_dir, "*.mp3"))
        
        if not mp3_files:
            print(f"Error: No MP3 files found in '{music_data_dir}' directory")
            return 1
        
        # Use the first MP3 file found
        audio_file = mp3_files[0]
        print(f"Using default audio file: {os.path.basename(audio_file)}")
    
    # Ensure the audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' does not exist")
        return 1
    
    print(f"Analyzing music file: {os.path.basename(audio_file)}")
    
    if not agent.load_audio(audio_file):
        print(f"Error: Could not load audio file '{audio_file}'")
        return 1
    
    # Define mask ranges - times in seconds where you don't want to detect rhythm points
    mask_ranges = [(0, 5)]
    
    # Detect rhythm points
    rhythm_data = agent.detect_rhythm_points(
        energy_threshold=0.4,
        min_interval=3.0,
        smoothing_window=5,
        mask_ranges=mask_ranges
    )
    
    # Print results
    print(f"Detected {rhythm_data['beat_data']['count']} rhythm points.")
    
    # Plot and save results
    plot_path = os.path.join(music_analysis_dir, f"rhythm_detection.png")
    agent.plot_rhythm_detection(show_plot=True, save_path=plot_path, dpi=300)
    
    # Analyze rhythm distribution and save
    distribution_path = os.path.join(music_analysis_dir, f"rhythm_distribution.png")
    agent.analyze_rhythm_distribution(show_plot=True, save_path=distribution_path, dpi=300)
    
    # Save to JSON
    json_path = os.path.join(music_analysis_dir, f"rhythm_points.json")
    agent.save_rhythm_points(json_path)
    
    print(f"Analysis complete! Results saved to {music_analysis_dir}")
    return 0



