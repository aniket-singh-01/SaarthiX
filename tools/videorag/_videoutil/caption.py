import os
import glob
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

def encode_video(video, frame_times):
    frames = []
    for t in frame_times:
        frames.append(video.get_frame(t))
    frames = np.stack(frames, axis=0)
    frames = [Image.fromarray(v.astype('uint8')).resize((1280, 720)) for v in frames]
    return frames

def load_character_references(face_db_path):
    """Load character reference images and names from the face database"""
    character_references = []
    
    # Get all character folders
    character_folders = glob.glob(os.path.join(face_db_path, "*"))
    
    for folder in character_folders:
        character_name = os.path.basename(folder)
        # Get first image from the character folder
        image_files = glob.glob(os.path.join(folder, "*.[jp][pn][g]"))
        
        if image_files:
            # Load the first image as a reference
            ref_image = Image.open(image_files[0]).convert("RGB")
            # Add character reference with name
            character_references.append({
                "name": character_name,
                "image": ref_image
            })
    
    return character_references
    
def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        current_dir = os.getcwd()
        model_path = os.path.join(current_dir, 'tools/MiniCPM-V-2_6-int4')
        
        if not os.path.exists(model_path):
            print(f"Warning: Local model not found at {model_path}, falling back to Hugging Face")
            model_id = "openbmb/MiniCPM-V-2_6"  
        else:
            model_id = model_path
            print(f"Using local MiniCPM model from: {model_path}")
        
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        
        # Load character references from face_db
        face_db_path = os.path.join(current_dir, 'dataset/video_edit/face_db')
        character_references = load_character_references(face_db_path)
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                
                # Create a message with character references first, then video frames
                content = []
                
                # Add character reference images with names
                for char_ref in character_references:
                    content.append(char_ref["image"])
                    content.append(f"This target character name is {char_ref['name']}, in the following video scenes you may use this character's name if it appears.")
                
                
                # Add query text
                query = (
                    f"""
                    - Above are some character's pictures with their names, following are video may contains the target characters
                    - Provide a video scene description of the following video (including characters' emotion, motion dynamics). Based on the character images provided, check if the given character appears in the following video (if so, please describe the scene using the character's name). If not, please do not mention the target character's name."
                    - Don't response with anything unrelated, you can only reponse in ENGLISH
                    - Example Output (A conherent scene description with/without target characters): eg. A brass telescope sat forgotten on the windowsill, and (Emily/a young girl) used lens to capture the last golden rays of the setting sun.
                    - Following is the video, focus on the storytelling of video coherent description: """
                )
                
                content.append(query)
                # Add video frames
                content.extend(video_frames)
                
                msgs = [{'role': 'user', 'content': content}]
                params = {
                    "use_image_id": False,
                    "max_slice_nums": 2 
                }
                
                segment_caption = model.chat(
                    image=True,
                    msgs=msgs,
                    tokenizer=tokenizer,
                    **params
                )
                
                caption_result[index] = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache()
                
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError

def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}" 
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments