# How We Made the Demos

## 1. 🎬 Agentic Video Editing
Ever dreamed of creating stunning video edits that captivate your audience? With Agentic-AIGC, you can transform your favorite video clips into breathtaking montages that tell your unique story, complete with perfectly synchronized music and transitions.

In **video production and editing**, finding visual information is crucial, as it allows for better alignment with music, audio, or text. Agentic-AIGC extracts useful visual information from videos, and here we use VideoRAG to index and caption videos of unlimited length. You can configure your settings in `videoragcontent.py`

```
class VideoRAG:
    working_dir: str = field(
        default_factory=lambda: f"./videorag_cache_{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}"
    )

    # video
    threads_for_split: int = 10
    video_segment_length: int = 30 # 30 seconds
    rough_num_frames_per_segment: int = 10 # 5 frames
```
Creating videos with Agentic-AIGC requires aligning with the user's ideas, which is a key step. First, the user inputs a query about their video idea, and Agentic-AIGC performs a more granular query decomposition of the user's idea. This results in several sub-queries, each of which can match a video clip in the material library, enabling video creation. When using the Movie Editing feature, the storyboard agent `story_editor.py` percept the available visual material, allowing for more precise generation and utilization of each sub-query.

In the final stage of video production, the duration of each shot may vary. The video editor agent `vid_editor.py` performs fine-grained visual editing based on each retrieved video segment by comparing the video content with the corresponding sub-queries. It selects the moments that best match the visual content with the lowest redundancy for use.

### 1.1 Agentic Movie Edits

🚀 **Technical Details**
- Users just need to prepare the video sources, music audio file and the idea they want.
- Automatically extract music rhythm points (optional) setting threshold & mask parameters.
- Automatically assists with storyboard query design through video content based on user-provided ideas.
- Automatically complete the editing and integration of the video.

In beat-synced video editing, cuts and transitions align with the music's rhythm. We also enhance the narrative through visual storytelling by featuring high-energy visuals during musical climaxes.
In the beat synchronization module, we load .mp3 audio files using librosa and calculate RMS (Root Mean Square) energy to identify rhythmic patterns. The system finds rhythm points by detecting peaks in the energy signal above a configurable threshold, with options to filter out points that are too close together temporally (since viewers may not prefer excessive transitions at the beginning of a video). The video transition times and spectrogram with detected rhythm points are sent to `story_editor.py`, enabling the agent to determine which transitions require high-energy frames.
You can configure your preferences in`music_filter.py`:
```
# Define mask ranges - times in seconds where you don't want to detect rhythm points
mask_ranges = [(0, 5)]

# Detect rhythm points
rhythm_data = agent.detect_rhythm_points(
    energy_threshold=0.4,
    min_interval=3.0,
    smoothing_window=5,
    mask_ranges=mask_ranges
    )
```

#### 1.1.1 *Spider-Man: Across the Spider-Verse*
<a href='https://www.bilibili.com/video/BV1C9Z6Y3ESo/  ' target='_blank'><img src='assets/spiderman_cover.png' width=60%/></a>

🌟 **Key Features:**
- Perfect sync between visuals and background music rhythm (eg. 13s, 22s, 25s)
- Expert capture of high-energy scenes (<1 min) from the full movie (>2 hours)
- Maintain visual continuity and reduce redundant clips
- Accurately align the storyboard description of the user prompt (eg. 1st, 2nd sections)

Through intelligent analysis of hours of film footage, Agentic-AIGC automatically identifies **rhythm cues**, **high-energy action scenes**, and **character highlights** to achieve precise editing.

📝 **Prompt**:
```
Begin with Gwen with blonde hair sitting at a dining table in front of a window, followed by her playing drums with pop textures and notes in the background. Include action scenes featuring Miguel O'Hara in his dark blue suit with red accents, sharp red claws and black/red eye lenses, Spider-Gwen in her white and pink suit with hood and ballet shoes, Miles Morales with curly hair and red spider logo on his chest, and The Spot in his black suit covered in white spots using portal powers. Focus on the chase scene in the blue sky with trains, and emphasize quality motion such as web-swinging, fighting, and colorful special effects throughout the sequence.
```

#### 1.1.2 *Interstella*

<a href='https://www.bilibili.com/video/BV1yQZ6YkEkw/  ' target='_blank'><img src='assets/interstella_cover_love.png' width=45%/></a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href='https://www.bilibili.com/video/BV1koZ6YuEeL/  ' target='_blank'><img src='assets/interstella_cover.png' width=45%/></a>

🌟 **Key Features:**
- For the same input video, edit different styles by adapting your prompts

We showcase two distinct edits of *Interstella* created using Agentic-AIGC. The first version focuses on the theme "love transcending space and time," while the second emphasizes humanity's courage in space exploration. Both edits demonstrate how different prompts can shape the narrative and emotional impact of the same source material.

📝 **Prompt**:
```
Version 1:
Love can transcend time and space.
```
```
Version 2:
Celebrate humanity's courage in space exploration. Include scenes featuring spaceships, wormholes, black holes, space station docking maneuvers, ocean planets, and glacial worlds. Show astronauts in their distinctive white spacesuits as they venture into the unknown, highlighting mankind's relentless drive to explore the cosmos.
```

#### 1.1.3 *Nezha*
<a href='https://www.bilibili.com/video/BV1NQZ6YCEPH/  ' target='_blank'><img src='assets/nezha_cover.png' width=60%/></a>

🌟 **Key Features:**
- Capturing scences of conflicts and battles

📝 **Prompt:**
```
Capture more scenes of conflicts and battles between Nezha and Shen Gongbao (black-robed), Dragon Prince Ao Bing (blue-robed).
```

#### 1.1.4 *Titanic*
<a href='https://www.bilibili.com/video/BV12mZ6YLEXJ/  ' target='_blank'><img src='assets/titanic_cover.png' width=60%/></a>

🌟 **Key Features:**
- Understanding of romantic scenes

📝 **Prompt:**
```
A romantic and sweet love story about Jack and Rose meeting on the Titanic. It cannot include the part where the ship is in distress, nor the night scene. In the first section, Rose, wearing a purple hat and a white shirt, walks out of a white car with a purple umbrella, looking thoughtfully.
```

### 1.2 Agentic Novel-to-Screen Edits
Want to bring your favorite novels to life? Agentic-AIGC transforms written narratives into compelling video adaptations, complete with AI-generated scenes, characters, and dialogues - all without the need for actual filming or actors. Experience your beloved stories in a whole new medium.

🚀 **Technical Details**

- Users just need to provide their idea, novel/book txt file and film/tv series video source files they want as visual materials.
- (Optional) Provide favorite commentary audio files for cloning.
- (Optional) Customized presentation style txt file.
- Automatically write copy based on the provided novel/book text content.
- Automatically complete video material splicing and audio integration.

In this stage, Agentic-AIGC will input a presentation style file for the commentary video, eg. `present_style.txt`, where you can customize your desired speaking style. Next, the commentary script generated by Agentic-AIGC will be split into sentence level. Each sentence will have a short voiceover generated by CosyVoice. Here, you can provide any .wav audio file (e.g., `ava_16k.wav`) that you'd like to clone the voice from (preferably within 10 seconds), and fill in the text content of the audio in `voice_maker.py` to improve accuracy.

```
# Process this single sentence
for audio_data in self.cosyvoice.inference_zero_shot(
        single_sentence_generator(),
        "hello everyone, I'm your assistant OpenAI Chat GPT.",
        self.prompt_speech_16k,
        stream=False):

    # Store this chunk's waveform
    chunk_waveform = audio_data['tts_speech']

```

🌟 **Key Features:**
- Transforming novel narratives into visual storytelling by adapting descriptive text into cinematic scenes
- Automated scene matching that pairs textual descriptions with appropriate visual elements, ensuring narrative coherence
- Audio generation

<a href="https://www.bilibili.com/video/BV1TmZ6YjEvV/  " target='_blank'><img src="assets/joylife_cover.png" width="60%"></a>

We used Agentic-AIGC to generate a video adaptation of the opening chapters from *Joy of Life*. Our agents analyzed the novel's text and automatically created a compelling video sequence by intelligently selecting and arranging relevant scenes from the TV series.

📝 **Prompt:**
```
Write fluent commentary script with 1500 words.
```

### 1.3 Agentic News Summary Edits
Want to create engaging tech news videos? Agentic-AIGC helps transform complex technical updates into visually appealing content with dynamic graphics and clear explanations that keep viewers informed and engaged.

🚀 **Technical Details**

- Users only need to provide their idea and the interview/news source files they want to summarize.
- (Optional) Provide favorite audio files for cloning.
- (Optional) Customize presentation style txt file.
- Automatically transcribe the interview content through voice and extract key information of people/events to write news summaries.
- Automatically complete video material splicing and audio integration.

#### 1.3.1 Tech News: OpenAI's GPT-4o Image Generation Release
<table>
<tr>
<td align="center" width="50%">
<a href="https://www.bilibili.com/video/BV12mZ6YLEqW/  " target='_blank'><img src="assets/openai_news_cover.png" width="100%"></a>
Tech News made by Agentic-AIGC
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=2f3K43FHRKo" target='_blank'><img src="assets/tech_news_original_cover.png" width="100%"></a>
Original Tech Report
</td>
</tr>
</table>

🌟 **Key Features:**
- Automated news content summarization
- Accurate audio and video clip alignment (eg. 1:00 Generate a comic about relativity, 1:09 Generate a trade card image including the dog Sanji)
- Audio generation

📝 **Prompt:**
```
Short tech news, colloquial expression within 250 words, check the accuracy of key terms, e.g. the GPT model name should be 4o instead of 4.0
```

#### 1.3.2 Dune 2 Movie Cast Update Interview
<table>
<tr>
<td align="center" width="50%">
<a href="  https://www.bilibili.com/video/BV1m1Z6Y2Erb/  " target='_blank'><img src="assets/dune_news_cover.png" width="100%"></a>
Podcast Summarization About <i>Dune</i> 2 Cast
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=AVQRnDFZ1Qs" target='_blank'><img src="assets/dune_original_cover.png" width="100%"></a>
Original Podcast with <i>Dune</i> 2 Cast
</td>
</tr>
</table>

🌟 **Key Features:**
- Automated news content summarization
- Accurate audio and main characters video clip alignment (eg. 00:13 Timothée Chalamet, 00:28 Zendaya, 00:38 Romance, 00:47 Florence Pugh, 00:56 Austin Butler, 01:06 Javier Bardem)
- Audio generation

📝 **Prompt:**
```
Short movie podcast, colloquial expression within 300 words, notice to identify which actor or host is talking, don't mention movie tickets available issue.
```

## 2. Agentic Video Remaking
Want to create engaging and hilarious meme videos? Agentic-AIGC helps you craft memorable content by intelligently combining video clips, text, and effects into shareable content that could go viral.

### 2.1 🎨 Agentic Meme Video

🚀 **Technical Details**
- Users just need to provide the video path and your requirements.
- Automatically preprocesses audio (voice separation, loudness normalization, resampling, transcription) with corresponding agents
- Automatically segments the audio and performs segment-level copywriting adaptation via the Writer Agent
- Uses the Infer Agent for zero-shot inference on audio segments
- Aligns and merges audio-visual content automatically with the Combiner Agent

🌟 **Key Features:**
- Intelligent understanding and transformation of meme concepts
- Precise audio synthesis and precise scene matching

For the production of Meme Videos, we first extract the audio portion from the video (`mad_tts.py`), then segment it according to specific rules (`mad_tts_slicer.py`) and transcribe it (`transcriber.py`). Next, we use LLM to rewrite the content of the audio segments based on user requirements (`mad_tts_writer.py`), and employ the open-source audio inference tool Fish-Speech to generate audio according to the rewritten script (`mad_tts_infer.py`). Finally, we dynamically adjust the duration of the corresponding segments in the original video based on the generated audio segment lengths and merge the adjusted video segments (`mad_tts_combiner.py`).

#### 2.1.1 Master Ma as AI Researcher
<table>
<tr>
<td align="center" width="50%">
<a href='  https://www.bilibili.com/video/BV1ucZ6YmEBU/  ' target='_blank'><img src='assets/masterma_cover.png' width=100%/></a>
Master Ma as AI Researcher
</td>
<td align="center" width="50%">
<a href='https://www.bilibili.com/video/BV1584y1N7cR/  ' target='_blank'><img src='assets/masterma_original_cover.png' width=100%/></a>
Original Video of Master Ma
</td>
</tr>
</table>

📝 **Prompt:**
```
Create a humorous narrative about two PhD students seeking advice from Master Ma. For the two PhD students, one of them is known for high citation counts and the other for numerous publications. Transform martial arts terms into AI research terminology while keeping phrase lengths similar (length difference should be less than two Chinese characters). The story highlights their academic rivalry and ends with Master Ma advising against "窝里斗" (internal competition). Keep signature phrases like "大意了没有闪" (wasn't cautious enough) and "四两拨千斤" (achieving great results with minimal effort) while avoiding mentions of real institutions. The word combinations should be logical and appropriate for an academic context.
```

#### 2.1.2 Xiao-Ming-Jian-Mo(小明剑魔) Meme
<table>
<tr>
<td align="center" width="50%">
<a href='https://www.bilibili.com/video/BV1gFZ6YEE5W  ' target='_blank'><img src='assets/xiaomingjianmo1_cover.png' width=100%/></a>
Video 1: Mixue's Response
</td>
<td align="center" width="50%">
<a href='https://www.bilibili.com/video/BV1ucZ6YmE5x  '><img src='assets/xiaomingjianmo_findyourproblem_meme.png' width=100%/></a>
Video 2: Find Your Own Problems
</td>
</tr>
<tr>
<td align="center" width="50%">
<a href='https://www.bilibili.com/video/BV1ucZ6YmEFQ  ' target='_blank'><img src='assets/xiaomingjianmo_mvp_cover.png' width=100%/></a>
Video 3: MVP
</td>
<td align="center" width="50%">
<a href='https://www.bilibili.com/video/BV1ZYQzY5E1x  ' target='_blank'><img src='assets/xiaomingjianmo_original_cover.png' width=100%/></a>
Video 4: Original 小明剑魔 Video
</td>
</tr>
</table>

The 小明剑魔 meme has gained massive popularity recently through his insightful yet comedic streaming commentary. Many content creators have successfully adapted his distinctive speech pattern into creative videos. We've used Agentic-AIGC to generate three videos of this viral meme format, each capturing the unique style and energy of the original while adding new creative elements.

📝 **Prompts**:
```
Video 1:
Background: Mixue Ice Cream is a national chain brand focusing on ice cream and tea beverages. On March 15th (Consumer Rights Day), they were reported to be using overnight lemons. However, compared to other exposures, using overnight lemons isn't considered a particularly serious violation and is somewhat understandable.

- Speaker: Snow King (Mixue's representative)
- Purpose: Emphasize that the **overnight lemon** situation isn't too serious, highlighting Mixue's good reputation
- Must preserve the phrases "Look in my eyes tell me why why baby why", "回答我"
- Must end with the word "说话"
- If the original text contains awkward phrasing, such as redundant words or confused semantics, don't imitate that style or sentence structure
- Ensure natural and fluent sentences
```
```
Video 2:
Based on the following scenario, create an angry rebuttal from Zhuge Liang (me):
- Speaker: Zhuge Liang (me)
- Start with "**北伐失败怎么不找找自己问题**" (Why don't you look at your own problems for the failure of the Northern Expedition), followed by "...找自己问题" pattern sentences that **all** reference anime events
- Anime examples must mention specific characters
- Only the **last** "...找自己问题" should return to the Northern Expedition scenario
- Use colloquial language and diverse anime references
```
```
Video 3:
Based on the following scenario, create an angry rebuttal from Zhuge Liang (me):
- Speaker: Zhuge Liang (me)
- Zhuge Liang (me) is challenged about why a certain Three Kingdoms character has a higher rating than him and launches a fierce rebuttal
- Must include: "三点零、十三点零、躺赢狗"
- Do not start with "零杠几"
- Later rating comparisons should show stark differences (can be exaggerated)
- Use colloquial language, align with historical facts, and only replace specific content
```

### 2.2 Agentic Music Videos
Ready to create music videos realizing your creative ideas? Agentic-AIGC helps you write lyrics, select singers you specify, and generate matching visuals to bring your musical vision to life. The system can coordinate lyrics, visuals, and music to create engaging amateur music videos.

🚀 **Technical Details**
- Users just need to provide the music MIDI file, original lyrics, BGM file (optional), target voice file, and requirements.
- Automatically performs loudness normalization and annotates the MIDI file using the Annotator Agent.
- Automatically calibrates and adapts lyrics at the word level via the Analyzer Agent.
- Automatically divides long rest intervals to reduce melodic errors and enables song covers.

🌟 **Key Features:**
- Automated lyric generation based on themes
- Intelligent matching of visuals and lyrics

For music video production, we first extract melodic information from MIDI files (`analyzer.py` and `annotator.py`). To achieve high-quality rhythmic impact, we use the open-source music generation tool DiffSinger to generate note-by-note audio (`mad_svc_spliter.py`). Accounting for timing discrepancies in the tool's output, we dynamically adjust segment durations to match the original music (`mad_svc_single.py`). Using user-provided vocal samples, we then perform voice cloning via Seed-VC (`mad_svc_coverist.py`). Finally, we reformat the musical data into the Movie Editing input specification (`mad_svc_translator.py`) to enable seamless integration with downstream agents for final video synthesis.

<a href='https://www.bilibili.com/video/BV1t8ZCYsEeA/  ' target='_blank'><img src='assets/airencuoguo_cover.png' width=60%/></a>

📝 **Prompts**:
```
The song is performed by Patrick Star, focusing on the theme of ​**"the struggles of manuscript submission and dealing with overly critical reviewers"**, following the original lyrics' sentence structure while replacing specific content. It incorporates elements of reviewer nitpicking (e.g., questioning innovation, demanding redundant experiments) and expresses frustration with lines like "If only I could swap reviewers, this academic fate is too cruel" to highlight the emotional toll of peer review.
```

### 2.3 Agentic Cross-Culture Comedy
Interested in bridging cultural gaps through comedy? Transform popular English talk show segments into authentic Chinese crosstalk performances, and vice versa. Complete with cultural adaptations and localized humor that resonates with audiences of different culture backgrounds.

🚀 **Technical Details**
- Users just need to provide the target cross-talk (comedy dialogue) audio file.
- Automatically adapts the script based on the provided target audio file.
- Automatically selects the appropriate vocal tone for voice cloning according to the emotional context of the script.

🌟 **Key Features:**
- Cultural context adaptation and localization of humor
- Performance style transformation while preserving core comedic elements
- Voice generation

To produce high-quality cross-cultural comedy content, we first generate character emotion audio tracks and audience reaction effects (e.g., laughter, applause). Using LLM, we rewrite scripts line by line based on user requirements while annotating speech tones and speaker identities (`adapter.py`). We then select pre-recorded audio samples according to these tone and speaker markers, employing the open-source CosyVoice tool for voice synthesis (`synth.py`). Finally, to create corresponding video footage, we format the audio components into Movie Editing-compatible input specifications (`translator.py`), leveraging the movie editing agent pipeline to assemble the final video output.

#### 2.3.1 English Stand-up Comedy to Chinese Crosstalk
<table>
<tr>
<td align="center" width="50%">
<a href="https://www.bilibili.com/video/BV1ucZ6YmESg/  " target='_blank'><img src="assets/adapted_crosstalk_cover.png" width="100%"></a>
Chinese Crosstalk Adaptation
</td>
<td align="center" width="50%">
<a href="https://www.bilibili.com/video/BV1u1421t78T  " target='_blank'><img src="assets/standup_original_cover.png" width="100%"></a>
Original Stand-up Comedy Segment
</td>
</tr>
</table>

#### 2.3.2 Chinese Crosstalk to English Stand-up Comedy
<table>
<tr>
<td align="center" width="50%">
<a href="https://www.bilibili.com/video/BV13oZzYnEZq/  " target='_blank'><img src="assets/adapted_standupcomedy_cover.png" width="100%"></a>
Stand-up Comedy Adaptation
</td>
<td align="center" width="50%">
<a href="https://www.bilibili.com/audio/au4765690/  " target='_blank'><img src="assets/crosstalk_original_cover.png" width="100%"></a>
Original Chinese Crosstalk Segment
</td>
</tr>
</table>

## 3. Agentic Video Generation
Want to generate original, multi-modal video content from scratch?​​ Agentic-AIGC empowers you to create compelling videos by intelligently synthesizing visuals, music, and narratives into cohesive productions that bring your ideas to life.

🚀 **Technical Details**
- Users just need to provide video description and requirements.
- Automatically generates detailed shot descriptions
- Automatically maintains character and scene consistency
- Automatically assembles final videos by orchestrating editing tools

🌟 **Key Features:**
- Restore the novel's plot in Ghibli style
- Maintain consistency in characters and scenes
- Automatic addition of subtitles

<a href='https://www.bilibili.com/video/BV1NYhAzjEWn  ' target='_blank'><img src='assets/the wandering earth.png' width=60%/></a>