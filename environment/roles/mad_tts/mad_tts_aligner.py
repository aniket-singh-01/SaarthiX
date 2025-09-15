from pathlib import Path
from openai import OpenAI
from environment.config.llm import deepseek
from environment.agents.base import BaseAgent
from environment.communication.message import Message
from environment.config.config import config


class MadTTSAligner(BaseAgent):
    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=config['llm']['api_key'],
            base_url=config['llm']['base_url']
        )
        self.max_retries = 3

    def process_message(self, message):
        audio_path = message.content.get("audio_dir")
        if not audio_path:
            return Message(
                content={
                    "status": "error",
                    "message": "No audio_path provided in the message"
                }
            )

        path = Path(audio_path)
        speech_path = path.parent / "speech.txt"

        if not speech_path.exists():
            return Message(
                content={
                    "status": "error",
                    "message": f"Speech file not found at {speech_path}"
                }
            )

        lab_files = sorted(path.glob("*.lab"))
        if not lab_files:
            return Message(
                content={
                    "status": "error",
                    "message": f"No .lab files found in {path}"
                }
            )

        with open(speech_path, "r", encoding="utf-8") as f:
            speech = f.read().strip()

        splits = []
        last_split = ""

        for i, lab_file in enumerate(lab_files):
            with open(lab_file, "r", encoding="utf-8") as f:
                lab_text = f.read().strip()

            if not lab_text:
                continue

            user_prompt = self._create_initial_prompt(speech, lab_text, i, last_split)
            retry_count = 0
            best_attempt = ""
            validation_passed = False

            while retry_count < self.max_retries and not validation_passed:
                try:
                    response = deepseek(user=user_prompt)
                    candidate = response.choices[0].message.content.strip()
                    print(candidate)
                    validation_result= self._validate_segment(
                        candidate=candidate,
                        reference=lab_text,
                        full_text=speech,
                        previous_segment=last_split
                    )

                    if validation_result:
                        best_attempt = candidate
                        validation_passed = True
                    else:
                        user_prompt = self._create_refinement_prompt(
                            candidate=candidate,
                            reference=lab_text,
                            full_text=speech,
                            previous_segment=last_split
                        )
                        retry_count += 1

                except Exception as e:
                    print(f"Error processing {lab_file} (attempt {retry_count}): {str(e)}")
                    break

            if validation_passed:
                splits.append(best_attempt)
                last_split = best_attempt
                print(f"Validated segment {i}: {best_attempt}")
            else:
                splits.append(candidate if candidate else "")
                print(f"Used unvalidated segment for {lab_file}, candidate: {candidate}")

        speech_dir = path / "speech"
        speech_dir.mkdir(exist_ok=True)

        for i, split in enumerate(splits):
            output_file = speech_dir / f"{i}.lab"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(split)

        return Message(
            content={
                "status": "success",
                "message": splits
            }
        )

    def _create_initial_prompt(self, speech, reference, index, previous):
        return f"""【初始任务】
        你是一个专业的文本对齐助手。请严格按照以下要求处理文本：

        1. 待处理文本：
        {speech}

        2. 参考片段：
        {reference}

        3. 上下文：
        {f"上一个截取片段：{previous}" if index > 0 else "这是第一个片段，从开头开始查找"}

        4. 要求：
        - 找出待处理文本中与参考片段最接近的最小连续片段
        - 字数相近（约浮动5个字左右）
        - 结构相似（如标点位置、句式）
        - 动名词相对位置的匹配度
        - 不需要考虑具体内容和主题是否相近
        - 不要修改片段内容,只要截取待处理文本前后多余部分
        - 输出仅包含截取片段本身

        请直接输出最佳匹配片段，不要在前后输出辅助字符："""

    def _validate_segment(self, candidate, reference, full_text, previous_segment):
        """反思验证并返回反馈"""
        validation_prompt = f"""【反思验证】
        请严格检查以下对齐结果：

        1. 参考片段：
        {reference}

        2. 候选片段：
        {candidate}

        3. 完整文本：
        {full_text}

        4. 验证标准：
        - 候选是否是与参考最接近的最小片段？
        - 字数是否相近？（参考：{len(reference)}字，候选：{len(candidate)}字）
        - 结构是否相似（标点位置、句式）？
        - 根据二者动名词等在句中的相对位置是否匹配
        - 不需要考虑具体内容和主题是否相近.
        - 是否在正确位置查找（{"无前序片段" if not previous_segment else f"前序片段结尾：{previous_segment[-10:]}"}）？
        
        请严格按以下格式回答,不要输出多余解释：
        <验证结果>（通过/不通过）"""

        try:
            response = deepseek(user=validation_prompt)
            result_text = response.choices[0].message.content
            print(result_text)
            # 解析响应
            if "<验证结果>通过" in result_text:
                return True
            else:
                return False

        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return False, "验证过程出错"

    def _create_refinement_prompt(self, candidate, reference, full_text, previous_segment):
        return f"""【修正任务】
        根据反馈改进对齐结果：

        1. 原候选片段：
        {candidate}

        2. 参考片段：
        {reference}

        3. 完整文本：
        {full_text}

        4. 额外要求：
        - 只能从原候选片段的前后进行截取或补充，不要修改句中的内容
        - 考虑字数匹配（参考：{len(reference)}字，候选：{len(candidate)}字）
        - 考虑结构相近（标点位置、句式）
        - 不需要考虑具体内容和主题是否相近
        - 不需要考虑语义完整，例如参考片段缺少了动名词，截取时也须按照参考片段的结构，不要擅自补充
        - 确保在正确位置继续查找（{"从开头开始" if not previous_segment else f"前序片段结尾：{previous_segment[-10:]}"}）
        - 如果需要截取或者补充,只截取原候选片段的冗余部分或者补充原候选片段的缺失部分
        - 如果不需要截取或者补充,返回原候选片段即可
        
        请输出修正后的最佳候选匹配片段，不要在前后添加辅助字符："""