import os
import time
import re
import sys
import asyncio
import logging
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from environment.config.llm import gpt

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, max_tokens: int = 10000):
        self.max_tokens = max_tokens
        self.timeout = 45

    async def process(self, user_idea: str, txt_path: str, pre_txt_path: str) -> str:
        """Main pipeline process - simplified to directly use presenter agent"""
        present_content = self.load_text(pre_txt_path)
        logger.info("Loaded presentation method")

        book_content = self.load_text(txt_path)
        
        word_count = len(book_content.split())
        char_count = len(book_content)
        logger.info(f"Text data statistics: {word_count} words, {char_count} characters")
        
        # Limit book content if it's too large for direct processing
        if len(book_content) > 30000:
            logger.warning(f"Content too large ({len(book_content)} chars), truncating to 30K")
            book_content = book_content[:30000]
        
        try:
            # Pass content directly to presenter agent
            presenter_output = await self.presenter_agent(user_idea, book_content, present_content)
            logger.info("Successfully generated presentation")
        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")
            presenter_output = book_content[:15000]
            logger.info("Used truncated content as fallback")
                
        try:
            formatted_content = await self.judger_agent(user_idea, presenter_output)
            logger.info("Successfully structured content with judger_agent")
        except Exception as e:
            logger.error(f"Error in judger_agent: {e}")
            formatted_content = f"/////\n{presenter_output}"
            logger.info("Used simple formatting fallback")

        return formatted_content

    def load_text(self, txt_path: str) -> str:
        """Load content from a text file"""
        encodings_to_try = ['utf-8', 'gb18030', 'gbk', 'gb2312', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read file with encoding: {encoding}")
                with open(txt_path, 'r', encoding=encoding, errors='replace') as file:
                    content = file.read()
                logger.info(f"Successfully read file with encoding: {encoding}")
                return content
            except UnicodeDecodeError:
                continue
        
        try:
            logger.info("Trying binary reading approach")
            with open(txt_path, 'rb') as file:
                content = file.read().decode('utf-8', errors='replace')
            return content
        except Exception as e:
            logger.error(f"Error loading text file: {e}")
            return ""

    @retry(
        retry=retry_if_exception_type((asyncio.TimeoutError, Exception)),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def _make_api_call(self, messages, temperature=0.7, max_tokens=None, timeout=None):
        """Make an API call with retries and exponential backoff using tenacity"""
        if max_tokens is None:
            max_tokens = min(2000, self.max_tokens)
        if timeout is None:
            timeout = self.timeout
            
        try:
            logger.info(f"Making API call with {len(messages)} messages, timeout={timeout}s")
            start_time = time.time()
            
            # Use gpt function with full messages array
            response = gpt(model="gpt-4o-mini", messages=messages)
            
            elapsed_time = time.time() - start_time
            logger.info(f"API call completed in {elapsed_time:.2f}s")
            return response.choices[0].message.content
        except asyncio.TimeoutError as e:
            logger.warning(f"API call timed out after {timeout} seconds, retrying...")
            raise e  # Re-raise to trigger tenacity retry
        except Exception as e:
            logger.warning(f"API call failed with error: {e}, retrying...")
            raise e  # Re-raise to trigger tenacity retry

    async def presenter_agent(self, user_idea: str, content: str, present_content: str) -> str:
        """Process content and adapt to user's idea - directly using input content"""
        
        # Create a unified conversation history
        messages = [
            {"role": "system", "content": "You are an experienced expert in writing review copy. Pay special attention to user's words/字数 count requirements."},
            {"role": "user", "content": f"""
            You can only create a narration copy, strictly following the user's ideas and presentation methods.


            User idea:
            "{user_idea}"
            
            Grounded text content:
            {content}

            Follow this presentation method, read it and apply it carefully:
            {present_content}
            
            Requirements:
            1. Format and Structure:
            - Response in grounded text language (English/中文)
            - Remove all chapter numbers
            - No more than 3 commas are allowed in a sentence
            - Present as one combined, coherent paragraph
            - Begin with clear story background establishment
            
            2. Content Guidelines:
            - Strictly follow user's words count/总字数 requirements
            - Use only original key dialogues (no fabricated dialogues)
            - Remove unnecessary environmental descriptions
            - Focus on plot-advancing elements
            - Do not use the " " and symbol
            
            3. Language and Style:
            - Maintain clear narrative flow (Be careful not to lose the plot in the second half of the narration copy)
            - Never mention or show user's requirements in content
            - Remove duplicated sentences
            
            Create a single, polished version that meets all these requirements.
            """}
        ]
        
        try:
            logger.info("Starting presenter agent processing")
            result = await self._make_api_call(
                messages,
                temperature=0.7,
                timeout=120,  # Increased timeout for processing full content
                max_tokens=16384
            )
            
            logger.info("Completed skit narration generation")
            print(result)
            return result
            
        except Exception as e:
            logger.error(f"Error in presenter_agent: {e}")
            return content[:15000]

    async def judger_agent(self, user_idea: str, presenter_output: str) -> str:
        """Structure the content with proper formatting based on presentation method"""
        
        # Unified conversation history for judger
        messages = [
            {"role": "system", "content": "You are a content formatting specialist with expertise in following guidelines"},
            {"role": "user", "content": f"""
            User's idea/用户的想法: "{user_idea}"

            Content to format/要结构化的内容:
            {presenter_output}

            Format above content into sections with requirements:


            - Don't response anything unrelated
            - Start each sentence with /////
            - Remove any chapter numbers
            - Keep original content 
            - The purpose is to separate each sentence
            - Remove all punctuation marks after segmentation

            Example format:
            /////\nsentence one. \n\n/////\nsentence two.
            
            #################

            将上述内容格式化为符合要求的部分：

            - 根据每个句子以 ///// 开头
            - 删除所有章节编号
            - 保留原始内容
            - 目的是为了分割每一句话
            - 划分完成后去除所有标点符号

            示例格式：
            /////\n你好吗。 \n\n/////\n早上好。
            """}
        ]
        
        for attempt in range(2):
            try:
                logger.info(f"Starting judger agent (attempt {attempt+1}/2)")
                
                result = await self._make_api_call(
                    messages,
                    temperature=0.4,
                    timeout=60,
                    max_tokens=16384
                )
                logger.info("Judger agent completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in judger_agent (attempt {attempt+1}/2): {e}")
                if attempt < 1:
                    await asyncio.sleep(2)
                    # Truncate presenter output for the retry
                    if len(presenter_output) > 15000:
                        presenter_output = presenter_output[:15000]
                        # Update the message for retry
                        messages[1]["content"] = messages[1]["content"].replace(
                            f"Content to format/要结构化的内容:\n{presenter_output}", 
                            f"Content to format/要结构化的内容:\n{presenter_output[:15000]}"
                        )
                    logger.info("Truncated presenter output for retry")
                else:
                    logger.info("Using simple formatting fallback")
                    return f"/////\n{presenter_output}"
