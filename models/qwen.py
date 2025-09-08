from models.base_model import BaseModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None
from qwen_vl_utils import process_vision_info
import torch

class Qwen2VL(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        max_pixels = 2048*28*28
        # Read device and memory options from config
        device_map = getattr(self.config, 'device_map', "balanced_low_0")
        max_memory = getattr(self.config, 'max_memory', None)
        torch_dtype = getattr(self.config, 'torch_dtype', "auto")

        # Optional 8-bit / 4-bit quantization (requires bitsandbytes)
        quantization_config = None
        if BitsAndBytesConfig is not None:
            if getattr(self.config, 'load_in_8bit', False):
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif getattr(self.config, 'load_in_4bit', False):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=getattr(self.config, 'bnb_4bit_quant_type', 'nf4'),
                    bnb_4bit_compute_dtype=torch.float16,
                )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
        )
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)  # , max_pixels=max_pixels
        self.create_ask_message = lambda question: {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        }
        self.create_ans_message = lambda ans: {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ans},
            ],
        }
        
    def create_text_message(self, texts, question):
        content = []
        for text in texts:
            content.append({"type": "text", "text": text})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
        
    def create_image_message(self, images, question):
        content = []
        for image_path in images:
            content.append({"type": "image", "image": image_path})
        content.append({"type": "text", "text": question})
        message = {
            "role": "user",
            "content": content
        }
        return message
    
    @torch.no_grad()
    def predict(self, question, texts = None, images = None, history = None):
        self.clean_up()
        messages = self.process_message(question, texts, images, history)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        generated_ids = self.model.generate(**inputs, max_new_tokens=self.config.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        messages.append(self.create_ans_message(output_text))
        self.clean_up()
        return output_text, messages
        
    def is_valid_history(self, history):
        if not isinstance(history, list):
            return False
        for item in history:
            if not isinstance(item, dict):
                return False
            if "role" not in item or "content" not in item:
                return False
            if not isinstance(item["role"], str) or not isinstance(item["content"], list):
                return False
            for content in item["content"]:
                if not isinstance(content, dict):
                    return False
                if "type" not in content:
                    return False
                if content["type"] not in content:
                    return False
        return True
