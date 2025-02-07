import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "ckpt",
    device_map="cuda",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("ckpt")
processor = AutoProcessor.from_pretrained("ckpt")

# Synthetic data example
timeseries = np.sin(np.arange(256) / 10) * 5.0
prompt = "<|im_start|>user\nAnalyze the local changes in this time series.<|im_end|>"

# Process and generate
inputs = processor(text=[prompt], timeseries=[timeseries], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=300)
print("\nModel Output:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))

