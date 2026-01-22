

'''
conda deactivate

cd vlm_learn/
conda activate qwenvl
export PYTHONNOUSERSITE=1
export HF_TOKEN= your_HF_Tocken
python QwenAttack/Qwen2_5_Inference.py
'''




from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from PIL import Image

# Load model and processor
model_path = "QwenAttack/Qwen2.5-VL-7B-Instruct"
model = AutoModelForVision2Seq.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_path)

# Load local image
image_path = "QwenAttack/dataSamples/dogs68.jpg"


try:
    image = Image.open(image_path).convert("RGB")
    print(f"Successfully loaded image: {image_path}")
    print(f"Image size: {image.size}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"}
        ]
    }
]

#print("image.shape", image.shape)

# Process inputs
text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=[text_prompt], images=[image], return_tensors="pt")
inputs = inputs.to(model.device)  # Use model.device instead of hardcoded "cuda"

# Generate
print("\nGenerating response...")
with torch.no_grad():
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=True,  # Optional: for more creative responses
        temperature=0.7,  # Optional: control randomness
    )
    
generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
print("\n=== MODEL RESPONSE ===")
print(generated_texts[0])