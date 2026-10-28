import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Device:", device)
print("Loading model and processor...")

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model = model.to(device)
model.eval()

# A simple test image created locally; no dataset download is needed.
image = Image.open("test_objects.png").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one sentence."},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages, add_generation_prompt=True
)
inputs = processor(
    images=image, text=prompt, return_tensors="pt"
).to(device)

print("Generating response...")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=40, do_sample=False)

new_tokens = output[:, inputs["input_ids"].shape[1]:]
print("Model response:", processor.batch_decode(
    new_tokens, skip_special_tokens=True
)[0])
print("One-image inference completed.")
