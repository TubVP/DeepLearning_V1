import argparse
import torch
import re
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def load_model(model_dir="model_weights"):
    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(">>> Loading model on GPU/CPU (float16)...")
    dtype = torch.float16  
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    devices = set(str(param.device) for _, param in model.named_parameters())
    print(f">>> Model loaded on devices: {devices}")
    if any("cuda" in str(d) for d in devices):
        print(">>> GPU is active.")
    else:
        print(">>> Running on CPU.")

    return tokenizer, model

def generate_code(model, tokenizer, prompt, max_new_tokens=1024):
    print(">>> Tokenizing input...")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(">>> Generating code...")
    start_time = time.time()
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

    outputs = model.generate(
        **inputs,
        generation_config=gen_config
    )

    code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elapsed = time.time() - start_time
    print(f">>> Code generation time: {elapsed:.2f}s")

    return code

def extract_python_code(text):
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return "\n".join(match.strip() for match in matches)
    else:
        return text.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, required=True, help="Path to problem.md")
    parser.add_argument("--output", type=str, required=True, help="Path to save solution.py")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max tokens to generate")
    args = parser.parse_args()

    with open(args.problem, "r", encoding="utf-8") as f:
        prompt = f.read().strip()

    tokenizer, model = load_model()
    raw_code = generate_code(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
    python_code = extract_python_code(raw_code)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(python_code)

    print(f">>> Code generated and saved to {args.output}")

if __name__ == "__main__":
    main()