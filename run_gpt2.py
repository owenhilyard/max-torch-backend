import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, CompileConfig
from max_torch_backend import MaxCompiler

# Suppress tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
compile_config = CompileConfig(backend=MaxCompiler, fullgraph=False)


def main():
    # Model and tokenizer setup
    model_name = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for faster inference
        device_map="auto",
    )
    model.eval()

    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prompt for text generation
    prompt = "The future of artificial intelligence is"

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()
        model = model.cuda()

    model.__call__ = torch.compile(model.__call__, backend=MaxCompiler)

    # Generate text
    print("Generating text with torch.compile optimized model...")
    print(f"Prompt: {prompt}")

    with torch.no_grad():
        # First call will be slow due to compilation
        outputs = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 50,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            compile_config=compile_config,
        )

    # Decode and print results
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")

    # Subsequent calls will be much faster
    print("\nGenerating again (faster compilation)...")
    with torch.no_grad():
        outputs2 = model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 30,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            compile_config=compile_config,
        )

    generated_text2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)
    print(f"Generated: {generated_text2}")


if __name__ == "__main__":
    main()
