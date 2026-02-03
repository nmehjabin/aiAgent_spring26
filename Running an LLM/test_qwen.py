"""
Quick Test - Qwen 2.5 MMLU Evaluation

Tests with just the smallest model (0.5B) and 2 subjects.
No access needed - Qwen models are fully open!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import time
import psutil
import os
import platform
from datetime import datetime

# Just test with the smallest model
MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]

USE_GPU = True
MAX_NEW_TOKENS = 1

# Only 2 subjects for quick testing
MMLU_SUBJECTS = ["astronomy", "business_ethics"]

VERBOSE_MODE = False  # Set to True to see each question


class TimingTracker:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.end_time = None
        self.start_cpu_time = None
    
    def start(self):
        self.start_time = time.time()
        self.start_cpu_time = self.process.cpu_times()
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def stop(self):
        if self.device == "cuda":
            torch.cuda.synchronize()
        self.end_time = time.time()
        self.end_cpu_time = self.process.cpu_times()
    
    def get_times(self):
        real_time = self.end_time - self.start_time if self.end_time else 0
        if self.end_cpu_time and self.start_cpu_time:
            user_time = self.end_cpu_time.user - self.start_cpu_time.user
            system_time = self.end_cpu_time.system - self.start_cpu_time.system
            cpu_time = user_time + system_time
        else:
            cpu_time = user_time = system_time = 0
        
        return {
            "real_time": real_time,
            "cpu_time": cpu_time,
            "user_time": user_time,
            "system_time": system_time
        }


def detect_device():
    if not USE_GPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def check_environment():
    print("="*70)
    print("Quick Test - Qwen 2.5")
    print("="*70)
    
    device = detect_device()
    
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    elif device == "mps":
        print("✓ Using Apple Metal (MPS)")
    else:
        print("✓ Using CPU")
    
    print(f"✓ Platform: {platform.system()}")
    print("="*70 + "\n")
    return device


def load_model(model_name, device):
    print(f"Loading {model_name}...")
    print("Expected memory: ~1 GB")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None
    )
    
    if device in ["cpu", "mps"]:
        model = model.to(device)
    
    model.eval()
    print("✓ Loaded\n")
    
    if device == "cuda":
        mem = torch.cuda.memory_allocated(0) / 1e9
        print(f"GPU memory: {mem:.2f} GB\n")
    
    return model, tokenizer


def format_prompt(question, choices):
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer (A, B, C, or D):"
    return prompt


def get_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    response = response.strip()
    
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    return None


def evaluate_subject(model, tokenizer, subject, timer, verbose=False):
    print(f"\nEvaluating: {subject}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None
    
    correct = 0
    total = 0
    
    timer.start()
    
    for example in tqdm(dataset, desc=subject, disable=verbose):
        question = example["question"]
        choices = example["choices"]
        correct_answer = ["A", "B", "C", "D"][example["answer"]]
        
        prompt = format_prompt(question, choices)
        predicted_answer = get_prediction(model, tokenizer, prompt)
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        if verbose:
            print(f"\nQ{total}: {question}")
            print(f"Correct: {correct_answer}, Model: {predicted_answer or 'NONE'}")
            print(f"{'✓ RIGHT' if is_correct else '✗ WRONG'}")
    
    timer.stop()
    timing = timer.get_times()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ {correct}/{total} = {accuracy:.2f}%")
    print(f"  Time: {timing['real_time']:.1f}s (CPU: {timing['cpu_time']:.1f}s)")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "timing": timing
    }


def main():
    print("\n" + "="*70)
    print("QUICK TEST - Qwen 2.5 (0.5B)")
    print("="*70)
    print(f"Model: {MODELS[0]}")
    print(f"Subjects: {len(MMLU_SUBJECTS)}")
    print(f"Verbose: {VERBOSE_MODE}\n")
    
    device = check_environment()
    
    try:
        model, tokenizer = load_model(MODELS[0], device)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nMake sure you have: pip install transformers torch datasets tqdm psutil")
        return
    
    timer = TimingTracker(device)
    results = []
    total_correct = 0
    total_questions = 0
    
    overall_start = time.time()
    
    for subject in MMLU_SUBJECTS:
        timer.reset()
        result = evaluate_subject(model, tokenizer, subject, timer, VERBOSE_MODE)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
    
    overall_time = time.time() - overall_start
    
    accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    total_cpu = sum(r["timing"]["cpu_time"] for r in results)
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Accuracy: {accuracy:.2f}% ({total_correct}/{total_questions})")
    print(f"Real Time: {overall_time:.2f}s ({overall_time/60:.2f} min)")
    print(f"CPU Time: {total_cpu:.2f}s")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"test_qwen_{timestamp}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "model": MODELS[0],
            "accuracy": accuracy,
            "real_time": overall_time,
            "cpu_time": total_cpu,
            "results": results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n✅ Test complete!")
    print("\nIf this worked, run the full version:")
    print("  python qwen25_mmlu_eval.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()