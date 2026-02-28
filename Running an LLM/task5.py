"""
Qwen 2.5 MMLU Evaluation Script

Evaluates 3 Qwen 2.5 models on MMLU benchmark with detailed timing.

Models:
1. Qwen2.5-0.5B-Instruct (0.5B params) - Smallest
2. Qwen2.5-1.5B-Instruct (1.5B params) - Small
3. Qwen2.5-3B-Instruct (3B params) - Medium-small

No access restrictions - all models are open and public!
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import time
import psutil
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Qwen 2.5 models - no access needed!
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # 0.5B - very small, fast
    "Qwen/Qwen2.5-1.5B-Instruct",  # 1.5B - small
    "Qwen/Qwen2.5-3B-Instruct"     # 3B - medium-small
]

# GPU settings
USE_GPU = True  # Set to False to force CPU-only execution

# Quantization settings (4, 8, or None)
QUANTIZATION_BITS = None  # Try 4 if you have memory issues

MAX_NEW_TOKENS = 1

# 10 MMLU subjects for evaluation
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy", 
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine"
]

# Verbose mode: print each question, model answer, and correctness
VERBOSE_MODE = False  # Set to True to see detailed output for each question


# ============================================================================
# Timing Utilities
# ============================================================================

class TimingTracker:
    """Track CPU and GPU timing for model evaluation"""
    
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process(os.getpid())
        self.reset()
    
    def reset(self):
        """Reset all timing counters"""
        self.start_time = None
        self.end_time = None
        self.start_cpu_time = None
        self.end_cpu_time = None
        self.gpu_time = 0
        
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.start_cpu_time = self.process.cpu_times()
        
        if self.device == "cuda":
            torch.cuda.synchronize()
    
    def stop(self):
        """Stop timing"""
        if self.device == "cuda":
            torch.cuda.synchronize()
            
        self.end_time = time.time()
        self.end_cpu_time = self.process.cpu_times()
    
    def get_times(self):
        """Get timing results in seconds"""
        real_time = self.end_time - self.start_time if self.end_time else 0
        
        if self.end_cpu_time and self.start_cpu_time:
            user_time = self.end_cpu_time.user - self.start_cpu_time.user
            system_time = self.end_cpu_time.system - self.start_cpu_time.system
            cpu_time = user_time + system_time
        else:
            cpu_time = 0
            user_time = 0
            system_time = 0
        
        return {
            "real_time": real_time,
            "cpu_time": cpu_time,
            "user_time": user_time,
            "system_time": system_time,
            "gpu_time": real_time if self.device == "cuda" else 0
        }


# ============================================================================
# Device Detection
# ============================================================================

def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""
    if not USE_GPU:
        return "cpu"
    
    if torch.cuda.is_available():
        return "cuda"
    
    if torch.backends.mps.is_available():
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"
        if is_apple_arm:
            if QUANTIZATION_BITS is not None:
                print("\n⚠️  Metal (MPS) does not support quantization")
                print("Switching to CPU for quantization support...")
                return "cpu"
            return "mps"
    
    return "cpu"


def check_environment():
    """Check environment and dependencies"""
    global QUANTIZATION_BITS  # Declare at the start of function
    
    print("="*70)
    print("Environment Check")
    print("="*70)

    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")

    device = detect_device()

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("✓ Using CPU")
       
    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"⚠️  bitsandbytes NOT installed - cannot use quantization")
            print("Install with: pip install bitsandbytes")
            print("Continuing without quantization...")
            QUANTIZATION_BITS = None
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    print("="*70 + "\n")
    return in_colab, device


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(model_name, device):
    """Load model and tokenizer with optional quantization"""
    print("="*70)
    print(f"Loading Model: {model_name}")
    print("="*70)
    print(f"Device: {device}")
    print(f"Quantization: {QUANTIZATION_BITS}-bit" if QUANTIZATION_BITS else "None (full precision)")
    
    # Estimate memory
    if "0.5B" in model_name:
        mem = "~1 GB" if QUANTIZATION_BITS is None else "~0.5 GB"
    elif "1.5B" in model_name:
        mem = "~3 GB" if QUANTIZATION_BITS is None else "~1 GB"
    elif "3B" in model_name:
        mem = "~6 GB" if QUANTIZATION_BITS is None else "~2 GB"
    else:
        mem = "~1-6 GB"
    
    print(f"Estimated memory: {mem}")
    print("="*70 + "\n")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if QUANTIZATION_BITS == 4:
        print("Loading with 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    elif QUANTIZATION_BITS == 8:
        print("Loading with 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
    else:
        print("Loading full precision model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto" if device == "cuda" else None
        )
        if device in ["cpu", "mps"]:
            model = model.to(device)
    
    model.eval()
    
    print(f"✓ Model loaded successfully")
    if device == "cuda":
        memory_allocated = torch.cuda.memory_allocated(0) / 1e9
        print(f"✓ GPU memory allocated: {memory_allocated:.2f} GB\n")
    
    return model, tokenizer


# ============================================================================
# MMLU Evaluation
# ============================================================================

def format_mmlu_prompt(question, choices):
    """Format a question in MMLU format"""
    prompt = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}. {choice}\n"
    prompt += "\nAnswer (A, B, C, or D):"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
    """Get model prediction for a single question"""
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
    """Evaluate model on a single MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {subject}")
    print(f"{'='*70}")
    
    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"✗ Failed to load {subject}: {e}")
        return None
    
    correct = 0
    total = 0
    
    timer.start()
    
    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True, disable=verbose):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]
        
        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(model, tokenizer, prompt)
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Question {total}/{len(dataset)}:")
            print(f"{'='*70}")
            print(f"{question}")
            print(f"\nChoices:")
            for i, choice in enumerate(choices):
                print(f"  {chr(65+i)}. {choice}")
            print(f"\nCorrect Answer: {correct_answer}")
            print(f"Model Answer: {predicted_answer if predicted_answer else 'NO ANSWER'}")
            print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")
            print(f"{'='*70}")
    
    timer.stop()
    timing = timer.get_times()
    
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    print(f"  Real time: {timing['real_time']:.2f}s")
    print(f"  CPU time: {timing['cpu_time']:.2f}s")
    
    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "timing": timing
    }


def evaluate_model(model_name, device):
    """Evaluate a single model on all subjects"""
    print("\n" + "="*70)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*70 + "\n")
    
    try:
        model, tokenizer = load_model_and_tokenizer(model_name, device)
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    timer = TimingTracker(device)
    
    results = []
    total_correct = 0
    total_questions = 0
    total_timing = {
        "real_time": 0,
        "cpu_time": 0,
        "user_time": 0,
        "system_time": 0,
        "gpu_time": 0
    }
    
    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")
    
    overall_start = time.time()
    
    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        
        timer.reset()
        
        result = evaluate_subject(model, tokenizer, subject, timer, verbose=VERBOSE_MODE)
        
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
            
            for key in total_timing:
                total_timing[key] += result["timing"][key]
    
    overall_end = time.time()
    total_timing["real_time"] = overall_end - overall_start
    
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print("\n" + "="*70)
    print(f"MODEL EVALUATION SUMMARY: {model_name}")
    print("="*70)
    print(f"Total Subjects: {len(results)}")
    print(f"Total Questions: {total_questions}")
    print(f"Total Correct: {total_correct}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print("\n" + "-"*70)
    print("TIMING BREAKDOWN (Cycles Consumed):")
    print("-"*70)
    print(f"Real Time (Wall Clock):  {total_timing['real_time']:.2f}s ({total_timing['real_time']/60:.2f} min)")
    print(f"CPU Time (User):         {total_timing['user_time']:.2f}s")
    print(f"CPU Time (System):       {total_timing['system_time']:.2f}s")
    print(f"CPU Time (Total):        {total_timing['cpu_time']:.2f}s")
    if device == "cuda":
        print(f"GPU Time (Estimate):     {total_timing['gpu_time']:.2f}s")
    print("="*70)
    
    del model
    del tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "model_name": model_name,
        "device": str(device),
        "quantization_bits": QUANTIZATION_BITS,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "timing": total_timing,
        "subject_results": results
    }


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("Qwen 2.5 MMLU Evaluation")
    print("="*70 + "\n")
    
    print(f"Models to evaluate: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")
    print(f"\nSubjects: {len(MMLU_SUBJECTS)}")
    print(f"Verbose mode: {'ON' if VERBOSE_MODE else 'OFF'}")
    print()

    in_colab, device = check_environment()
    
    all_results = []
    
    for model_name in MODELS:
        try:
            result = evaluate_model(model_name, device)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"\n✗ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_results:
        print("\n✗ No models were successfully evaluated!")
        return
    
    print("\n" + "="*70)
    print("COMPARISON ACROSS ALL MODELS")
    print("="*70)
    print(f"\n{'Model':<30} {'Params':<8} {'Accuracy':<12} {'Real Time':<15} {'CPU Time':<15}")
    print("-"*70)
    for result in all_results:
        model_name = result['model_name']
        
        if "0.5B" in model_name:
            params = "0.5B"
        elif "1.5B" in model_name:
            params = "1.5B"
        elif "3B" in model_name:
            params = "3B"
        else:
            params = "?"
            
        model_short = model_name.split('/')[-1][:28]
        print(f"{model_short:<30} {params:<8} {result['overall_accuracy']:>6.2f}%     "
              f"{result['timing']['real_time']/60:>6.2f} min     "
              f"{result['timing']['cpu_time']:>8.2f}s")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
    output_file = f"qwen25_mmlu_results{quant_suffix}_{timestamp}.json"
    
    output_data = {
        "timestamp": timestamp,
        "device": str(device),
        "quantization_bits": QUANTIZATION_BITS,
        "num_subjects": len(MMLU_SUBJECTS),
        "subjects": MMLU_SUBJECTS,
        "verbose_mode": VERBOSE_MODE,
        "model_results": all_results
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\n✅ Evaluation complete!")
    
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()