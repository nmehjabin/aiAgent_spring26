#!/usr/bin/env python3
"""
Llama 3.2-1B MMLU Evaluation Script - Mathematics Topic
Optimized for Ollama on Google Colab

This script evaluates Llama 3.2-1B on a single MMLU subject using Ollama.

Usage:
1. Ensure Ollama is running with llama3.2:1b model
2. Run: time python llama_mmlu_eval_math.py
   Or in Colab: !time python llama_mmlu_eval_math.py

The 'time' command will measure execution time.
"""

import requests
import json
from datasets import load_dataset
import time
from typing import Dict, List
from datetime import datetime
from tqdm.auto import tqdm
import sys

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

MODEL_NAME = "llama3.2:1b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Choose one mathematics topic from:
# "abstract_algebra", "college_mathematics", "elementary_mathematics",
# "high_school_mathematics", "high_school_statistics"
TOPIC = "astronomy"  # Change this to your desired math topic


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_ollama_connection():
    """Check if Ollama server is running and model is available"""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print(" Ollama server is running")
            
            # Check if our model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any(MODEL_NAME in name for name in model_names):
                print(f" Model '{MODEL_NAME}' is available")
                return True
            else:
                print(f"✗ Model '{MODEL_NAME}' not found")
                print(f"Available models: {model_names}")
                print(f"\nRun: ollama pull {MODEL_NAME}")
                return False
        else:
            print(f" Ollama server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(" Cannot connect to Ollama server")
        print("Make sure Ollama is running (ollama serve)")
        return False
    except Exception as e:
        print(f" Error checking Ollama: {e}")
        return False


def query_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    """Query the Ollama API with a prompt."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Low temperature for more deterministic answers
            "top_p": 0.9,
            "num_predict": 10,  # Limit response length
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()['response'].strip()
        else:
            print(f"Error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Exception during query: {e}")
        return ""

def format_question(question: str, choices: List[str]) -> str:
    """Format a multiple choice question for the model."""
    prompt = f"Question: {question}\n\n"
    prompt += "Options:\n"
    for i, choice in enumerate(choices):
        letter = chr(65 + i)  # A, B, C, D
        prompt += f"{letter}) {choice}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D) of the correct answer:"
    return prompt

def extract_answer(response: str) -> str:
    """Extract the answer letter from the model's response."""
    response = response.strip().upper()
    
    # First, look for answer at the start
    if response and response[0] in ['A', 'B', 'C', 'D']:
        return response[0]
    
    # Look for patterns like "Answer: A" or "A)" or "(A)"
    import re
    patterns = [
        r'ANSWER[:\s]+([A-D])',
        r'^([A-D])[).]',
        r'\(([A-D])\)',
        r'^([A-D])\s',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1)
    
    # Look for any A, B, C, or D in the response
    for char in response:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    # Default to A if nothing found
    return "A"

def evaluate_topic(topic: str, limit: int = None) -> Dict:
    """Evaluate the model on a specific MMLU topic."""
    print(f"\n{'='*70}")
    print(f"Evaluating topic: {topic}")
    print(f"{'='*70}\n")
    
    # Load the dataset for this topic
    try:
        dataset = load_dataset("cais/mmlu", topic, split="test")
        print(f" Dataset loaded: {len(dataset)} questions")
    except Exception as e:
        print(f" Error loading dataset for {topic}: {e}")
        return {"error": str(e)}
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        print(f"Limiting to {len(dataset)} questions for testing")
    
    correct = 0
    total = len(dataset)
    
    print(f"\nStarting evaluation...\n")
    
    # Use tqdm for progress bar
    for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {topic}")):
        question = example['question']
        choices = example['choices']
        correct_answer_idx = example['answer']
        correct_answer = chr(65 + correct_answer_idx)  # Convert 0,1,2,3 to A,B,C,D
        
        # Format and query
        prompt = format_question(question, choices)
        response = query_ollama(prompt)
        predicted_answer = extract_answer(response)
        
        # Check if correct
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct += 1
        
        # Print details for first 3 questions
        if i < 3:
            print(f"\n--- Question {i+1} ---")
            print(f"Q: {question[:100]}...")
            print(f"Predicted: {predicted_answer} | Correct: {correct_answer} | {'✓' if is_correct else '✗'}")
    
    # Final results
    accuracy = (correct / total) * 100 if total > 0 else 0
    results = {
        "topic": topic,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }
    
    print(f"\n{'='*70}")
    print(f"Results for {topic}:")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"{'='*70}\n")
    
    return results

def main():
    """Main evaluation function."""
    print("\n" + "="*70)
    print("Llama 3.2-1B MMLU Evaluation - Mathematics Topic")
    print("="*70 + "\n")
    
    # Check Ollama connection
    if not check_ollama_connection():
        print("\n✗ Ollama is not properly set up. Please fix the issues above.")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Model: {MODEL_NAME}")
    print(f"Topic: {TOPIC}")
    print(f"Ollama URL: {OLLAMA_URL}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    # Record start time
    start_time = time.time()
    
    # Run evaluation (set limit=10 for quick testing, None for full evaluation)
    results = evaluate_topic(TOPIC, limit=None)
    
    # Record end time
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print timing summary
    print(f"{'='*70}")
    print("Timing Summary")
    print(f"{'='*70}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")
    print(f"                      {elapsed_time/60:.2f} minutes")
    if results.get('total', 0) > 0:
        print(f"Time per question:    {elapsed_time/results['total']:.2f} seconds")
    print(f"{'='*70}\n")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{TOPIC}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            **results,
            "execution_time_seconds": elapsed_time,
            "execution_time_minutes": elapsed_time / 60,
            "model": MODEL_NAME,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"✓ Results saved to: {output_file}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)