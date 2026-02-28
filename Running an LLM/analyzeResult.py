"""
MMLU Results Analysis and Visualization

This script:
1. Loads results from the evaluation JSON file
2. Creates graphs comparing models
3. Analyzes mistake patterns
4. Identifies common mistakes across models

Usage:
1. First run: python qwen25_mmlu_eval.py (or smallmodels.py)
2. Then run: python analyze_results.py <json_file>

Example:
  python analyzeResults.py qwen25_mmlu_results_full_20260118_123456.json
"""

import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results(json_file):
    """Load results from JSON file"""
    print(f"Loading results from: {json_file}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded results for {len(data['model_results'])} models")
    print(f"  Subjects: {data['num_subjects']}")
    print(f"  Device: {data['device']}")
    print()
    
    return data


def plot_overall_accuracy(data, output_prefix):
    """Plot 1: Overall accuracy comparison across models"""
    print("Creating Plot 1: Overall Accuracy Comparison...")
    
    models = []
    accuracies = []
    
    for result in data['model_results']:
        model_name = result['model_name'].split('/')[-1]
        models.append(model_name)
        accuracies.append(result['overall_accuracy'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#3498db', '#e74c3c', '#2ecc71'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Overall MMLU Accuracy by Model', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_overall_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_overall_accuracy.png")
    plt.close()


def plot_timing_comparison(data, output_prefix):
    """Plot 2: Timing comparison across models"""
    print("Creating Plot 2: Timing Comparison...")
    
    models = []
    real_times = []
    cpu_times = []
    
    for result in data['model_results']:
        model_name = result['model_name'].split('/')[-1]
        models.append(model_name)
        real_times.append(result['timing']['real_time'] / 60)  # Convert to minutes
        cpu_times.append(result['timing']['cpu_time'] / 60)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, real_times, width, label='Real Time', color='#3498db')
    bars2 = ax.bar(x + width/2, cpu_times, width, label='CPU Time', color='#e74c3c')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Time (minutes)', fontsize=12)
    ax.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_timing_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_timing_comparison.png")
    plt.close()


def plot_per_subject_accuracy(data, output_prefix):
    """Plot 3: Per-subject accuracy heatmap"""
    print("Creating Plot 3: Per-Subject Accuracy Heatmap...")
    
    # Build matrix of accuracies
    subjects = data['subjects']
    models = [r['model_name'].split('/')[-1] for r in data['model_results']]
    
    accuracy_matrix = []
    for result in data['model_results']:
        subject_accuracies = []
        for subject_result in result['subject_results']:
            subject_accuracies.append(subject_result['accuracy'])
        accuracy_matrix.append(subject_accuracies)
    
    accuracy_matrix = np.array(accuracy_matrix)
    
    plt.figure(figsize=(14, 6))
    sns.heatmap(accuracy_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=subjects, yticklabels=models,
                cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)
    plt.xlabel('Subject', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.title('Accuracy by Subject and Model', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_subject_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_subject_heatmap.png")
    plt.close()


def plot_accuracy_vs_time(data, output_prefix):
    """Plot 4: Accuracy vs Time (efficiency plot)"""
    print("Creating Plot 4: Accuracy vs Time...")
    
    models = []
    accuracies = []
    times = []
    
    for result in data['model_results']:
        model_name = result['model_name'].split('/')[-1]
        models.append(model_name)
        accuracies.append(result['overall_accuracy'])
        times.append(result['timing']['real_time'] / 60)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(times, accuracies, s=200, alpha=0.6, c=['#3498db', '#e74c3c', '#2ecc71'])
    
    for i, model in enumerate(models):
        plt.annotate(model, (times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.xlabel('Real Time (minutes)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Efficiency: Accuracy vs Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_efficiency.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_prefix}_efficiency.png")
    plt.close()


def analyze_mistake_patterns(data):
    """Analyze patterns in mistakes across models"""
    print("\n" + "="*70)
    print("MISTAKE PATTERN ANALYSIS")
    print("="*70)
    
    # This analysis requires the detailed per-question results
    # Since we don't have that in the JSON, we'll analyze at subject level
    
    subjects = data['subjects']
    models = [r['model_name'].split('/')[-1] for r in data['model_results']]
    
    # Find subjects where all models struggle
    print("\n1. Subjects Where ALL Models Struggle (< 40% accuracy):")
    print("-" * 70)
    
    struggling_subjects = []
    for i, subject in enumerate(subjects):
        accuracies = [r['subject_results'][i]['accuracy'] 
                     for r in data['model_results']]
        avg_accuracy = np.mean(accuracies)
        
        if avg_accuracy < 40:
            struggling_subjects.append((subject, avg_accuracy, accuracies))
    
    if struggling_subjects:
        for subject, avg_acc, accuracies in struggling_subjects:
            print(f"\n  {subject}:")
            print(f"    Average accuracy: {avg_acc:.1f}%")
            for model, acc in zip(models, accuracies):
                print(f"    {model}: {acc:.1f}%")
    else:
        print("  None - all subjects have at least one model > 40%")
    
    # Find subjects where all models excel
    print("\n2. Subjects Where ALL Models Excel (> 60% accuracy):")
    print("-" * 70)
    
    excel_subjects = []
    for i, subject in enumerate(subjects):
        accuracies = [r['subject_results'][i]['accuracy'] 
                     for r in data['model_results']]
        avg_accuracy = np.mean(accuracies)
        
        if avg_accuracy > 60:
            excel_subjects.append((subject, avg_accuracy, accuracies))
    
    if excel_subjects:
        for subject, avg_acc, accuracies in excel_subjects:
            print(f"\n  {subject}:")
            print(f"    Average accuracy: {avg_acc:.1f}%")
            for model, acc in zip(models, accuracies):
                print(f"    {model}: {acc:.1f}%")
    else:
        print("  None - no subjects where all models > 60%")
    
    # Find subjects with high variance (models disagree)
    print("\n3. Subjects with High Variance (models disagree most):")
    print("-" * 70)
    
    variance_subjects = []
    for i, subject in enumerate(subjects):
        accuracies = [r['subject_results'][i]['accuracy'] 
                     for r in data['model_results']]
        variance = np.var(accuracies)
        variance_subjects.append((subject, variance, accuracies))
    
    # Sort by variance
    variance_subjects.sort(key=lambda x: x[1], reverse=True)
    
    for subject, var, accuracies in variance_subjects[:3]:  # Top 3
        print(f"\n  {subject}:")
        print(f"    Variance: {var:.1f}")
        for model, acc in zip(models, accuracies):
            print(f"    {model}: {acc:.1f}%")
    
    # Best and worst performing model per subject
    print("\n4. Model Performance Summary:")
    print("-" * 70)
    
    best_count = defaultdict(int)
    worst_count = defaultdict(int)
    
    for i, subject in enumerate(subjects):
        accuracies = [r['subject_results'][i]['accuracy'] 
                     for r in data['model_results']]
        best_idx = np.argmax(accuracies)
        worst_idx = np.argmin(accuracies)
        
        best_count[models[best_idx]] += 1
        worst_count[models[worst_idx]] += 1
    
    print("\n  Times each model was BEST on a subject:")
    for model in models:
        print(f"    {model}: {best_count[model]} times")
    
    print("\n  Times each model was WORST on a subject:")
    for model in models:
        print(f"    {model}: {worst_count[model]} times")


def create_summary_report(data, output_prefix):
    """Create a text summary report"""
    print("\nCreating Summary Report...")
    
    report_file = f'{output_prefix}_summary_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MMLU EVALUATION SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Evaluation Date: {data['timestamp']}\n")
        f.write(f"Device: {data['device']}\n")
        f.write(f"Number of Subjects: {data['num_subjects']}\n")
        f.write(f"Quantization: {data['quantization_bits'] if data['quantization_bits'] else 'None'}\n")
        f.write(f"Verbose Mode: {data['verbose_mode']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("MODEL RESULTS\n")
        f.write("-"*70 + "\n\n")
        
        for result in data['model_results']:
            model_name = result['model_name']
            f.write(f"Model: {model_name}\n")
            f.write(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%\n")
            f.write(f"  Correct: {result['total_correct']}/{result['total_questions']}\n")
            f.write(f"  Real Time: {result['timing']['real_time']/60:.2f} minutes\n")
            f.write(f"  CPU Time: {result['timing']['cpu_time']:.2f}s\n")
            f.write(f"  User Time: {result['timing']['user_time']:.2f}s\n")
            f.write(f"  System Time: {result['timing']['system_time']:.2f}s\n")
            
            if 'gpu_time' in result['timing'] and result['timing']['gpu_time'] > 0:
                f.write(f"  GPU Time: {result['timing']['gpu_time']:.2f}s\n")
            
            f.write("\n  Per-Subject Results:\n")
            for subject_result in result['subject_results']:
                f.write(f"    {subject_result['subject']}: {subject_result['accuracy']:.2f}% ")
                f.write(f"({subject_result['correct']}/{subject_result['total']})\n")
            
            f.write("\n")
        
        f.write("-"*70 + "\n")
        f.write("SUBJECTS EVALUATED\n")
        f.write("-"*70 + "\n\n")
        for i, subject in enumerate(data['subjects'], 1):
            f.write(f"{i}. {subject}\n")
    
    print(f"  ✓ Saved: {report_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <json_file>")
        print("\nExample:")
        print("  python analyze_results.py qwen25_mmlu_results_full_20260118_123456.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    # Load results
    try:
        data = load_results(json_file)
    except FileNotFoundError:
        print(f"✗ Error: File not found: {json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"✗ Error: Invalid JSON file: {json_file}")
        sys.exit(1)
    
    # Create output prefix from input filename
    output_prefix = json_file.replace('.json', '')
    
    print("="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    # Create all plots
    plot_overall_accuracy(data, output_prefix)
    plot_timing_comparison(data, output_prefix)
    plot_per_subject_accuracy(data, output_prefix)
    plot_accuracy_vs_time(data, output_prefix)
    
    # Analyze patterns
    analyze_mistake_patterns(data)
    
    # Create summary report
    create_summary_report(data, output_prefix)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print(f"  • {output_prefix}_overall_accuracy.png")
    print(f"  • {output_prefix}_timing_comparison.png")
    print(f"  • {output_prefix}_subject_heatmap.png")
    print(f"  • {output_prefix}_efficiency.png")
    print(f"  • {output_prefix}_summary_report.txt")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()