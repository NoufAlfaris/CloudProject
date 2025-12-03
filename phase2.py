
import time
import json
import os
import re
import statistics
import requests
import psutil
import threading
import platform
from datetime import datetime

#for graphs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np



API_URL = "http://localhost:1234/v1/chat/completions"
#device's hardware specifications
def get_hardware_specs():
    """Collect detailed hardware specifications"""
    try:
        specs = {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else "N/A",
        }
        return specs
    except Exception as e:
        print(f"Error collecting hardware specs: {e}")
        return {}

def get_process_power_estimate():
    """Estimate energy using CPU/process metrics"""
    try:
        # Get CPU usage percentage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Get current process info
        current_process = psutil.Process()
        process_cpu_percent = current_process.cpu_percent(interval=0.1)

        # Estimate base power consumption
        base_power_mw = 5000  # 5W baseline
        cpu_power_mw = (cpu_percent / 100) * 30000  # Scale to 30W max CPU

        # Memory power estimate 
        ram_percent = psutil.virtual_memory().percent
        ram_power_mw = (ram_percent / 100) * 2000  # Roughly 2W max for RAM
        
        return {
            "cpu_percent": cpu_percent,
            "process_cpu_percent": process_cpu_percent,
            "ram_percent": ram_percent,
            "cpu_power_mw": cpu_power_mw,
            "ram_power_mw": ram_power_mw,
            "estimated_total_mw": base_power_mw + cpu_power_mw + ram_power_mw
        }
    except Exception as e:
        print(f"Error getting process metrics: {e}")
        return {"cpu_percent": 0, "process_cpu_percent": 0, "estimated_total_mw": 5000}
    

def calculate_energy_from_samples(power_samples_mw, duration_seconds):
    """Convert power measurements to energy (kWh)"""
    if not power_samples_mw:
        return 0
    
    avg_power_mw = statistics.mean(power_samples_mw)
    avg_power_w = avg_power_mw / 1000
    duration_hours = duration_seconds / 3600
    energy_kwh = (avg_power_w * duration_hours) / 1000
    
    return energy_kwh

def estimate_co2(energy_kwh):
    """Rough CO2 estimate"""
    co2_per_kwh = 0.5
    return energy_kwh * co2_per_kwh

#categorize based on our input size
def get_text_size_category(filename, text):
    word_count = len(text.split())

    if filename.endswith('S.txt'):
        category = "Short"
    elif filename.endswith('M.txt'):
        category = "Medium"
    elif filename.endswith('L.txt'):
        category = "Large"
    
    return category, word_count

#connecting with llm
def summarize_text(text):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek-r1-distill-qwen-7b",
        "messages": [
            {"role": "system", "content": "Summarize the following text clearly and concisely."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.3,
    }
    start = time.time()
    response = requests.post(API_URL, headers=headers, data=json.dumps(data))
    end = time.time()

    summary = response.json()["choices"][0]["message"]["content"]
    import re 
    summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
    inferenceTime = end - start
    return summary , inferenceTime

def log_hardware_specs(specs):
    print("\n" + "="*80)
    print("HARDWARE SPECIFICATIONS")
    print("="*80)
    print(f"System: {specs.get('system')} ({specs.get('machine')})")
    print(f"Processor: {specs.get('processor')}")
    print(f"CPU Cores (Physical): {specs.get('cpu_count_physical')}")
    print(f"CPU Threads (Logical): {specs.get('cpu_count_logical')}")
    print(f"CPU Frequency: {specs.get('cpu_freq_mhz')} MHz")
    print(f"RAM: {specs.get('ram_gb')} GB")
    print(f"GPU: Apple Silicon M3 (Integrated)")
    print("="*80 + "\n")

#visualizing data function
def generate_graphs(results):
    """Generate all visualization graphs"""
    Short_results = [r for r in results.values() if r['size_category'] == 'Short']
    medium_results = [r for r in results.values() if r['size_category'] == 'Medium']
    large_results = [r for r in results.values() if r['size_category'] == 'Large']
    
    os.makedirs("graphs", exist_ok=True)
    
    #prepare data for graphs
    categories = ['Short', 'Medium', 'Large']
    colors = ['#3b82f6', '#10b981', '#f59e0b']

    
    #calculate averages
    avg_inference_times = [
        statistics.mean([r['inference_time_s'] for r in Short_results]) if Short_results else 0,
        statistics.mean([r['inference_time_s'] for r in medium_results]) if medium_results else 0,
        statistics.mean([r['inference_time_s'] for r in large_results]) if large_results else 0,
    ]
    
    avg_cpu_powers = [
        statistics.mean([r['avg_cpu_power_mw'] for r in Short_results]) if Short_results else 0,
        statistics.mean([r['avg_cpu_power_mw'] for r in medium_results]) if medium_results else 0,
        statistics.mean([r['avg_cpu_power_mw'] for r in large_results]) if large_results else 0,
    ]
    
    avg_ram_powers = [
        statistics.mean([r['avg_ram_power_mw'] for r in Short_results]) if Short_results else 0,
        statistics.mean([r['avg_ram_power_mw'] for r in medium_results]) if medium_results else 0,
        statistics.mean([r['avg_ram_power_mw'] for r in large_results]) if large_results else 0,
    ]
    
    avg_total_powers = [
        statistics.mean([r['avg_total_power_mw'] for r in Short_results]) if Short_results else 0,
        statistics.mean([r['avg_total_power_mw'] for r in medium_results]) if medium_results else 0,
        statistics.mean([r['avg_total_power_mw'] for r in large_results]) if large_results else 0,
    ]
    
    avg_energies = [
        statistics.mean([r['total_energy_kwh'] for r in Short_results]) if Short_results else 0,
        statistics.mean([r['total_energy_kwh'] for r in medium_results]) if medium_results else 0,
        statistics.mean([r['total_energy_kwh'] for r in large_results]) if large_results else 0,
    ]
    
    #graph1
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(categories))
    bars = ax.bar(x_pos, avg_inference_times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Average Inference Time by Input Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, value) in enumerate(zip(bars, avg_inference_times)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, f'{value:.2f}s', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/01_inference_time_comparison.png', dpi=300, bbox_inches='tight')
   
    plt.close()


    
   #graph2
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x_pos, avg_energies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Energy (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Average Total Energy Consumption by Input Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, value) in enumerate(zip(bars, avg_energies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e-9, f'{value:.2e}', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/02_total_energy_comparison.png', dpi=300, bbox_inches='tight')

    plt.close()

    
    #graph3
    fig, ax = plt.subplots(figsize=(10, 6))
    all_energies = [r['total_energy_kwh'] for r in results.values()]
    all_times = [r['inference_time_s'] for r in results.values()]
    all_categories = [r['size_category'] for r in results.values()]
    
    for i, category in enumerate(['Short', 'Medium', 'Large']):
        cat_energies = [all_energies[j] for j, cat in enumerate(all_categories) if cat == category]
        cat_times = [all_times[j] for j, cat in enumerate(all_categories) if cat == category]
        ax.scatter(cat_times, cat_energies, label=category, s=150, alpha=0.7, edgecolors='black', linewidth=1.5, color=colors[i])
    
    ax.set_xlabel('Inference Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Consumed (kWh)', fontsize=12, fontweight='bold')
    ax.set_title('Energy vs Inference Time Correlation', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('graphs/03_energy_vs_inference_scatter.png', dpi=300, bbox_inches='tight')
  
    plt.close()



input_files = [
    "weatherS.txt",
    "birdsM.txt",
    "evaluationL.txt",
    "agriculture_roleM.txt",
    "artificial_intelligenceS.txt",
    "climate_changeS.txt",
    "creativity_innovationM.txt",
    "education_in_the_digital_ageS.txt",
    "genetic_engineeringL.txt",
    "globalization_economyM.txt",
    "importance_of_sportsS.txt",
    "mental_health_awarenessL.txt",
    "renewable_energyL.txt",
    "social_mediaM.txt",
    "space_explorationS.txt",
    "sustainable_livingL.txt",
    "travel_benefitsL.txt",
    "water_conservationM.txt",
]

os.makedirs("summaries", exist_ok=True)
results ={}
hardware_specs = get_hardware_specs()
log_hardware_specs(hardware_specs)
#loop to process each file
for file_name in input_files:
    if not os.path.exists(file_name):
        print(f"⚠️  File not found: {file_name}, skipping...")
        continue
    
    print(f"\n--- Processing {file_name} ---")
    

    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()
    
    size_category, word_count= get_text_size_category(file_name, text)
    print(f"Size: {size_category} | Words: {word_count}")
    
    #Collect baseline power samples
    baseline_samples = []
    baseline_cpu = []
    baseline_ram = []
    
    for _ in range(5):
        metrics = get_process_power_estimate()
        baseline_samples.append(metrics["estimated_total_mw"])
        baseline_cpu.append(metrics["cpu_power_mw"])
        baseline_ram.append(metrics["ram_power_mw"])
        time.sleep(0.05)
    
    #process file with power monitoring
    power_samples = []
    cpu_samples = []
    ram_samples = []
    sample_thread_active = True
    
     #parallel power consumption measures
    def collect_samples():
        while sample_thread_active:
            metrics = get_process_power_estimate()
            power_samples.append(metrics["estimated_total_mw"])
            cpu_samples.append(metrics["cpu_power_mw"])
            ram_samples.append(metrics["ram_power_mw"])
            time.sleep(0.05)
    
   
    sample_thread = threading.Thread(target=collect_samples, daemon=True)
    sample_thread.start()
    
    start_time = time.time()
    summary, inference_time = summarize_text(text)
    end_time = time.time()
    
    sample_thread_active = False
    time.sleep(0.1) 
    
    actual_duration = end_time - start_time
    
    #combine all samples
    all_samples = baseline_samples + power_samples
    all_cpu = baseline_cpu + cpu_samples
    all_ram = baseline_ram + ram_samples
    
    #calculate energy consumption & avg powers
    total_energy_kwh = calculate_energy_from_samples(all_samples, actual_duration)
    cpu_energy_kwh = calculate_energy_from_samples(all_cpu, actual_duration)
    ram_energy_kwh = calculate_energy_from_samples(all_ram, actual_duration)
    co2_kg = estimate_co2(total_energy_kwh)
 
    avg_total_power_mw = statistics.mean(all_samples) if all_samples else 0
    avg_cpu_power_mw = statistics.mean(all_cpu) if all_cpu else 0
    avg_ram_power_mw = statistics.mean(all_ram) if all_ram else 0
    
    summary_word_count = len(summary.split())
    
    #detailed results to save to the json file
    results[file_name] = {
        "size_category": size_category,
        "input_word_count": word_count,
        "output_word_count": summary_word_count,
        "inference_time_s": inference_time,
        "total_energy_kwh": total_energy_kwh,
        "cpu_energy_kwh": cpu_energy_kwh,
        "ram_energy_kwh": ram_energy_kwh,
        "co2_kg": co2_kg,
        "avg_total_power_mw": avg_total_power_mw,
        "avg_cpu_power_mw": avg_cpu_power_mw,
        "avg_ram_power_mw": avg_ram_power_mw,
        "energy_per_input_token": (total_energy_kwh / word_count * 1000000) if word_count > 0 else 0,
        "compression_ratio": (word_count / summary_word_count) if summary_word_count > 0 else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Inference time: {inference_time:.2f}s")
    print(f"Total energy: {total_energy_kwh:.8f} kWh")
    print(f"CPU energy: {cpu_energy_kwh:.8f} kWh")
    print(f"RAM energy: {ram_energy_kwh:.8f} kWh")
    print(f"CO2: {co2_kg:.8f} kg")
    print(f"Compression: {word_count} → {summary_word_count} words ({results[file_name]['compression_ratio']:.2f}x)")


Short_results = [r for r in results.values() if r['size_category'] == 'Short']
medium_results = [r for r in results.values() if r['size_category'] == 'Medium']
large_results = [r for r in results.values() if r['size_category'] == 'Large']

#save results
with open("energy_results_detailed.json", "w") as f:
    json.dump({
        "hardware_specs": hardware_specs,
        "results": results,
        "summary": {
            "Short": {
                "count": len(Short_results),
                "avg_energy": statistics.mean([r['total_energy_kwh'] for r in Short_results]) if Short_results else 0,
                "avg_time": statistics.mean([r['inference_time_s'] for r in Short_results]) if Short_results else 0,
                "avg_cpu_power": statistics.mean([r['avg_cpu_power_mw'] for r in Short_results]) if Short_results else 0,
                "avg_ram_power": statistics.mean([r['avg_ram_power_mw'] for r in Short_results]) if Short_results else 0,
            },
            "medium": {
                "count": len(medium_results),
                "avg_energy": statistics.mean([r['total_energy_kwh'] for r in medium_results]) if medium_results else 0,
                "avg_time": statistics.mean([r['inference_time_s'] for r in medium_results]) if medium_results else 0,
                "avg_cpu_power": statistics.mean([r['avg_cpu_power_mw'] for r in medium_results]) if medium_results else 0,
                "avg_ram_power": statistics.mean([r['avg_ram_power_mw'] for r in medium_results]) if medium_results else 0,
            },
            "large": {
                "count": len(large_results),
                "avg_energy": statistics.mean([r['total_energy_kwh'] for r in large_results]) if large_results else 0,
                "avg_time": statistics.mean([r['inference_time_s'] for r in large_results]) if large_results else 0,
                "avg_cpu_power": statistics.mean([r['avg_cpu_power_mw'] for r in large_results]) if large_results else 0,
                "avg_ram_power": statistics.mean([r['avg_ram_power_mw'] for r in large_results]) if large_results else 0,
            }
        }
    }, f, indent=2)

generate_graphs(results)
