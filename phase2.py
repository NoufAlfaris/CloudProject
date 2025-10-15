from codecarbon import EmissionsTracker
import requests
import time
import json
import os
import requests
import re



API_URL = "http://localhost:1234/v1/chat/completions"

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

def log_hardware_specs():
    import platform, psutil
    print("\n--- Hardware Specifications ---")
    print("CPU:", platform.processor())
    print("Cores:", psutil.cpu_count(logical=False))
    print("Threads:", psutil.cpu_count(logical=True))
    print("RAM:", round(psutil.virtual_memory().total / (1024**3), 2), "GB")
    
    # Try to detect GPU info if torch is available
    try:
        import torch
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("GPU: None detected by torch")
    except ImportError:
        print("GPU: torch not installed, skipping GPU detection")



input_files = [
    "weather.txt",
    "birds.txt",
    "evaluation.txt",
    "agriculture_role.txt",
    "artificial_intelligence.txt",
    "climate_change.txt",
    "creativity_innovation.txt",
    "education_in_the_digital_age.txt",
    "genetic_engineering.txt",
    "globalization_economy.txt",
    "importance_of_sports.txt",
    "mental_health_awareness.txt",
    "social_media.txt",
    "space_exploration.txt",
    "sustainable_living.txt",
    "travel_benefits.txt",
    "water_conservation.txt",
]

os.makedirs("summaries", exist_ok=True)
run_ids = {}
results ={}
log_hardware_specs()
for file_name in input_files:
    # tracker = EmissionsTracker()
    # tracker.start()
    print(f"\n--- Processing {file_name} ---")
    tracker = EmissionsTracker(save_to_file=True)  # prevent periodic logs
    tracker.start()
    with open(file_name, "r", encoding="utf-8") as f:
        text = f.read()

    summary, inference_time = summarize_text(text)
    # emissions: dict = tracker.final_emissions_data
    emissions= tracker.stop()
    results[file_name] = {
        "energy_kWh": getattr(emissions, "energy_consumed", 0),
        "co2_kg": getattr(emissions, "emissions", emissions),  # fallback for older versions
        "inference_time_s": inference_time
    }

    run_ids[file_name] = tracker.run_id


#     # Print to console
    print("Summary:\n", summary)
    print("-" * 60)

#     # Save output
    output_name = os.path.splitext(file_name)[0] + "_summary.txt"
    output_path = os.path.join("summaries", output_name)
    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"Summary for {file_name}\n")
        out.write(f"Inference time: {inference_time:.2f} seconds\n\n")
        out.write(summary)
    





import pandas as pd

df = pd.read_csv("emissions.csv")
print("\n--- Energy Summary per File ---")
for file, data in results.items():
    print(f"{file}: Energy (kWh): {data['energy_kWh']}, CO2e (kg): {data['co2_kg']}")


# print("\n--- Energy Summary per File ---")
# for file, rid in run_ids.items():
#     row = df[df["run_id"] == rid]
#     summary = row[["energy_consumed","emissions"]].rename(columns={
#         "energy_consumed": "Energy (kWh)",
#         "emissions": "CO2e (kg)"
#     })
#     print(f"{file}:\n{summary}\n")