import boto3
import time
import json
import os
import re
import statistics
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np


CLOUD_CONFIG = {
    "provider": "aws_sagemaker",
    "endpoint_name": "flan-t5-summarization-v1",
    "region": "us-east-1", 
    "model_name": "google/flan-t5-base",
    "instance_cost_per_hour": 0.23,  #ml.m5.xlarge cost in $
}

runtime_client = boto3.client("sagemaker-runtime", region_name=CLOUD_CONFIG["region"])


def summarize_text(text, config):
    
    payload = {
        "inputs": f"Summarize the following text:\n\n{text}",
        "parameters": {
            "max_length": 200,
            "min_length": 30,
            "do_sample": False,
            "temperature": 1.0
        }
    }
    
    payload_bytes = json.dumps(payload).encode("utf-8")
    
    start_time = time.time()
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=config["endpoint_name"],
            ContentType="application/json",
            Body=payload_bytes
        )
        
        end_time = time.time()
        response_body = response['Body'].read()
        total_time = end_time - start_time
        
    
        result_json = json.loads(response_body)
        
        #Flan-T5 returns summary_text": "..." or "generated_text": "..." so delete that from the response
        if isinstance(result_json, list) and len(result_json) > 0:
            first_item = result_json[0]
            if isinstance(first_item, dict):
                summary = first_item.get('summary_text') or first_item.get('generated_text', '')
            else:
                summary = str(first_item)
        elif isinstance(result_json, dict):
            summary = result_json.get('summary_text') or result_json.get('generated_text', str(result_json))
        else:
            summary = str(result_json)
        
        #cost
        cost_per_second = config["instance_cost_per_hour"] / 3600
        estimated_cost = cost_per_second * total_time
        
        metrics = {
            "success": True,
            "total_time_s": total_time,
            "total_time_ms": total_time * 1000,
            "request_size_bytes": len(payload_bytes),
            "response_size_bytes": len(response_body),
            "total_transfer_bytes": len(payload_bytes) + len(response_body),
            "status_code": response.get("ResponseMetadata", {}).get("HTTPStatusCode", 200),
            "estimated_cost_usd": estimated_cost,
            "timestamp": datetime.now().isoformat()
        }
        
        return summary, metrics
    
    except Exception as e:
        return "", {
            "success": False,
            "error": str(e),
            "total_time_s": 0,
            "estimated_cost_usd": 0
        }



def token_count(text):
    return int(len(text) * 0.25) #common words are 1 token but other words are divided so a logical estimation is 1 token per 4 characters


def get_text_size_category(filename, text):
    
    word_count = len(text.split())
    
    if filename.endswith('S.txt'):
        category = "Short"
    elif filename.endswith('M.txt'):
        category = "Medium"
    elif filename.endswith('L.txt'):
        category = "Large"

    
    return category, word_count


def measure_baseline_overhead(endpoint_name, samples=5):
    #Measure baseline overhead with minimal payload
    latencies = []
    successful = 0
    
    
    for i in range(samples):
        start = time.time()
        try:
            runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=json.dumps({"inputs": "Test"}).encode("utf-8")
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            successful += 1
            print(f"  Sample {i+1}: {latency:.2f}ms")
        except Exception as e:
            print(f"  Sample {i+1}: Failed - {e}")
        
        time.sleep(0.3)
    
    if latencies:
        result = {
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "avg_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "std_dev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "successful_samples": successful,
            "total_samples": samples
        }
        print(f"Baseline: avg={result['avg_ms']:.2f}ms\n")
        return result
    
    print("❌ Failed to measure baseline\n")
    return None


def generate_graphs(results):
    categories = ['Short', 'Medium', 'Large']
    colors = ['#3b82f6', '#10b981', '#f59e0b']
    
    os.makedirs("graphs", exist_ok=True)

    avg_costs = []
    avg_times = []
    avg_transfer = []

    for cat in categories:
        cat_results = [r for r in results.values() if r['size_category'] == cat]
        if cat_results:
            avg_costs.append(statistics.mean([r['estimated_cost_usd'] for r in cat_results]))
            avg_times.append(statistics.mean([r['total_time_s'] for r in cat_results]))
            avg_transfer.append(statistics.mean([r['total_transfer_bytes'] / 1024 for r in cat_results]))
        else:
            avg_costs.append(0)
            avg_times.append(0)
            avg_transfer.append(0)

    x = np.arange(len(categories))
    width = 0.6

    #graph 1: cost
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_costs, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Cost (USD)', fontsize=12, fontweight='bold')
    ax.set_title('AWS SageMaker (FLAN-T5) - Cost by Input Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, avg_costs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(avg_costs)*0.02,
                f'${value:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("graphs/01_cloud_cost.png", dpi=300, bbox_inches='tight')
    plt.close()

    #graph 2: time
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_times, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Response Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('AWS SageMaker (FLAN-T5) - Response Time by Input Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, avg_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(avg_times)*0.02,
                f'{value:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("graphs/02_cloud_response_time.png", dpi=300, bbox_inches='tight')
    plt.close()

    #graph 3: data transfer
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, avg_transfer, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Input Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Data Transfer (KB)', fontsize=12, fontweight='bold')
    ax.set_title('AWS SageMaker (FLAN-T5) - Data Transfer by Input Size', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, value in zip(bars, avg_transfer):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + max(avg_transfer)*0.02,
                f'{value:.1f} KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("graphs/03_cloud_data_transfer.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():

    
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
    
    existing_files = [f for f in input_files if os.path.exists(f)]

    
    results = {}
    total_cost = 0
    failed_count = 0
    
    #measure baseline
    baseline = measure_baseline_overhead(CLOUD_CONFIG["endpoint_name"], samples=5)

    for i, file_name in enumerate(existing_files, 1):
        
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            continue

        size_category, word_count = get_text_size_category(file_name, text)
        print(f"Category: {size_category} | Words: {word_count}")
        
        summary, metrics = summarize_text(text, CLOUD_CONFIG)
        
        if not metrics.get("success"):
            print(f"Failed: {metrics.get('error')}")
            failed_count += 1
            continue
        
        summary_word_count = len(summary.split()) if summary else 0
        
        results[file_name] = {
            "size_category": size_category,
            "word_count": word_count,
            "char_count": len(text),
            "summary_word_count": summary_word_count,
            "summary_char_count": len(summary) if summary else 0,
            "compression_ratio": word_count / summary_word_count if summary_word_count > 0 else 0,
            "estimated_input_tokens": token_count(text),
            "estimated_output_tokens": token_count(summary) if summary else 0,
            "baseline_overhead_ms": baseline['avg_ms'] if baseline else 0,
            **metrics
        }
        
        total_cost += metrics['estimated_cost_usd']
      
        print(f"Summary: {summary_word_count} words")


    if results:
        short_results = [r for r in results.values() if r['size_category'] == 'Short']
        medium_results = [r for r in results.values() if r['size_category'] == 'Medium']
        large_results = [r for r in results.values() if r['size_category'] == 'Large']
        
        os.makedirs("results", exist_ok=True)
        output_data = {
            "config": CLOUD_CONFIG,
            "baseline_overhead": baseline,
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_files": len(existing_files),
                "successful": len(results),
                "failed": failed_count
            },
            "results": results,
            "summary": {
                "total_cost_usd": total_cost,
                "avg_cost_per_file_usd": total_cost / len(results),
                "short": {
                    "count": len(short_results),
                    "avg_time_s": statistics.mean([r['total_time_s'] for r in short_results]) if short_results else 0,
                    "avg_cost_usd": statistics.mean([r['estimated_cost_usd'] for r in short_results]) if short_results else 0,
                },
                "medium": {
                    "count": len(medium_results),
                    "avg_time_s": statistics.mean([r['total_time_s'] for r in medium_results]) if medium_results else 0,
                    "avg_cost_usd": statistics.mean([r['estimated_cost_usd'] for r in medium_results]) if medium_results else 0,
                },
                "large": {
                    "count": len(large_results),
                    "avg_time_s": statistics.mean([r['total_time_s'] for r in large_results]) if large_results else 0,
                    "avg_cost_usd": statistics.mean([r['estimated_cost_usd'] for r in large_results]) if large_results else 0,
                }
            }
        }
        
        with open("results/cloud_results.json", "w") as f:
            json.dump(output_data, f, indent=2)
        


if __name__ == "__main__":
    main()

