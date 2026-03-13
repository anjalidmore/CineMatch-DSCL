import time
import psutil
import os
import random
import pandas as pd
import numpy as np

# This mimics the "Engines" you and your friend are building
def mock_content_engine(movie_title):
    # Simulating heavy matrix calculation
    time.sleep(random.uniform(0.01, 0.05)) 
    return ["Movie A", "Movie B", "Movie C"]

def mock_collab_engine(movie_title):
    # Simulating even heavier user-pivot table calculation
    time.sleep(random.uniform(0.08, 0.15)) 
    return ["Movie X", "Movie Y", "Movie Z"]

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

def run_benchmark():
    test_movies = ["The Dark Knight", "Inception", "Toy Story", "Interstellar", "Joker"]
    results = []

    print(f"🚀 Starting Benchmark... Initial Memory: {get_memory_usage():.2f} MB\n")

    for movie in test_movies:
        # Test Content-Based
        start_time = time.perf_counter()
        mock_content_engine(movie)
        content_time = time.perf_counter() - start_time
        
        # Test Collaborative
        start_time = time.perf_counter()
        mock_collab_engine(movie)
        collab_time = time.perf_counter() - start_time
        
        results.append({
            "Movie": movie,
            "Content_Time": content_time,
            "Collab_Time": collab_time,
            "Memory_MB": get_memory_usage()
        })

    # Summary Statistics
    df_results = pd.DataFrame(results)
    print("--- 📊 EFFICIENCY REPORT ---")
    print(f"Avg Content Latency: {df_results['Content_Time'].mean():.4f}s")
    print(f"Avg Collab Latency:  {df_results['Collab_Time'].mean():.4f}s")
    print(f"Peak Memory Usage:   {df_results['Memory_MB'].max():.2f} MB")
    
    # Save for your report
    df_results.to_csv("data/processed/benchmark_results.csv", index=False)
    print("\n✅ Results saved to data/processed/benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()