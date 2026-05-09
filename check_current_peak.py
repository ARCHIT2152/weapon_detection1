import pandas as pd
import os

try:
    
    csv_path = "runs/detect/weapon_detection_model7/results.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}. The model might be saving to a different folder.")
        exit()

    df = pd.read_csv(csv_path)
    
    
    df.columns = df.columns.str.strip()
    
    
    epochs = df['epoch'].tolist()
    map_scores = df['metrics/mAP50-95(B)'].tolist()
    
    
    peak_score = max(map_scores)
    peak_epoch = map_scores.index(peak_score) + 1 
    current_epoch = len(epochs)
    
    epochs_since_peak = current_epoch - peak_epoch
    
    print("\n" + "="*50)
    print(" LIVE TRAINING PEAK ANALYSIS ")
    print("="*50)
    print(f"Current Epoch: {current_epoch} / 150")
    print(f"Highest mAP Score Achieved: {peak_score:.4f}")
    print(f"Epoch that achieved this peak: Epoch {peak_epoch}")
    print("-" * 50)
    
    if epochs_since_peak == 0:
        print(" Your AI is currently at its absolute smartest point right now! It just peaked.")
    else:
        print(f" It has been {epochs_since_peak} epochs since the AI last peaked.")
        print(f" If this number reaches 25, the patience trigger will abort the training.")
        
    print("="*50 + "\n")

except Exception as e:
    print(f"Could not read the results: {e}")
