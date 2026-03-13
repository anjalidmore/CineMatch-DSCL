from src.utils.data_loader import load_clean_data
from src.algos.content_engine import train_content_model
# from src.algos.collab_engine import train_collab_model # Your friend's part

def main():
    print("📥 Loading real movie data...")
    df = load_clean_data()
    
    print("🧠 Training Content-Based Engine (Cosine Similarity)...")
    # This creates the .pkl file in your /models folder
    train_content_model(df)
    
    print("✨ Training Complete! Check your /models folder.")

if __name__ == "__main__":
    main()