import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

API_KEY = "5dqGi9KXXpknt8L3gxWb"

def run_end_to_end():
    print("🚀 Starting End-to-End Railway Model Setup...")
    
    # 1. Initialize Roboflow and fetch dataset
    rf = Roboflow(api_key=API_KEY)
    
    # Using a known railway instance segmentation project
    # Project: railway-instancesegmentation by saen
    print("📥 Connecting to Roboflow Universe...")
    try:
        project = rf.workspace("saen").project("railway-instancesegmentation")
        version = project.version(8)
        
        print(f"Downloading dataset for version {version.version}...")
        dataset = version.download("yolov8")
        print(f"✅ Dataset downloaded at {dataset.location}")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return

    # 2. Train a lightweight YOLOv8n model for just 1 epoch to generate the .pt file
    print("\n🚂 Starting YOLOv8 Training...")
    # Make sure we use the yolov8n base model
    model = YOLO("yolov8n.pt") 
    
    data_yaml_path = os.path.join(dataset.location, "data.yaml")
    
    if not os.path.exists(data_yaml_path):
        print(f"❌ data.yaml not found at {data_yaml_path}")
        return
        
    print(f"Training on {data_yaml_path}...")
    try:
        # Train for 1 epoch just to bake the weights for the local environment
        results = model.train(
            data=data_yaml_path,
            epochs=1,  # Kept low for speed; user can increase later for better accuracy
            imgsz=640,
            batch=4,   # Low batch size to prevent memory issues
            project="runs/detect",
            name="railways_train"
        )
        print("✅ Training completed.")
        
        # 3. Move the best.pt to backend/models/railways.pt
        best_pt_path = os.path.join("runs/detect/railways_train", "weights", "best.pt")
        target_path = os.path.join(os.getcwd(), "models", "railways.pt")
        
        if os.path.exists(best_pt_path):
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(best_pt_path, target_path)
            print(f"🎉 Model successfully generated and copied to {target_path}!")
        else:
            print(f"❌ Could not find {best_pt_path} after training.")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")

if __name__ == "__main__":
    run_end_to_end()
