from ultralytics import YOLO

def main():
    print("🚀 Starting YOLOv8 Training for Railway Tracks...")
    
    # Load a pretrained YOLOv8 Nano Segmentation model
    model = YOLO('yolov8n-seg.pt')
    
    # Train the model on the Roboflow dataset
    # We use 20 epochs for a good balance of speed and accuracy
    # Metal Performance Shaders (mps) is used for Apple Silicon acceleration
    results = model.train(
        data='dataset/railways/data.yaml',
        epochs=20,
        imgsz=640,
        device='mps',
        batch=16,
        project='runs/train',
        name='railway_segmentation'
    )
    
    print("✅ Training complete! Best model saved to: runs/train/railway_segmentation/weights/best.pt")

if __name__ == '__main__':
    main()
