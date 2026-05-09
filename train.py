from ultralytics import YOLO

def main():
    
    model = YOLO("yolov8m.pt") 

    print("Starting training on weapon detection dataset...")
    
    
    results = model.train(
        data="weapon detection.v1i.yolov8 (1)/data.yaml",
        epochs=150,                  
        imgsz=640,                  
        batch=16,                   
        name="weapon_detection_model",
        device="0",               
        patience=25,                
        save=True,                 
        cache  = True,
        workers = 4

    )
    
    print("\nTraining complete! Results and weights saved to 'runs/detect/weapon_detection_model'.")
    print("Your best model weights are located at: 'runs/detect/weapon_detection_model/weights/best.pt'")

if __name__ == '__main__':
    main()
