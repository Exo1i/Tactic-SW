from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")  # Load YOLOv8 model

    model.train(data="/home/ziad/Projects/Target-Shooting/datasets/balloon2.v2i.yolov8/data.yaml",  
                epochs=50,  
                batch=8,  
                imgsz=640,  
                device=0,
                workers=0)
