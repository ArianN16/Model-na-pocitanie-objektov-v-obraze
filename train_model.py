from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')

model.train(data='config.yaml', epochs= 150, imgsz=640, batch=8, device=0)