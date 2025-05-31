from ultralytics import YOLO

model = YOLO("yolo12n.yaml") 
model = YOLO("yolo12n.pt")  
model = YOLO("yolo12n.yaml").load("yolo12n.pt")  

results = model.train(data="data.yaml", epochs=100, imgsz=640)
model.export(format="onnx", dynamic=True)