import os
import ultralytics
from ultralytics import YOLO

ultralytics.checks()
dir = 'datasets\\'

# model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8m.pt')  # load a pretrained model (recommended for training)

results = model.train(data="config.yaml", epochs=150) # train the model
results = model.val() # evaluate model performance on the validation set
results = model([
    dir+'test\\images\\AUTOMOTIVE OIL\\1.jpg', 
    dir+'test\\images\\AUTOMOTIVE OIL\\2.jpg', 
    dir+'test\\images\\AUTOMOTIVE OIL\\3.jpg',
    ],conf=0.8)  # predict on an image

# Process results list
for i, result in enumerate(results):
    boxes = result.boxes  # Boxes object for bounding box outputs
    # masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    
    # Construa o caminho completo para o arquivo
    save_path = os.path.join(dir+'results', f'result_v4_{i}.jpg')
    
    # Salve a imagem no caminho especificado
    result.save(filename=save_path)