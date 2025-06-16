

import argparse
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=30, help='Training epochs')
    p.add_argument('--imgsz',  type=int, default=640, help='Image size')
    p.add_argument('--model',  type=str, default='yolov8n.pt', help='Base model')
    p.add_argument('--batch',  type=int, default=16, help='Batch size')
    p.add_argument('--name',    type=str, default='train')
    p.add_argument('--exist_ok', action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data='dataset.yaml',
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project='runs',
        name=args.name,        
        exist_ok=args.exist_ok 
    )
if __name__ == '__main__':
    main()
