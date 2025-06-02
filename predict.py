# runs prediction on image
import argparse
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

categories = {
    0: "Empty",
    1: "Tweezers",
    2: "Needle_driver"
}

colors = {
    0: (255, 0, 0),   # empty is red
    1: (0, 255, 0),   # tweezers is green
    2: (0, 0, 255)    # blue is needle driver
}

def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{categories.get(cls, 'Unknown')} {conf:.2f}"
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

def main(image_path, model_path, output_path):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.5)
    image = cv2.imread(image_path)
    image = draw_boxes(image, results)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis("off")
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    # plt.savefig(output_path)
    plt.show()

# How to run it in your terminal
# python predict.py /path/image.jpg(input) path/name.jpg(output) --model runs/detect/best.pt
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("output_path", help="Path to output image")
    parser.add_argument("--model", default="trained_models/base_model2.pt", help="Path to trained YOLO model")
    args = parser.parse_args()
    main(args.image_path, args.model, args.output_path)