# runs predictions on a video 
import argparse
import cv2
from ultralytics import YOLO

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


def draw_boxes(frame, results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{categories.get(cls, 'Unknown')} {conf:.2f}"
            color = colors.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main(video_path, output_path, model_path):
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.5, verbose=False)
        frame = draw_boxes(frame, results)
        # cv2.imshow("Detection", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        out.write(frame)

    cap.release()
    out.release()
    # cv2.destroyAllWindows()
    print(f"Saved annotated video to: {output_path}")

# How to run it in your terminal
# python video.py --model trained_models/base_model2.pt (or another model) /path/video.mp4(imput) video_with_boxes.mp4 (custom_name) 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("output_path", help="Path to save output video")
    parser.add_argument("--model", default="runs/detect/base_model2/weights/best.pt", help="Path to trained YOLO model")
    args = parser.parse_args()
    main(args.video_path, args.output_path, args.model)
