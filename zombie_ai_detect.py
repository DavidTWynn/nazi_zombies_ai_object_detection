import cv2
import ultralytics
from ultralytics.yolo.engine.results import Results

model = ultralytics.YOLO("zombies_1.pt")


def main() -> None:
    for predict in model.predict(source=0, show=True, stream=True):
        break

    print(predict.boxes)


if __name__ == "__main__":
    main()
