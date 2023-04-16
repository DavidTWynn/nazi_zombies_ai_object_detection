"""Trying to get the speed settings understood for zombie detection
with a Yolov8 pretrained model."""

from time import perf_counter

import cv2
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from run_yolo import RunYolo
from window_capture import WindowCapture


def capture_and_show_window(window_name="Plutonium T4 Singleplayer (r3417)"):
    """Simple function to display the game window and fps for testing.
    
    Uses WindowCapture class to get screenshots from the pywin32 package.
    The WindowCapture class has a threaded option when calling the instance's
    .start() method. The screenshots are then stored to instance attribute
    .screenshot. You the screenshot can be taken in the same thread with
    .get_screenshot(). Good for testing speed.
    
    Args:
        window_name: screenshot window. Defaults to "Plutonium T4 Singleplayer (r3417)".
    """
    # Create WindowCapture object and start thread of capturing screenshots
    cap = WindowCapture(window_name)
    # cap.start()

    while True:
        start = perf_counter()
        # Only show available screenshots
        cv2.imshow("test", cap.get_screenshot())
        cv2.waitKey(1)

        print(f"FPS: {1.0 / (perf_counter() - start)}")

def run_yolo_model(window_name="Plutonium T4 Singleplayer (r3417)"):
    # Create WindowCapture object and start thread of capturing screenshots
    cap = WindowCapture(window_name)
    # cap.start()
    # model = RunYolo("zombies_1.pt", 0.35)
    model = YOLO("zombies_1.pt")
    # model.start()
    # cap.start()

    while True:
        start = perf_counter()
        results = model.predict(source=cap.get_screenshot(), verbose=False)
        # larger_image = cv2.resize(larger_image, (1280, 720))
        # cv2.imshow("Result3", larger_image)
        # cv2.waitKey(1)

        # results = model.get_prediction()

        # time.sleep(1)
        # pyautogui.moveRel(1470, 0)

        if results:
            
            result, zombie_x, zombie_y, annotated_frame = result_parser(results)


            if zombie_x and zombie_y:
                
                #zombie_x_real = zombie_x + cap.offset_x
                #zombie_y_real = zombie_y + cap.offset_y
                x, y = cap.get_screen_position((zombie_x, zombie_y))
                #mid_y = int(cap.h / 2) + cap.offset_y
                #mid_x = int(cap.w / 2) + cap.offset_x
                cv2.drawMarker(annotated_frame, (x, y), color=(255, 255, 255), thickness=2)
            
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            cv2.waitKey(1)
            print(f"FPS: {1.0 / (perf_counter() - start):.2f}")


def result_parser(results):
    zombie_x = None
    zombie_y = None
    results: Results
    annotated_frame = results[0].plot()
    class_names = results[0].names

    # Setup counter for classes found per image
    class_count = {}
    for class_name in class_names.values():
        class_count[class_name] = 0

    # Store zombie data
    zombies = []

    # parse a single image lists of results
    for box in results[0].boxes:
        box_class_num = box.cls.cpu().numpy().astype(int)[0]
        box_class_name = class_names[box_class_num]

        # update counter
        class_count[box_class_name] += 1

        # get zombie cords
        if box_class_name == "zombie":
            # Parse out the xyxy of the box detection of the zombie
            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]
            # Find the center point of the box for clicking on
            zombie_x, zombie_y = (int((x1 + x2) / 2), int((y2 + y1) / 2))
            # y1 is the top of the box
            # Find the difference of top and center and use it as a percentage
            percentage_near_top = 0.2
            difference = zombie_y - y1
            zombie_y = int(y1 + (percentage_near_top * difference))

            zombie_length = x2 - x1

            zombie_data = {
                "zombie_x": zombie_x,
                "zombie_y": zombie_y,
                "zombie_length": zombie_length,
                "box": (x1, y1, x2, y2)
            }

            zombies.append(zombie_data)

    if zombies:
        max_length = max(zombies, key=lambda x: x['zombie_length'])

        cv2.drawMarker(annotated_frame, (max_length['zombie_x'], max_length['zombie_y']), color=(255, 0, 255), thickness=2)
        return class_count, max_length['zombie_x'], max_length['zombie_y'], annotated_frame
        
    # Cross crosshair on zombie

    # print(f"FPS: {1.0 / (perf_counter() - start)}")
    
    return class_count, None, None, annotated_frame
    

def main() -> None:
    run_yolo_model()

if __name__ == "__main__":
    main()
