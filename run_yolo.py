from threading import Lock, Thread

import ultralytics


class RunYolo:

    

    def __init__(self, model, confidence=0.25):
        # create a thread lock object
        self.lock = Lock()

        # Make model once
        self.model = ultralytics.YOLO(model)

        self.image = None

        self.confidence = confidence

        self.prediction = None

    def make_prediction(self, image):
        with self.lock:
            self.prediction = self.model.predict(source=image, verbose=False, conf=self.confidence)
        

    def start(self):
        self.stopped = False
        t = Thread(target=self.run)
        t.start()

    def stop(self):
        self.stopped = True

    def set_image(self, image):
        with self.lock:
            self.image = image

    def get_image(self):
        image = self.image
        with self.lock:
            self.image = None
            return image

    def set_prediction(self, prediction):
        with self.lock:
            self.prediction = prediction

    def get_prediction(self):
        prediction = self.prediction
        with self.lock:
            self.prediction = None
            return prediction

    def run(self):
        while not self.stopped:
            if self.image is not None:
                image = self.get_image()
                # get an updated image of the game
                self.make_prediction(image)
