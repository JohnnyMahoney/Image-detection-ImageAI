from imageai.Detection import ObjectDetection
from imageai.Detection import VideoObjectDetection
import os

modelPath = "./models/yolo-tiny.h5"
inputPath = "./input/test_im.jpg"
outputPath = "./output/newImage.jpg"

detector = ObjectDetection()
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(modelPath)
detector.loadModel()

detection = detector.detectObjectsFromImage(
    input_image = inputPath,
    output_image_path = outputPath
)
#
# videoDetector = VideoObjectDetection()
# videoDetector.setModelTypeAsTinyYOLOv3()
# videoDetector.setModelPath(modelPath)
# videoDetector.loadModel()
# detection = videoDetector.detectObjectsFromVideo(
# input_file_path='input/road.mp4',
# output_file_path='output/carsDetected',
# frames_per_second=10,
# minimum_percentage_probability=30
# )