# Copyright 2025 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from utils.face_utils import face_alignment


class TFLiteFaceEngine(object):
    """
    Face recognition model using TensorFlow Lite for inference and OpenCV for image preprocessing,
    utilizing an external face alignment function.
    """

    def __init__(self, model_path: str):
        """
        Initializes the TFLiteFaceEngine model for inference.

        Args:
            model_path (str): Path to the TFLite model file.
        """
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        # Get input size from model
        self.input_size = tuple(self.input_details['shape'][1:3][::-1])
        self.input_mean = 127.5
        self.input_std = 127.5


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image: resize, normalize, and change data layout.

        Args:
            image (np.ndarray): Input image in BGR format.

        Returns:
            np.ndarray: Preprocessed image as a NumPy array ready for inference.
        """
        # 1. Resize to model's input size
        image = cv2.resize(image, self.input_size)
        
        # 2. Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. Normalize the image
        image = (image.astype(np.float32) - self.input_mean) / self.input_std
        
        # 4. Add a batch dimension and ensure it's float32
        # The model expects input shape (1, 112, 112, 3)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        return image

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Extracts face embedding from an aligned image.

        Args:
            image (np.ndarray): Input face image (BGR format).
            landmarks (np.ndarray): Facial landmarks (5 points for alignment).

        Returns:
            np.ndarray: 512-dimensional face embedding.
        """
        aligned_face = face_alignment(image, landmarks)  # Use your function for alignment
        blob = self.preprocess(aligned_face)  # Convert to blob
        
        self.interpreter.set_tensor(self.input_details['index'], blob)
        self.interpreter.invoke()
        embedding = self.interpreter.get_tensor(self.output_details['index'])
        
        return embedding  # Return the feature vector
