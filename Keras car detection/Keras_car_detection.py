import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from scipy import ndimage


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """
    Filter YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence
    boxes
    box_class_probs
    threshold --  if [ highest class probability score < threshold], then filter the box

    Returns:
    scores -- probability scores for selected boxes
    boxes -- (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- index of the class detected by the selected boxes
    """

    # Compute box scores
    box_scores = box_confidence * box_class_probs

    # Find the box_classes and their score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)

    # Filtering mask based on "box_class_scores" and "threshold".
    filtering_mask = (box_class_scores >= threshold)

    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def iou(box1, box2):
    """
    Intersection over union (IoU) between box1 and box2

    Arguments:
    box1
    box2
    """

    # Calculate coordinates of the intersection of box1 and box2.
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    # Calculate the union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU
    iou = inter_area / union_area

    return iou


def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Non-max suppression (NMS) to set of boxes

    Arguments:
    scores
    boxes
    classes
    max_boxes -- maximum number of predicted boxes
    iou_threshold --  IoU threshold used for NMS filtering

    Returns:
    scores -- predicted score for each box
    boxes -- predicted box coordinates
    classes -- predicted class for each box
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold)

    # Select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def yolo_eval(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    """
    Convert the output of YOLO encoding to predicted boxes, scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model contains:
                    box_confidence
                    box_xy
                    box_wh
                    box_class_probs
    image_shape -- input shape
    max_boxes -- maximum number of predicted boxes
    score_threshold -- if [ highest class probability score < threshold], then filter
    iou_threshold -- IoU threshold used for NMS filtering

    Returns:
    scores -- predicted score for each box
    boxes -- predicted box coordinates
    classes -- predicted class for each box
    """

    # YOLO outputs
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Filter a threshold of score_threshold
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)

    # Scale boxes back to original image shape. Non-max suppression.
    boxes = scale_boxes(boxes, image_shape)
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)

    return scores, boxes, classes


def predict(sess, image_file):
    """
    Predict boxes for "image_file"

    Arguments:
    sess -- Keras session
    image_file -- name of a test image stored in the "images" folder.

    Returns:
    out_scores -- scores of the predicted boxes
    out_boxes -- coordinates of the predicted boxes
    out_classes -- class index of the predicted boxes
    """

    image, image_data = preprocess_image("images/" + image_file, model_image_size=(416, 416))

    out_scores, out_boxes, out_classes = sess.run(yolo_eval(yolo_outputs), feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Predictions info
    # print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
    # Draw bounding boxes on the image file
    output_stats=draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))

    return out_scores, out_boxes, out_classes, output_image, output_stats


sess = K.get_session()

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
yolo_model = load_model("model_data/yolov2.h5")
#yolo_model.summary()

image_shape = (720., 1280.)
my_image="car_test.jpg" # Test image stored in the "images" folder
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
out_scores, out_boxes, out_classes , out_image, output_stats= predict(sess, my_image)

plt.title('Found {} boxes for {}'.format(len(out_boxes), my_image) + output_stats)
plt.imshow(out_image)
plt.show()