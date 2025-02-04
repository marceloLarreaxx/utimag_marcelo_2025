import tensorflow as tf
from keras import backend as K
from statistics import mean, mode, median, stdev
import numpy as np
import math

def iou_coef(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    intersection = K.sum(targets * inputs)
    total = K.sum(targets) + K.sum(inputs)
    union = total - intersection

    IoU = (intersection + smooth) / (union + smooth)
    return IoU.numpy().tolist()
    # intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    # union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
    # iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    # return iou

def dice_loss_v2(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    #y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return (1 - numerator / denominator).numpy().tolist()

def dice_coef(y_true, y_pred, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)

    intersection = K.sum(targets * inputs)
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice.numpy().tolist()
def precision_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    precision = np.mean(intersect/total_pixel_pred)
    return round(precision, 3).tolist()

def recall_score(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    recall = np.mean(intersect/total_pixel_truth)
    return round(recall, 3).tolist()
def accuracy_(groundtruth_mask, pred_mask):
    intersect = np.sum(pred_mask*groundtruth_mask)
    union = np.sum(pred_mask) + np.sum(groundtruth_mask) - intersect
    xor = np.sum(groundtruth_mask==pred_mask)
    acc = np.mean(xor/(union + xor - intersect))
    return round(acc, 3).tolist()

def print_statistic_data(dice_losses,iou_losses,recall, precision, accuracy, model='Binary_loss'):

    dice_min=min(dice_losses)
    dice_max=max(dice_losses)
    dice_mean=mean(dice_losses)
    dice_median=median(dice_losses)
    dice_moda=mode(dice_losses)
    dice_stdv=stdev(dice_losses)

    iou_min = min(iou_losses)
    iou_max = max(iou_losses)
    iou_mean = mean(iou_losses)
    iou_median = median(iou_losses)
    iou_moda = mode(iou_losses)
    iou_stdv = stdev(iou_losses)

    recall_min = min(recall)
    recall_max = max(recall)
    recall_mean = mean(recall)
    recall_median = median(recall)
    recall_moda = mode(recall)
    recall_stdv = stdev(recall)

    precision_min = min(precision)
    precision_max = max(precision)
    precision_mean = mean(precision)
    precision_median = median(precision)
    precision_moda = mode(precision)
    precision_stdv = stdev(precision)

    accuracy_min = min(accuracy)
    accuracy_max = max(accuracy)
    accuracy_mean = mean(accuracy)
    accuracy_median = median(accuracy)
    accuracy_moda = mode(accuracy)
    accuracy_stdv = stdev(accuracy)

    # cross_ent_min = min(binary_cross_entropy)
    # cross_ent_max = max(binary_cross_entropy)
    # cross_ent_mean = mean(binary_cross_entropy)
    # cross_ent_median = median(binary_cross_entropy)
    # cross_ent_moda = mode(binary_cross_entropy)
    # cross_ent_stdv = stdev(binary_cross_entropy)

    print('=====PARA EL MODELO '+ model +'=====')
    print('')
    print('Dice min:    '+ str(dice_min) + ' - IOU min:    '+ str(iou_min)+ ' - recall min:    '+ str(recall_min)+ ' - precision min:    '+ str(precision_min)+ ' - accuracy min:    '+ str(accuracy_min))#+ ' - BCE min:    '+ str(cross_ent_min))
    print('Dice max:    ' + str(dice_max) + ' - IOU max:    '+ str(iou_max)+ ' - recall max:    '+ str(recall_max)+ ' - precision max:    '+ str(precision_max)+ ' - accuracy max:    '+ str(accuracy_max))#+ ' - BCE max:    '+ str(cross_ent_max))
    print('Dice mean:   ' + str(dice_mean) + ' - IOU mean:    '+ str(iou_mean)+ ' - recall mean:    '+ str(recall_mean)+ ' - precision mean:    '+ str(precision_mean)+ ' - accuracy mean:    '+ str(accuracy_mean))#+ ' - BCE mean:    '+ str(cross_ent_mean))
    print('Dice median: ' + str(dice_median) + ' - IOU median:    '+ str(iou_median)+ ' - recall median:    '+ str(recall_median)+ ' - precision median:    '+ str(precision_median)+ ' - accuracy median:    '+ str(accuracy_median))#+ ' - BCE median:    '+ str(cross_ent_median))
    print('Dice moda:   ' + str(dice_moda) + ' - IOU moda:    '+ str(iou_moda)+ ' - recall moda:    '+ str(recall_moda)+ ' - precision moda:    '+ str(precision_moda)+ ' - accuracy moda:    '+ str(accuracy_moda))#+ ' - BCE moda:    '+ str(cross_ent_moda))
    print('Dice stdev:  ' + str(dice_stdv) + ' - IOU stdev:    '+ str(iou_stdv)+ ' - recall stdev:    '+ str(recall_stdv)+ ' - precision stdev:    '+ str(precision_stdv)+ ' - accuracy stdev:    '+ str(accuracy_stdv))#+ ' - BCE stdev:    '+ str(cross_ent_stdv))

def load_segmentation_losses(data_gt,data_test,model, umbral=0.5, model_label='Binary_loss'):
    dice_losses = []
    iou_losses = []
    recall = []
    precision = []
    accuracy = []

    n_test = data_test.shape[0]

    for i in range(n_test):
        input = data_test[i, :, :, :, 0]
        pred = np.float32(model(np.expand_dims(input, axis=0))>umbral)[0, :, :, :, 0]
        gt = data_gt[i, :, :,:,0]

        if math.isnan(dice_coef(gt, pred)):
            dice_losses.extend([1])
        else:
            dice_losses.extend([dice_coef(gt, pred)])
        if math.isnan(recall_score(gt, pred)):
            recall.extend([1])
        else:
            recall.extend([recall_score(gt, pred)])
        if math.isnan(precision_score(gt, pred)):
            precision.extend([1])
        else:
            precision.extend([precision_score(gt, pred)])
        if math.isnan(accuracy_(gt, pred)):
            accuracy.extend([1])
        else:
            accuracy.extend([accuracy_(gt, pred)])
        if math.isnan(iou_coef(gt, pred)):
            iou_losses.extend([1])
        else:
            iou_losses.extend([iou_coef(gt, pred)])

    print_statistic_data(dice_losses,iou_losses,recall, precision, accuracy, model_label)