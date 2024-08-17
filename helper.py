import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import confusion_matrix

class Helper:
    
    def plot_train_and_val_curves(self, history):
    
        # collect data for result
        result = {}
        result["epoch"] = history.epoch
        result["accuracy"] = history.history["accuracy"]
        result["val_accuracy"] = history.history["val_accuracy"]
        result["loss"] = history.history["loss"]
        result["val_loss"] = history.history["val_loss"]

        # create a data frame
        result = pd.DataFrame(result)

        # get some values
        acc_train = result.iloc[-1]["accuracy"]
        acc_valid = result.iloc[-1]["val_accuracy"]
        loss = result.iloc[-1]["loss"]
        loss_valid = result.iloc[-1]["val_loss"]

        plt.figure(figsize=(10,4), dpi=300)
        plt.subplot(121)
        plt.plot(result.epoch, result.accuracy, label=f"Train acc {acc_train:.2f}")
        plt.plot(result.epoch, result.val_accuracy, label=f"Validation acc {acc_valid:.2f}")
        plt.ylabel("Acc")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(122)
        plt.plot(result.epoch, result.loss, label=f"Train loss {loss:.2f}")
        plt.plot(result.epoch, result.val_loss, label=f"Validation loss {loss_valid:.2f}")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
    

    
    
    def plot_confusion_matrix(self, model, valid_gen):
        predicted = np.argmax(model.predict(valid_gen), axis=1)
        cm = pd.DataFrame(
            confusion_matrix(valid_gen.labels, predicted),
            columns=[f for f in valid_gen.class_indices.keys()],
            index=[f for f in valid_gen.class_indices.keys()]
            )
        sns.heatmap(cm, annot=True, cmap="crest", linewidths=0.5, cbar=False, fmt="d")

    def plot_roc_curve(self, model, valid_gen):
        y_true = valid_gen.classes
        class_labels = list(valid_gen.class_indices.keys())

        # One-hot encode the labels
        y_true = tf.keras.utils.to_categorical(y_true, num_classes=len(class_labels))

        # Generate predictions
        y_pred = model.predict(valid_gen)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(class_labels)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        for i in range(len(class_labels)):
            plt.plot(fpr[i], tpr[i],
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(class_labels[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
