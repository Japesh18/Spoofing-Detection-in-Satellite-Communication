import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,BatchNormalization
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
extracted_features=pd.read_csv(r'D:\ISRO\Spoofing Detection\Deep Learning Models\Final Model\Model-2\Features.csv')
features=["Mean Of Amplitude","Standard Deviation Of Amplitude","Kurtosis Of Amplitude","Skewness Of Amplitude","Mean Of Phase","Standard Deviation Of Phase"]
x=extracted_features[features].values
y=extracted_features["Label"].values
scaler=StandardScaler()
x_train,x_intermediate,y_train,y_intermediate=train_test_split(x,y,test_size=0.3,stratify=y,random_state=42)
x_validation,x_test,y_validation,y_test=train_test_split(x_intermediate,y_intermediate,test_size=0.5,stratify=y_intermediate,random_state=42)
x_train=scaler.fit_transform(x_train)
x_validation=scaler.transform(x_validation)
x_test=scaler.transform(x_test)
class_weights=compute_class_weight(class_weight="balanced",classes=np.unique(y_train),y=y_train)
class_weight_dictionary={0:class_weights[0],1:class_weights[1]}
model=Sequential([
    Dense(64,activation="relu",input_shape=(x_train.shape[1],)),
    BatchNormalization(),
    Dense(32,activation="relu"),
    BatchNormalization(),
    Dense(16,activation="relu"),
    BatchNormalization(),
    Dense(8,activation="relu"),
    BatchNormalization(),
    Dense(4,activation="relu"),
    BatchNormalization(),
    Dense(2,activation="relu"),
    BatchNormalization(),
    Dense(1,activation="sigmoid")
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),loss="binary_crossentropy",metrics=["accuracy"])
callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)]
history=model.fit(x_train,y_train,validation_data=(x_validation,y_validation),epochs=80,callbacks=callbacks,class_weight=class_weight_dictionary,verbose=1,shuffle=False)
test_loss,test_accuracy=model.evaluate(x_test,y_test,verbose=0)
print(f"Test Set's Accuracy: {test_accuracy*100:.2f}%")
y_predicted_probability=model.predict(x_test)
y_predicted=(y_predicted_probability>0.3484).astype(int).reshape(-1)
print("Classification Report:")
print(classification_report(y_test,y_predicted))
ConfusionMatrix=confusion_matrix(y_test,y_predicted)
confusion_matrix_percentage=ConfusionMatrix/ConfusionMatrix.sum(axis=1,keepdims=True)*100
labels_percentage=np.array([f"True Negative: {confusion_matrix_percentage[0,0]:.2f}%",f"False Positive: {confusion_matrix_percentage[0,1]:.2f}%",f"False Negative: {confusion_matrix_percentage[1,0]:.2f}%",f"True Positive: {confusion_matrix_percentage[1,1]:.2f}%"]).reshape(2,2)
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix_percentage,annot=labels_percentage,fmt="",cmap="Blues",xticklabels=["Genuine","Spoofed"],yticklabels=["Genuine","Spoofed"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
print(confusion_matrix_percentage)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Validation")
plt.title("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("Loss")
plt.legend()
plt.show()