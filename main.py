# Importing all the required libraries
import pandas as pd
import numpy as np
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import librosa.display as lplt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# Reading the csv file
df = pd.read_csv("Data/features_3_sec.csv")
df.head()

# Loading a sample audio from the dataset
audio = "Data/genres_original/reggae/reggae.00010.wav"
data, sr = librosa.load(audio)
print(type(data), type(sr))

# Initializing sample rate to 45600 we obtain the signal value array
librosa.load(audio, sr=45600)

# Taking Short-time Fourier transform of the signal
y = librosa.stft(data)
S_db = librosa.amplitude_to_db(np.abs(y), ref=np.max)

# Wave form of the audio
plt.figure(figsize=(7, 4))
librosa.display.waveshow(data, color="#2B4F72", alpha=0.5)
plt.show()

# Spectrogram of the audio
stft = librosa.stft(data)
stft_db = librosa.amplitude_to_db(abs(stft))
plt.figure(figsize=(7, 6))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.show()

# Data Pre Processing #

# Spectral Roll Off
spectral_rolloff=librosa.feature.spectral_rolloff(y=data, sr=sr)[0]
plt.figure(figsize=(7, 6))
librosa.display.waveshow(data, sr=sr, alpha=0.4, color="#2B4F72")

# Chroma Feature
chroma = librosa.feature.chroma_stft(y=data, sr=sr)
plt.figure(figsize=(7, 4))
lplt.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", cmap="BuPu")
plt.colorbar()
plt.title("Chroma Features")
plt.show()

# Zero Crossing Rate #
start = 1000
end = 1200
plt.figure(figsize=(12, 4))
plt.plot(data[start:end], color="#2B4F72")
plt.show()

# Printing the number of times signal crosses the x-axis
zero_cross_rate = librosa.zero_crossings(data[start:end], pad=False)
print("The number of zero_crossings are :", sum(zero_cross_rate))

# Finding misssing values
# Find all columns with any NA values
print("Columns containing missing values", list(df.columns[df.isnull().any()]))

# Label Encoding - encod the categorical classes with numerical integer values for training

# Blues - 0
# Classical - 1
# Country - 2
# Disco - 3
# Hip-hop - 4
# Jazz - 5
# Metal - 6
# Pop - 7
# Reggae - 8
# Rock - 9

class_encod = df.iloc[:, -1]
converter = LabelEncoder()
y = converter.fit_transform(class_encod)

# features
print(df.iloc[:, :-1])

# Drop the column filename as it is no longer required for training
df = df.drop(labels="filename", axis=1)

# scaling
fit = StandardScaler()
X = fit.fit_transform(np.array(df.iloc[:, :-1], dtype=float))

# splitting 70% data into training set and the remaining 30% to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# test data size
len(y_test)

# size of training data
print(len(y_train))

# KNN

print('KNN')
# Applying K nearest Neighbour algorithm to predict the results

# Create an instance of the KNeighborsClassifier with 3 neighbors.
clf1 = KNeighborsClassifier(n_neighbors=3)
# Train the classifier using the training data (X_train) and labels (y_train).
clf1.fit(X_train, y_train)
# Predict the labels for the test data (X_test) using the trained classifier.
y_prediction = clf1.predict(X_test)
# Print the accuracy of the classifier on the training set.
print("Training set score: {:.3f}".format(clf1.score(X_train, y_train)))
# Print the accuracy of the classifier on the test set.
print("Test set score: {:.3f}".format(clf1.score(X_test, y_test)))
# Calculate the confusion matrix comparing the true labels (y_test) and predicted labels (y_prediction).
cf_matrix = confusion_matrix(y_test, y_prediction)
# Set the size of the plot using seaborn's set function.
sns.set(rc={'figure.figsize': (8, 3)})
# Create a heatmap of the confusion matrix with annotations.
sns.heatmap(cf_matrix, annot=True)
# Print a classification report that includes precision, recall, f1-score for each class.
print(classification_report(y_test, y_prediction))

# SVM

print('SVM')
# Applying Support Vector Machines to predict the results

# Create an instance of the SVC with Radial Basis Function (RBF) kernel and a degree of 8.
svclassifier = SVC(kernel='rbf', degree=8)
# Train the classifier using the training data (X_train) and labels (y_train).
svclassifier.fit(X_train, y_train)
# Print the accuracy of the classifier on the training set.
print("Training set score: {:.3f}".format(svclassifier.score(X_train, y_train)))
# Print the accuracy of the classifier on the test set.
print("Test set score: {:.3f}".format(svclassifier.score(X_test, y_test)))
# Predict the labels for the test data (X_test) using the trained classifier.
y_prediction = svclassifier.predict(X_test)
# Calculate the confusion matrix comparing the true labels (y_test) and predicted labels (y_prediction).
cf_matrix3 = confusion_matrix(y_test, y_prediction)
# Set the size of the plot using seaborn's set function.
sns.set(rc={'figure.figsize': (9, 4)})
# Create a heatmap of the confusion matrix with annotations.
sns.heatmap(cf_matrix3, annot=True)
# Print a classification report that includes precision, recall, f1-score for each class.
print(classification_report(y_test, y_prediction))

