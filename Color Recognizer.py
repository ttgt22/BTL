import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load the image to detect colors from
img = cv2.imread('color_image.png')

# Define the names for the columns
index = ["color", "color_name", "hex", "R", "G", "B"]

# Read the csv according to the names given above
df = pd.read_csv('colors.csv', names=index, header=None)

# Prepare the dataset from the image
data = []
for y in range(img.shape[0]):  # Duyệt qua chiều cao
    for x in range(img.shape[1]):  # Duyệt qua chiều rộng
        b, g, r = img[y, x]  # Lấy giá trị BGR
        # Tìm màu gần nhất trong df
        d = abs(r - df["R"]) + abs(g - df["G"]) + abs(b - df["B"])
        min_index = d.idxmin()  # Chỉ số của màu gần nhất
        data.append([df.loc[min_index, "color_name"], r, g, b])

# Chuyển đổi dữ liệu thành DataFrame
data_df = pd.DataFrame(data, columns=["color_name", "R", "G", "B"])

# Prepare the dataset
X = data_df[["R", "G", "B"]]
y = data_df["color_name"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
svm_classifier = SVC()
knn_classifier = KNeighborsClassifier()
decision_tree_classifier = DecisionTreeClassifier()

# Train the classifiers
svm_classifier.fit(X_train, y_train)
knn_classifier.fit(X_train, y_train)
decision_tree_classifier.fit(X_train, y_train)

# Make predictions
svm_predictions = svm_classifier.predict(X_test)
knn_predictions = knn_classifier.predict(X_test)
tree_predictions = decision_tree_classifier.predict(X_test)

# Print classification reports for each classifier
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions, zero_division=0))

print("KNN Classification Report:")
print(classification_report(y_test, knn_predictions, zero_division=0))

print("Decision Tree Classification Report:")
print(classification_report(y_test, tree_predictions, zero_division=0))

# Calculate and print accuracy for each model
print("SVM Accuracy:", accuracy_score(y_test, svm_predictions))
print("KNN Accuracy:", accuracy_score(y_test, knn_predictions))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_predictions))

# Some global variables
clicked = False
r = g = b = xpos = ypos = 0

# Color recognition function
def color_recognizer(R, G, B):
    rgb = np.array([[R, G, B]])
    cname = svm_classifier.predict(rgb)[0]  # Using SVM for real-time color recognition
    return cname

# Mouse click function
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, xpos, ypos, clicked
        clicked = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

    # Our Application Window
    cv2.namedWindow('Color Recognizer', flags=cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Color Recognizer', mouse_click)

    while True:
        cv2.imshow('Color Recognizer', img)
        if clicked:
            cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
            text = color_recognizer(R=r, G=g, B=b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            if r + g + b >= 600:
                cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

            clicked = False

        # Close the window if 'esc' key is pressed
        if cv2.waitKey(20) & 0xFF == 27:
            break

    # Release all the windows
    cv2.destroyAllWindows()