import os
import pandas as pd
import numpy as np
import cv2
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_images_from_dataset(dataset_path):
    rgb = []
    colors = []

    for dirname, _, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.cvtColor(cv2.imread(os.path.join(dirname, filename)), cv2.COLOR_BGR2RGB)
                rgb.append(image[0][0])  # Lấy giá trị màu ở góc trái
                colors.append(dirname.split('/')[-1])  # Lấy tên thư mục
    return rgb, colors


def create_dataframe(rgb, colors):
    rgb_df = pd.DataFrame(rgb, columns=['red', 'green', 'blue'])
    color_df = pd.DataFrame({'color': colors})
    return rgb_df.join(color_df)


def plot_rgb_scatter(df):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = np.array(df[['red', 'green', 'blue']]) / 255
    ax.scatter(df['red'], df['green'], df['blue'], c=colors)
    ax.set(title='Scatter RGB 3D', xlabel='Red', ylabel='Green', zlabel='Blue')
    plt.show()


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def plot_metrics(metrics, values):
    plt.figure(figsize=(8, 6))
    plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
    plt.ylim(0, 1)
    plt.title('Model Evaluation Metrics')
    plt.ylabel('Score')
    plt.show()


def main():
    # Đọc hình ảnh để kiểm tra
    test_image_path = 'training_dataset/red/red19.png'
    test_image = cv2.cvtColor(cv2.imread(test_image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(test_image)
    plt.show()

    # Tải dữ liệu
    rgb, colors = load_images_from_dataset('training_dataset')

    # Tạo DataFrame
    df = create_dataframe(rgb, colors)

    # Kiểm tra và làm sạch dữ liệu
    print("Giá trị duy nhất trong cột color:", df['color'].unique())
    df = df[df['color'] != '']  # Loại bỏ các giá trị chuỗi rỗng
    print("Giá trị duy nhất sau khi xử lý:", df['color'].unique())

    # Vẽ biểu đồ RGB
    plot_rgb_scatter(df)

    # Phân chia dữ liệu
    X = df[['red', 'green', 'blue']]
    y = df['color']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Huấn luyện mô hình Decision Tree
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    y_pred = dtree.predict(X_test)

    # So sánh giá trị dự đoán với thực tế
    results_df = pd.DataFrame({'y_actual': y_test, 'y_predict': y_pred})
    print(results_df)

    # Đánh giá mô hình
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Vẽ biểu đồ đánh giá
    plot_metrics(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1])


if __name__ == "__main__":
    main()