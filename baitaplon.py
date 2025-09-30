# Import các thư viện cần thiết
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Đọc dữ liệu train và test
try:
    # Thử đọc từ các đường dẫn phổ biến trên Kaggle
    possible_paths = [
        '/kaggle/input/mushroom-classification/mushrooms.csv',
        '/kaggle/input/mushroom-classification/train.csv',
        '/kaggle/input/uci-mushroom-dataset/mushrooms.csv',
        '/kaggle/input/test-csv/test.csv',
        '/kaggle/input/train-csv/train.csv'
    ]
    
    train_df = None
    test_df = None
    
    for path in possible_paths:
        try:
            if 'train' in path or 'mushroom' in path:
                train_df = pd.read_csv(path)
                print(f"Đọc train data thành công từ: {path}")
            elif 'test' in path:
                test_df = pd.read_csv(path)
                print(f"Đọc test data thành công từ: {path}")
        except:
            continue
    
    # Nếu không tìm thấy file test, tạo từ train data
    if train_df is not None and test_df is None:
        test_df = train_df.sample(100, random_state=42).copy()
        if 'class' in test_df.columns:
            test_df.drop('class', axis=1, inplace=True)
        print("Tạo test data từ train data")
    
except Exception as e:
    print(f"Lỗi khi đọc dữ liệu: {e}")
    # Fallback: tạo dữ liệu mẫu
    print("Tạo dữ liệu mẫu để demo...")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(20)]
    train_df = pd.DataFrame(X, columns=feature_names)
    train_df['class'] = ['e' if label == 0 else 'p' for label in y]
    test_df = pd.DataFrame(np.random.randn(100, 20), columns=feature_names)

print(f"Kích thước tập train: {train_df.shape}")
print(f"Kích thước tập test: {test_df.shape}")

# Kiểm tra xem có cột 'class' không
if 'class' not in train_df.columns:
    # Nếu không có cột class, tạo cột mục tiêu giả
    print("Không tìm thấy cột 'class', tạo cột mục tiêu giả...")
    train_df['class'] = np.random.choice(['e', 'p'], size=len(train_df))

# =============================================================================
# 1. KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU (EDA)
# =============================================================================

print("="*60)
print("1. KHÁM PHÁ VÀ PHÂN TÍCH DỮ LIỆU (EDA)")
print("="*60)

# Kiểm tra cấu trúc dữ liệu
print("1.1. Kiểm tra cấu trúc dữ liệu:")
print(f"Số lượng mẫu: {train_df.shape[0]}")
print(f"Số lượng đặc trưng: {train_df.shape[1]}")
print("\nThông tin về các cột:")
print(train_df.info())
print("\n5 dòng đầu của dataset:")
print(train_df.head())

# Phân tích đặc trưng mục tiêu
print("\n1.2. Phân tích đặc trưng mục tiêu (class):")
print("Phân phối của biến mục tiêu:")
print(train_df['class'].value_counts())

# Vẽ biểu đồ
plt.figure(figsize=(8, 6))
sns.countplot(data=train_df, x='class')
plt.title('Phân phối của biến mục tiêu (class)')
plt.xlabel('Loại nấm')
plt.ylabel('Số lượng')
plt.show()

# Phân tích các đặc trưng riêng lẻ
print("\n1.3. Phân tích các đặc trưng riêng lẻ:")

# Phân tích đặc trưng số (nếu có)
numeric_columns = train_df.select_dtypes(include=[np.number]).columns
if len(numeric_columns) > 0:
    print("Thống kê mô tả cho các đặc trưng số:")
    print(train_df[numeric_columns].describe())
else:
    print("Không có đặc trưng số trong dataset.")

# Phân tích đặc trưng phân loại
categorical_columns = train_df.select_dtypes(include=['object']).columns
categorical_columns = [col for col in categorical_columns if col != 'class']
print(f"\nCó {len(categorical_columns)} đặc trưng phân loại:")

# Vẽ biểu đồ cho một số đặc trưng phân loại quan trọng
important_features = ['cap-shape', 'cap-color', 'odor', 'gill-color', 'population', 'habitat']
available_features = [f for f in important_features if f in train_df.columns]

if available_features:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_features):
        if i < len(axes):
            value_counts = train_df[feature].value_counts()
            # Giới hạn số lượng categories để hiển thị đẹp
            if len(value_counts) > 10:
                top_10 = value_counts.head(10)
                axes[i].bar(range(len(top_10)), top_10.values)
                axes[i].set_xticks(range(len(top_10)))
                axes[i].set_xticklabels(top_10.index, rotation=45)
            else:
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45)
            axes[i].set_title(f'Phân phối của {feature}')
    
    # Ẩn các subplot không sử dụng
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    plt.show()
else:
    # Vẽ biểu đồ cho 6 đặc trưng phân loại đầu tiên
    if len(categorical_columns) > 0:
        features_to_plot = categorical_columns[:6]
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(features_to_plot):
            value_counts = train_df[feature].value_counts()
            # Giới hạn số lượng categories
            if len(value_counts) > 10:
                top_10 = value_counts.head(10)
                axes[i].bar(range(len(top_10)), top_10.values)
                axes[i].set_xticks(range(len(top_10)))
                axes[i].set_xticklabels(top_10.index, rotation=45)
            else:
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45)
            axes[i].set_title(f'Phân phối của {feature}')
        
        # Ẩn các subplot không sử dụng
        for i in range(len(features_to_plot), len(axes)):
            axes[i].set_visible(False)
            
        plt.tight_layout()
        plt.show()

# Kiểm tra dữ liệu thiếu
print("\n1.4. Kiểm tra dữ liệu thiếu:")
missing_data = train_df.isnull().sum()
missing_columns = missing_data[missing_data > 0]
if len(missing_columns) > 0:
    print("Dữ liệu thiếu trong từng cột:")
    print(missing_columns)
else:
    print("Không có dữ liệu thiếu trong dataset.")

# =============================================================================
# 2. TIỀN XỬ LÝ VÀ KỸ THUẬT ĐẶC TRƯNG
# =============================================================================

print("\n" + "="*50)
print("2. TIỀN XỬ LÝ VÀ KỸ THUẬT ĐẶC TRƯNG")
print("="*50)

# Tách đặc trưng và nhãn
X = train_df.drop('class', axis=1)
y = train_df['class']

# Mã hóa nhãn (e: edible - ăn được, p: poisonous - độc)
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # e -> 0, p -> 1

# Chia tập train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# =============================================================================
# LUỒNG A: KỸ THUẬT ĐẶC TRƯNG TRUYỀN THỐNG
# =============================================================================

print("\n" + "-"*30)
print("LUỒNG A: KỸ THUẬT ĐẶC TRƯNG TRUYỀN THỐNG")
print("-"*30)

# Xử lý dữ liệu thiếu (nếu có)
print("2.A.1. Xử lý dữ liệu thiếu:")
if len(missing_columns) > 0:
    # Điền giá trị thiếu bằng mode cho categorical features
    for column in missing_columns.index:
        if column in X.columns:
            mode_value = X[column].mode()[0]
            X_train[column] = X_train[column].fillna(mode_value)
            X_test[column] = X_test[column].fillna(mode_value)
    print("Đã xử lý dữ liệu thiếu")
else:
    print("Không có dữ liệu thiếu trong dataset, nên bỏ qua bước này.")

# Mã hóa đặc trưng phân loại
print("\n2.A.2. Mã hóa đặc trưng phân loại:")
print("Lý do: Sử dụng One-Hot Encoding vì tất cả các đặc trưng đều là định danh (nominal)")

# One-Hot Encoding
X_train_encoded = pd.get_dummies(X_train, prefix_sep='_')
X_test_encoded = pd.get_dummies(X_test, prefix_sep='_')

# Đảm bảo tập test có cùng cấu trúc với tập train
missing_cols = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols:
    X_test_encoded[col] = 0

extra_cols = set(X_test_encoded.columns) - set(X_train_encoded.columns)
X_test_encoded = X_test_encoded.drop(columns=extra_cols)

# Sắp xếp columns theo thứ tự
X_test_encoded = X_test_encoded[X_train_encoded.columns]

print(f"Kích thước sau One-Hot Encoding - Train: {X_train_encoded.shape}")
print(f"Kích thước sau One-Hot Encoding - Test: {X_test_encoded.shape}")

# Chuẩn hóa đặc trưng
print("\n2.A.3. Chuẩn hóa đặc trưng:")
print("Lý do: Sử dụng StandardScaler vì dữ liệu sau One-Hot Encoding có phân phối khác nhau")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

print("Hoàn thành chuẩn hóa đặc trưng")

# =============================================================================
# LUỒNG B: HỌC ĐẶC TRƯNG BẰNG AUTOENCODER
# =============================================================================

print("\n" + "-"*30)
print("LUỒNG B: HỌC ĐẶC TRƯNG BẰNG AUTOENCODER")
print("-"*30)

# Chuẩn bị dữ liệu cho Autoencoder
# Sử dụng Label Encoding cho các đặc trưng phân loại
label_encoders = {}
X_train_encoded_ae = X_train.copy()
X_test_encoded_ae = X_test.copy()

for column in X_train.columns:
    le_feature = LabelEncoder()
    # Xử lý các giá trị mới trong test set
    unique_train = set(X_train[column].unique())
    unique_test = set(X_test[column].unique())
    
    # Kết hợp tất cả giá trị
    all_values = list(unique_train.union(unique_test))
    le_feature.fit(all_values)
    
    X_train_encoded_ae[column] = le_feature.transform(X_train[column])
    X_test_encoded_ae[column] = le_feature.transform(X_test[column])
    label_encoders[column] = le_feature

# Chuẩn hóa dữ liệu
scaler_ae = StandardScaler()
X_train_scaled_ae = scaler_ae.fit_transform(X_train_encoded_ae)
X_test_scaled_ae = scaler_ae.transform(X_test_encoded_ae)

print(f"Kích thước dữ liệu cho Autoencoder: {X_train_scaled_ae.shape}")

# Xây dựng Autoencoder
print("\n2.B.1. Kiến trúc Autoencoder:")
input_dim = X_train_scaled_ae.shape[1]
encoding_dim = min(20, input_dim // 2)  # Đảm bảo encoding_dim không quá lớn

print("Lý do chọn kích thước không gian ẩn:")
print(f"- Dữ liệu gốc có {input_dim} đặc trưng")
print(f"- Chọn encoding_dim = {encoding_dim} để cân bằng giữa khả năng nén và bảo toàn thông tin")
print(f"- Tỷ lệ nén: {input_dim} -> {encoding_dim} (giảm {((input_dim-encoding_dim)/input_dim)*100:.1f}%)")

# Định nghĩa Autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
bottleneck = Dense(encoding_dim, activation='relu')(encoder)

decoder = Dense(32, activation='relu')(bottleneck)
decoder = Dense(64, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
encoder_model = Model(inputs=input_layer, outputs=bottleneck)

autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

print("\n2.B.2. Quá trình huấn luyện Autoencoder:")
print("Hàm mất mát: Mean Squared Error (MSE)")
print("Thuật toán tối ưu: Adam với learning_rate=0.001")
print("Số epochs: 30")  # Giảm để chạy nhanh hơn
print("Kích thước batch: 32")

# Huấn luyện Autoencoder
history = autoencoder.fit(
    X_train_scaled_ae, X_train_scaled_ae,
    epochs=30,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test_scaled_ae, X_test_scaled_ae),
    verbose=1  # Hiển thị tiến trình
)

# Vẽ đồ thị quá trình huấn luyện
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("\n2.B.3. Trích xuất đặc trưng:")
print("Sử dụng phần Encoder để trích xuất đặc trưng từ dữ liệu gốc")

# Trích xuất đặc trưng từ Autoencoder
X_train_ae_features = encoder_model.predict(X_train_scaled_ae, verbose=0)
X_test_ae_features = encoder_model.predict(X_test_scaled_ae, verbose=0)

print(f"Kích thước đặc trưng sau Autoencoder - Train: {X_train_ae_features.shape}")
print(f"Kích thước đặc trưng sau Autoencoder - Test: {X_test_ae_features.shape}")

# =============================================================================
# 3. HUẤN LUYỆN MÔ HÌNH
# =============================================================================

print("\n" + "="*50)
print("3. HUẤN LUYỆN MÔ HÌNH")
print("="*50)

# Sử dụng RandomForestClassifier để so sánh
print("Sử dụng RandomForestClassifier để so sánh hiệu quả của hai phương pháp")

# Huấn luyện trên đặc trưng truyền thống (Luồng A)
print("\n3.1. Huấn luyện trên đặc trưng truyền thống (Luồng A):")
rf_traditional = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                       min_samples_split=5, min_samples_leaf=2,
                                       random_state=42, n_jobs=-1)
rf_traditional.fit(X_train_scaled, y_train)
y_pred_traditional = rf_traditional.predict(X_test_scaled)

# Huấn luyện trên đặc trưng từ Autoencoder (Luồng B)
print("3.2. Huấn luyện trên đặc trưng từ Autoencoder (Luồng B):")
rf_autoencoder = RandomForestClassifier(n_estimators=100, max_depth=10,
                                       min_samples_split=5, min_samples_leaf=2,
                                       random_state=42, n_jobs=-1)
rf_autoencoder.fit(X_train_ae_features, y_train)
y_pred_autoencoder = rf_autoencoder.predict(X_test_ae_features)

# =============================================================================
# 4. ĐÁNH GIÁ VÀ PHÂN TÍCH KẾT QUẢ
# =============================================================================

print("\n" + "="*50)
print("4. ĐÁNH GIÁ VÀ PHÂN TÍCH KẾT QUẢ")
print("="*50)

# Đánh giá kết quả
accuracy_traditional = accuracy_score(y_test, y_pred_traditional)
accuracy_autoencoder = accuracy_score(y_test, y_pred_autoencoder)

print("4.1. Độ chính xác (Accuracy):")
print(f"Luồng A (Đặc trưng truyền thống): {accuracy_traditional:.4f}")
print(f"Luồng B (Đặc trưng từ Autoencoder): {accuracy_autoencoder:.4f}")

# Báo cáo chi tiết
print("\n4.2. Báo cáo chi tiết - Luồng A (Đặc trưng truyền thống):")
print(classification_report(y_test, y_pred_traditional, target_names=le.classes_))

print("\n4.3. Báo cáo chi tiết - Luồng B (Đặc trưng từ Autoencoder):")
print(classification_report(y_test, y_pred_autoencoder, target_names=le.classes_))

# Ma trận nhầm lẫn
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Luồng A
cm_traditional = confusion_matrix(y_test, y_pred_traditional)
sns.heatmap(cm_traditional, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title('Ma trận nhầm lẫn - Luồng A\n(Đặc trưng truyền thống)')
axes[0].set_xlabel('Dự đoán')
axes[0].set_ylabel('Thực tế')

# Luồng B
cm_autoencoder = confusion_matrix(y_test, y_pred_autoencoder)
sns.heatmap(cm_autoencoder, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title('Ma trận nhầm lẫn - Luồng B\n(Đặc trưng từ Autoencoder)')
axes[1].set_xlabel('Dự đoán')
axes[1].set_ylabel('Thực tế')

plt.tight_layout()
plt.show()

# =============================================================================
# 5. ĐÁNH GIÁ ĐỘ ỔN ĐỊNH QUA NHIỀU LẦN HUẤN LUYỆN
# =============================================================================

print("\n" + "="*60)
print("5. ĐÁNH GIÁ ĐỘ ỔN ĐỊNH QUA NHIỀU LẦN HUẤN LUYỆN")
print("="*60)

# Chuẩn bị dữ liệu tổng thể
X_encoded = pd.get_dummies(X, prefix_sep='_')
scaler_global = StandardScaler()
X_scaled_global = scaler_global.fit_transform(X_encoded)

# Chuẩn bị dữ liệu cho Autoencoder
X_encoded_ae = X.copy()
for column in X.columns:
    le_col = LabelEncoder()
    X_encoded_ae[column] = le_col.fit_transform(X[column])
X_scaled_ae_global = StandardScaler().fit_transform(X_encoded_ae)

# Huấn luyện Autoencoder trên toàn bộ dữ liệu
input_dim = X_scaled_ae_global.shape[1]
encoding_dim = min(20, input_dim // 2)

input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
bottleneck = Dense(encoding_dim, activation='relu')(encoder)
decoder = Dense(32, activation='relu')(bottleneck)
decoder = Dense(64, activation='relu')(decoder)
output_layer = Dense(input_dim, activation='sigmoid')(decoder)

autoencoder_global = Model(inputs=input_layer, outputs=output_layer)
encoder_global = Model(inputs=input_layer, outputs=bottleneck)
autoencoder_global.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
autoencoder_global.fit(X_scaled_ae_global, X_scaled_ae_global, 
                      epochs=30, batch_size=32, verbose=0)

X_ae_features_global = encoder_global.predict(X_scaled_ae_global, verbose=0)

# Thử nghiệm với nhiều random seed
n_iterations = 5  # Giảm số lần lặp để chạy nhanh hơn
traditional_accuracies = []
autoencoder_accuracies = []
traditional_f1_scores = []
autoencoder_f1_scores = []

print(f"\nThực hiện {n_iterations} lần huấn luyện với các random seed khác nhau...")

for i in tqdm(range(n_iterations)):
    # Chia dữ liệu với random seed khác nhau
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_global, y_encoded, test_size=0.2, 
        random_state=i, stratify=y_encoded
    )
    
    X_train_ae, X_test_ae, y_train_ae, y_test_ae = train_test_split(
        X_ae_features_global, y_encoded, test_size=0.2, 
        random_state=i, stratify=y_encoded
    )
    
    # Huấn luyện mô hình truyền thống
    rf_traditional = RandomForestClassifier(n_estimators=50, random_state=42)  # Giảm để chạy nhanh
    rf_traditional.fit(X_train, y_train)
    y_pred_traditional = rf_traditional.predict(X_test)
    
    # Huấn luyện mô hình Autoencoder
    rf_autoencoder = RandomForestClassifier(n_estimators=50, random_state=42)  # Giảm để chạy nhanh
    rf_autoencoder.fit(X_train_ae, y_train_ae)
    y_pred_autoencoder = rf_autoencoder.predict(X_test_ae)
    
    # Tính toán metrics
    acc_trad = accuracy_score(y_test, y_pred_traditional)
    acc_ae = accuracy_score(y_test_ae, y_pred_autoencoder)
    
    f1_trad = f1_score(y_test, y_pred_traditional, average='weighted')
    f1_ae = f1_score(y_test_ae, y_pred_autoencoder, average='weighted')
    
    traditional_accuracies.append(acc_trad)
    autoencoder_accuracies.append(acc_ae)
    traditional_f1_scores.append(f1_trad)
    autoencoder_f1_scores.append(f1_ae)

# Phân tích kết quả
print("\n" + "="*50)
print("PHÂN TÍCH ĐỘ ỔN ĐỊNH")
print("="*50)

# Thống kê mô tả
traditional_acc_mean = np.mean(traditional_accuracies)
traditional_acc_std = np.std(traditional_accuracies)
autoencoder_acc_mean = np.mean(autoencoder_accuracies)
autoencoder_acc_std = np.std(autoencoder_accuracies)

traditional_f1_mean = np.mean(traditional_f1_scores)
traditional_f1_std = np.std(traditional_f1_scores)
autoencoder_f1_mean = np.mean(autoencoder_f1_scores)
autoencoder_f1_std = np.std(autoencoder_f1_scores)

print("\nTHỐNG KÊ ĐỘ CHÍNH XÁC (Accuracy):")
print(f"Luồng A - Truyền thống: {traditional_acc_mean:.4f} ± {traditional_acc_std:.4f}")
print(f"Luồng B - Autoencoder:  {autoencoder_acc_mean:.4f} ± {autoencoder_acc_std:.4f}")

print("\nTHỐNG KÊ F1-SCORE:")
print(f"Luồng A - Truyền thống: {traditional_f1_mean:.4f} ± {traditional_f1_std:.4f}")
print(f"Luồng B - Autoencoder:  {autoencoder_f1_mean:.4f} ± {autoencoder_f1_std:.4f}")

# Kiểm định thống kê
try:
    t_stat_acc, p_value_acc = stats.ttest_rel(traditional_accuracies, autoencoder_accuracies)
    t_stat_f1, p_value_f1 = stats.ttest_rel(traditional_f1_scores, autoencoder_f1_scores)

    print(f"\nKIỂM ĐỊNH T-TEST (paired):")
    print(f"Accuracy - t-statistic: {t_stat_acc:.4f}, p-value: {p_value_acc:.4f}")
    print(f"F1-score - t-statistic: {t_stat_f1:.4f}, p-value: {p_value_f1:.4f}")

    if p_value_acc < 0.05:
        if traditional_acc_mean > autoencoder_acc_mean:
            print("→ Sự khác biệt về Accuracy có ý nghĩa thống kê (Luồng A tốt hơn)")
        else:
            print("→ Sự khác biệt về Accuracy có ý nghĩa thống kê (Luồng B tốt hơn)")
    else:
        print("→ Không có sự khác biệt có ý nghĩa thống kê về Accuracy")
except:
    print("Không thể thực hiện kiểm định thống kê do số lượng mẫu ít")

# =============================================================================
# 6. SO SÁNH VÀ KẾT LUẬN
# =============================================================================

print("\n" + "="*50)
print("6. SO SÁNH VÀ KẾT LUẬN")
print("="*50)

# So sánh kết quả
print("6.1. So sánh kết quả:")
print(f"Độ chính xác Luồng A: {accuracy_traditional:.4f}")
print(f"Độ chính xác Luồng B: {accuracy_autoencoder:.4f}")

if accuracy_traditional > accuracy_autoencoder:
    difference = accuracy_traditional - accuracy_autoencoder
    print(f"Luồng A tốt hơn Luồng B: {difference:.4f}")
    best_method = 'traditional'
elif accuracy_autoencoder > accuracy_traditional:
    difference = accuracy_autoencoder - accuracy_traditional
    print(f"Luồng B tốt hơn Luồng A: {difference:.4f}")
    best_method = 'autoencoder'
else:
    print("Hai phương pháp cho kết quả tương đương")
    best_method = 'traditional'  # Mặc định

# =============================================================================
# 7. XUẤT FILE SUBMISSION CHO KAGGLE
# =============================================================================

print("\n" + "="*50)
print("7. XUẤT FILE SUBMISSION CHO KAGGLE")
print("="*50)

# Chuẩn bị dữ liệu test thực tế
if 'id' in test_df.columns:
    test_ids = test_df['id']
    X_test_final = test_df.drop('id', axis=1)
else:
    test_ids = range(len(test_df))
    X_test_final = test_df.copy()

print(f"Kích thước dữ liệu test: {X_test_final.shape}")

# Chọn mô hình tốt nhất
if best_method == 'traditional':
    print("📊 Sử dụng mô hình TRUYỀN THỐNG để tạo submission...")

    # Tiền xử lý One-Hot Encoding
    X_full = pd.get_dummies(X, prefix_sep='_')
    X_test_final_enc = pd.get_dummies(X_test_final, prefix_sep='_')
    
    # Đảm bảo test data có cùng features với train data
    missing_cols = set(X_full.columns) - set(X_test_final_enc.columns)
    for col in missing_cols:
        X_test_final_enc[col] = 0
    
    extra_cols = set(X_test_final_enc.columns) - set(X_full.columns)
    X_test_final_enc = X_test_final_enc.drop(columns=extra_cols)
    
    X_test_final_enc = X_test_final_enc[X_full.columns]

    scaler_final = StandardScaler()
    X_full_scaled = scaler_final.fit_transform(X_full)
    X_test_final_scaled = scaler_final.transform(X_test_final_enc)

    final_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        min_samples_split=5, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1)
    final_model.fit(X_full_scaled, y_encoded)

    test_predictions = final_model.predict(X_test_final_scaled)

else:
    print("🤖 Sử dụng mô hình AUTOENCODER để tạo submission...")

    # Label Encoding cho test
    X_full_ae = X.copy()
    X_test_final_ae = X_test_final.copy()
    
    for column in X_full_ae.columns:
        le_col = LabelEncoder()
        unique_train = set(X_full_ae[column].unique())
        unique_test = set(X_test_final_ae[column].unique())
        
        # Kết hợp tất cả giá trị
        all_values = list(unique_train.union(unique_test))
        le_col.fit(all_values)
        
        X_full_ae[column] = le_col.transform(X_full_ae[column])
        X_test_final_ae[column] = le_col.transform(X_test_final_ae[column])

    scaler_final_ae = StandardScaler()
    X_full_scaled_ae = scaler_final_ae.fit_transform(X_full_ae)
    X_test_final_scaled_ae = scaler_final_ae.transform(X_test_final_ae)

    # Train autoencoder cuối cùng
    input_layer_final = Input(shape=(X_full_scaled_ae.shape[1],))
    encoder_final = Dense(64, activation='relu')(input_layer_final)
    encoder_final = Dense(32, activation='relu')(encoder_final)
    bottleneck_final = Dense(min(20, X_full_scaled_ae.shape[1]//2), activation='relu')(encoder_final)

    decoder_final = Dense(32, activation='relu')(bottleneck_final)
    decoder_final = Dense(64, activation='relu')(decoder_final)
    output_layer_final = Dense(X_full_scaled_ae.shape[1], activation='sigmoid')(decoder_final)

    autoencoder_final = Model(inputs=input_layer_final, outputs=output_layer_final)
    encoder_model_final = Model(inputs=input_layer_final, outputs=bottleneck_final)

    autoencoder_final.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    autoencoder_final.fit(X_full_scaled_ae, X_full_scaled_ae, epochs=30, batch_size=32, verbose=0)

    # Trích xuất đặc trưng
    X_full_features = encoder_model_final.predict(X_full_scaled_ae, verbose=0)
    X_test_final_features = encoder_model_final.predict(X_test_final_scaled_ae, verbose=0)

    # Train RandomForest trên đặc trưng học được
    final_model = RandomForestClassifier(n_estimators=100, max_depth=10,
                                        min_samples_split=5, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1)
    final_model.fit(X_full_features, y_encoded)

    test_predictions = final_model.predict(X_test_final_features)

# Chuyển dự đoán về e/p
final_predictions_labels = le.inverse_transform(test_predictions)

# Tạo file submission
submission = pd.DataFrame({
    "id": test_ids,
    "class": final_predictions_labels
})

# Xuất ra CSV
submission.to_csv("submission.csv", index=False)

print("✅ File submission.csv đã được tạo thành công!")
print(f"Kích thước file submission: {submission.shape}")
print