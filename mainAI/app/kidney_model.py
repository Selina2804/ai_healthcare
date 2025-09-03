import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import numpy as np

def train_kidney_model():
    print("Đang tải dữ liệu...")
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'kidney_disease.csv')
        df = pd.read_csv(csv_path)
        print("Đã tải dữ liệu thành công!")
    except Exception as e:
        print(f"Lỗi khi tải file CSV: {e}")
        return None
    
    df = preprocess_data(df)
    
    # Kiểm tra lại các cột số
    numeric_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_cols:
        if col in df.columns:
            # Chuyển đổi sang số, thay thế giá trị lỗi bằng NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Thay thế NaN bằng giá trị trung bình
    df = df.ffill()
    
    X = df.drop('classification', axis=1)
    y = df['classification']
    
    # Chỉ giữ lại các cột số
    X = X.select_dtypes(include=[np.number])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Đang train model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Độ chính xác model: {accuracy:.2%}')
    
    model_path = os.path.join(current_dir, 'kidney_model.pkl')
    joblib.dump(model, model_path)
    print(f"Đã lưu model tại: {model_path}")
    
    return model

def preprocess_data(df):
    print("Đang tiền xử lý dữ liệu...")
    
    # Xử lý các giá trị đặc biệt
    df.replace(['\t?', '\t', '?', 'nan', 'NaN'], np.nan, inplace=True)
    
    # Chuyển đổi categorical features
    categorical_mappings = {
        'rbc': {'normal': 1, 'abnormal': 0},
        'pc': {'normal': 1, 'abnormal': 0},
        'pcc': {'present': 1, 'notpresent': 0},
        'ba': {'present': 1, 'notpresent': 0},
        'htn': {'yes': 1, 'no': 0},
        'dm': {'yes': 1, 'no': 0},
        'cad': {'yes': 1, 'no': 0},
        'appet': {'good': 1, 'poor': 0},
        'pe': {'yes': 1, 'no': 0},
        'ane': {'yes': 1, 'no': 0},
        'classification': {'ckd': 1, 'notckd': 0}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df

if __name__ == "__main__":
    print("==== BẮT ĐẦU QUÁ TRÌNH TRAIN MODEL ====")
    trained_model = train_kidney_model()
    if trained_model:
        print("==== HOÀN TẤT ====")
    else:
        print("==== CÓ LỖI XẢY RA ====")