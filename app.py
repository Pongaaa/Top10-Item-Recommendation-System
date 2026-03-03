import streamlit as st
import json
import pandas as pd
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Recommender Audit", layout="wide", page_icon="👶")

# --- 1. HÀM LOAD DỮ LIỆU (Đã cập nhật xử lý định dạng chuỗi đặc biệt) ---
@st.cache_data
def load_all_data():
    # ---------------------------------------------------------
    # BƯỚC 1: LOAD THÔNG TIN SẢN PHẨM (PARQUET)
    # ---------------------------------------------------------
    item_path = 'dataset/item_2024.parquet' 
    
    if os.path.exists(item_path):
        items_df = pd.read_parquet(item_path)
        items_df['item_id'] = items_df['item_id'].astype(str)
        
        # Tạo map: ID -> Tên hiển thị
        item_map = items_df.set_index('item_id').apply(
            lambda x: f"{x['category_l3']} ({x['category_l1']})", axis=1
        ).to_dict()
    else:
        st.error(f"Không tìm thấy file {item_path}")
        item_map = {}

    # ---------------------------------------------------------
    # BƯỚC 2: LOAD GROUND TRUTH (CSV) - XỬ LÝ CHUỖI ĐẶC BIỆT
    # ---------------------------------------------------------
    gt_path = 'dataset/groundtruth_feb_2025.csv'
    ground_truth = {}
    
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        gt_df['customer_id'] = gt_df['customer_id'].astype(str)
        
        # --- LOGIC MỚI: XỬ LÝ CHUỖI "['item1' 'item2']" ---
        def parse_numpy_string(s):
            # 1. Bỏ dấu ngoặc vuông [] và dấu nháy đơn '
            clean_s = str(s).replace('[', '').replace(']', '').replace("'", "")
            # 2. Tách bằng khoảng trắng (vì file của bạn dùng dấu cách thay vì dấu phẩy)
            # Dùng split() không tham số sẽ tự động cắt mọi loại khoảng trắng
            return clean_s.split()

        # Áp dụng hàm xử lý
        gt_df['parsed_items'] = gt_df['item_id'].apply(parse_numpy_string)
        
        # Chuyển thành Dictionary: UserID -> List Items
        ground_truth = gt_df.set_index('customer_id')['parsed_items'].to_dict()
    else:
        st.error(f"Không tìm thấy file {gt_path}")

    # ---------------------------------------------------------
    # BƯỚC 3: LOAD PREDICTION (JSON)
    # ---------------------------------------------------------
    pred_path = 'dataset/submission.json'
    predictions = {}
    
    if os.path.exists(pred_path):
        with open(pred_path, 'r', encoding='utf-8') as f:
            predictions = json.load(f)
    else:
        st.error(f"Không tìm thấy file {pred_path}")

    return ground_truth, predictions, item_map

# Hàm phụ trợ để lấy tên hiển thị đẹp
def get_display_name(item_id, item_map):
    # Nếu tìm thấy trong map thì trả về tên, không thì trả về ID gốc
    return item_map.get(str(item_id), f"Item ID: {item_id}")

# --- 2. XỬ LÝ LOGIC TÍNH TOÁN ---
def calculate_metrics(actual, predicted):
    if not actual or not predicted:
        return 0.0, []
    
    # Chuyển về set để tính toán (đảm bảo tính duy nhất)
    set_actual = set(map(str, actual))
    set_predicted = set(map(str, predicted))
    
    # Tìm sản phẩm đúng
    correct_items = list(set_actual & set_predicted)
    
    # Precision@K = (Số món đúng) / (Tổng số món gợi ý)
    precision = len(correct_items) / len(predicted) if len(predicted) > 0 else 0
    
    return precision * 100, correct_items

# --- 3. GIAO DIỆN CHÍNH ---
def main():
    st.title("🛍️ Hệ thống Đánh giá Gợi ý Sản phẩm")
    st.caption("Dữ liệu: Groundtruth (CSV) vs Prediction (JSON) vs Item Info (Parquet)")
    st.markdown("---")

    # Load dữ liệu
    ground_truth, predictions, item_map = load_all_data()
    
    if not predictions:
        st.warning("Chưa có dữ liệu dự đoán. Vui lòng kiểm tra đường dẫn file.")
        return

    # --- SIDEBAR: CHỌN KHÁCH HÀNG ---
    st.sidebar.header("🔍 Tra cứu Khách hàng")
    
    # Lấy danh sách user có trong file dự đoán
    all_users = list(predictions.keys())
    
    if not all_users:
        st.stop()
        
    selected_user = st.sidebar.selectbox(
        "Chọn Mã Khách Hàng (User ID):", 
        all_users,
        index=4
    )

    # Lấy dữ liệu của user
    actual_items = ground_truth.get(selected_user, [])
    predicted_items = predictions.get(selected_user, [])

    # Tính toán chỉ số
    score, correct_items = calculate_metrics(actual_items, predicted_items)

    # --- HIỂN THỊ KPI ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sản phẩm gợi ý", len(predicted_items))
    with col2:
        st.metric("Thực tế mua", len(actual_items))
    with col3:
        st.metric("Độ chính xác (Precision)", f"{score:.1f}%", f"{len(correct_items)} đúng")

    st.markdown("---")

    # --- HIỂN THỊ SO SÁNH ---
    col_pred, col_true = st.columns(2)

    with col_pred:
        st.subheader("🤖 AI Dự đoán (Tháng tới)")
        if predicted_items:
            for item_id in predicted_items:
                item_name = get_display_name(item_id, item_map)
                
                display_content = f"**{item_name}** (ID: {item_id})"
                
                if str(item_id) in correct_items:
                    st.success(f"✅ {display_content}")
                else:
                    st.warning(f"⬜ {display_content}")
        else:
            st.info("Không có gợi ý nào cho khách hàng này.")

    with col_true:
        st.subheader("🛒 Thực tế Khách mua")
        if actual_items:
            for item_id in actual_items:
                item_name = get_display_name(item_id, item_map)
                
                display_content = f"**{item_name}** (ID: {item_id})"
                
                if str(item_id) in correct_items:
                    st.success(f"🎯 {display_content}")
                else:
                    st.error(f"⚠️ {display_content} (AI bỏ sót)")
        else:
            st.info("Khách hàng không mua gì trong tháng này.")

if __name__ == "__main__":
    main()