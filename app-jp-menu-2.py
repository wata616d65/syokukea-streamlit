import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path
import random

# --- 初期設定 ---

# 栄養素名（英語→日本語）のマッピング辞書
nutrition_jp_map = {
    'energy_kcal': 'エネルギー (kcal)', 'protein_g': 'タンパク質 (g)', 'fat_g': '脂質 (g)',
    'carbohydrate_g': '炭水化物 (g)', 'calcium_mg': 'カルシウム (mg)', 'iron_mg': '鉄 (mg)',
    'vitamin_c_mg': 'ビタミンC (mg)', 'vitamin_b1_mg': 'ビタミンB1 (mg)', 'vitamin_b2_mg': 'ビタミンB2 (mg)',
    'fiber_g': '食物繊維 (g)', 'sodium_mg': 'ナトリウム (mg)'
}

# パス設定
SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_BASE_PATH = SCRIPT_DIR / "UECFOOD256"


# --- ヘルパー関数 ---

def find_random_image(directory):
    """指定されたディレクトリ内のランダムな画像ファイルパスを文字列で返す"""
    p = Path(directory)
    if not p.is_dir():
        return None
    
    image_files = [
        file_path for file_path in p.iterdir() 
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    
    if not image_files:
        return None
        
    return str(random.choice(image_files))

def detect_and_sum_nutrition(image, model, nutrition_df, master_df):
    results = model(image)
    detected_classes = results[0].boxes.cls.cpu().numpy().astype(int)
    
    if len(detected_classes) == 0:
        return pd.DataFrame(), set()

    detected_ids = set()
    for class_id in detected_classes:
        food_name = master_df.loc[master_df['class_id'] == class_id, 'food_name'].values
        if len(food_name) > 0:
            # nutrition_dfの'料理名'列と照合
            matched_id = nutrition_df.loc[nutrition_df['料理名'] == food_name[0], 'num'].values
            if len(matched_id) > 0:
                detected_ids.add(matched_id[0])

    if not detected_ids:
        return pd.DataFrame(), set()

    total_nutrition = nutrition_df[nutrition_df['num'].isin(detected_ids)].iloc[:, 4:-3].sum()
    return total_nutrition.to_frame('Total'), detected_ids

def recommend_foods(deficiency_data, nutrition_df, excluded_ids):
    recommendations = {}
    top_deficiencies = deficiency_data.head(3).index

    # 検出された料理を除外
    available_foods = nutrition_df[~nutrition_df['num'].isin(excluded_ids)].copy()

    for nutrient in top_deficiencies:
        nutrient_jp = nutrition_jp_map.get(nutrient, nutrient)
        
        # 欠損値を含む行をドロップ
        recommended_df = available_foods.dropna(subset=[nutrient]).nlargest(5, nutrient)
        
        # おすすめ料理に画像パスを追加
        recommended_df['image_path'] = recommended_df['num'].apply(
            lambda x: find_random_image(IMAGE_BASE_PATH / str(x))
        )
        recommendations[nutrient_jp] = recommended_df
    
    return recommendations


# --- Streamlit アプリケーション ---

st.title("栄養管理アプリ")

# データの読み込み
model = YOLO("best-2.pt")
master_df = pd.read_csv("master_natrition.csv")

# 栄養素データの読み込み
nutrition_df = pd.read_csv("master_natrition.csv")
# ★★★ ここが修正点です ★★★
# 'food_name'列を'料理名'にリネームして、エラーを防ぎます
nutrition_df = nutrition_df.rename(columns={'food_name': '料理名'})


uploaded_file = st.file_uploader("食事の写真をアップロードしてください", type=["jpg", "jpeg", "png"])

# 目標値の設定
target_nutrition = {
    'energy_kcal': 2200, 'protein_g': 65, 'fat_g': 55,
    'carbohydrate_g': 330, 'calcium_mg': 800, 'iron_mg': 10.5,
    'vitamin_c_mg': 100, 'vitamin_b1_mg': 1.4, 'vitamin_b2_mg': 1.6,
    'fiber_g': 21, 'sodium_mg': 2800 
}
df_target = pd.DataFrame.from_dict(target_nutrition, orient='index', columns=['Target'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    if st.button("診断"):
        with st.spinner("栄養素を計算中..."):
            img_array = np.array(image)
            total_nutrition, detected_ids = detect_and_sum_nutrition(img_array, model, nutrition_df, master_df)
        
        if not total_nutrition.empty:
            df_comparison = pd.concat([total_nutrition, df_target], axis=1).fillna(0)
            df_comparison.index = df_comparison.index.map(nutrition_jp_map)
            df_comparison.columns = ['摂取量', '目標値']
            
            st.subheader("摂取栄養素と目標値")
            st.dataframe(df_comparison.style.format('{:.2f}'))
            
            deficiency_data = df_target['Target'] - total_nutrition['Total']
            deficiency_data = deficiency_data[deficiency_data > 0].sort_values(ascending=False)
            
            if not deficiency_data.empty:
                df_deficiency = deficiency_data.to_frame('不足分')
                df_deficiency.index = df_deficiency.index.map(nutrition_jp_map)
                st.warning("以下の栄養素が不足しています。")
                st.dataframe(df_deficiency.style.format('{:.2f}'))

                recommendations = recommend_foods(deficiency_data, nutrition_df, set(detected_ids))

                if recommendations:
                    st.subheader("💡 不足分を補うおすすめメニュー")
                    st.write("特に不足している栄養素を補うには、以下のような料理がおすすめです。")
                    
                    for nutrient, food_df in recommendations.items():
                        with st.expander(f"**「{nutrient}」**が豊富な料理TOP5"):
                            for index, row in food_df.iterrows():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    image_path = row.get('image_path')
                                    if image_path and os.path.exists(image_path):
                                        st.image(image_path)
                                    else:
                                        st.text("画像なし")
                                with col2:
                                    st.write(f"**{row['料理名']}**")
                                    st.write(f"{nutrient}: {row[nutrient]:.2f}")
                                st.divider()
                
            else:
                st.success("素晴らしい！この食事で1日の主要な栄養素目標を達成できそうです。")
        else:
            st.info("写真から料理を検出できませんでした。別の画像を試してください。")