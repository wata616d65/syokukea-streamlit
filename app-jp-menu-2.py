import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import os
from pathlib import Path

# --- 初期設定 ---

nutrition_jp_map = {
    'energy_kcal': 'エネルギー (kcal)', 'protein_g': 'タンパク質 (g)', 'fat_g': '脂質 (g)',
    'carbohydrate_g': '炭水化物 (g)', 'calcium_mg': 'カルシウム (mg)', 'iron_mg': '鉄 (mg)',
    'vitamin_c_mg': 'ビタミンC (mg)', 'vitamin_b1_mg': 'ビタミンB1 (mg)', 'vitamin_b2_mg': 'ビタミンB2 (mg)',
    'fiber_g': '食物繊維 (g)', 'sodium_mg': 'ナトリウム (mg)'
}

IMAGE_BASE_PATH = "UECFOOD256"

# --- ヘルパー関数 ---

def get_single_image_path(food_id):
    """
    指定したfood_idフォルダ内にある1枚だけの画像ファイルのパスを返す。
    画像がなければNoneを返す。
    """
    folder = Path(IMAGE_BASE_PATH) / str(food_id)
    if not folder.is_dir():
        return None
    # jpg, jpeg, pngすべて対応
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    if len(image_files) == 1:
        return str(image_files[0])
    elif len(image_files) > 0:
        # 万一複数ある場合は1枚目
        return str(image_files[0])
    return None

def recommend_foods(deficiency_data, nutrition_df, detected_ids, num_recommendations=5):
    jp_to_eng_map = {v: k for k, v in nutrition_jp_map.items()}
    recommendations = {}
    sorted_deficiencies = sorted(deficiency_data.items(), key=lambda item: item[1]['不足分'], reverse=True)

    for jp_nutrient, values in sorted_deficiencies[:3]:
        eng_nutrient_col = jp_to_eng_map.get(jp_nutrient)

        if eng_nutrient_col and eng_nutrient_col in nutrition_df.columns:
            recommend_df = nutrition_df[~nutrition_df.index.isin(detected_ids)]
            top_foods = recommend_df.sort_values(by=eng_nutrient_col, ascending=False).head(num_recommendations)

            # 画像パスを取得
            top_foods['image_path'] = top_foods.index.to_series().apply(get_single_image_path)

            result_df = top_foods[['food_name', eng_nutrient_col, 'image_path']].copy()
            result_df.rename(columns={'food_name': '料理名', eng_nutrient_col: jp_nutrient}, inplace=True)
            recommendations[jp_nutrient] = result_df

    return recommendations

# --- データとモデルの読み込み ---

@st.cache_resource
def load_yolo_model(path="best-2.pt"):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"モデル '{path}' の読み込みに失敗しました: {e}")
        return None

@st.cache_data
def load_nutrition_data(path="master_natrition.csv"):
    try:
        df = pd.read_csv(path)
        for col in df.columns[4:]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\(\)-]', '0', regex=True), errors='coerce').fillna(0)
        df.set_index('num', inplace=True)
        return df
    except FileNotFoundError:
        st.error(f"栄養素データベース '{path}' が見つかりません。")
        return None
    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
        return None

model = load_yolo_model()
nutrition_df = load_nutrition_data()

daily_needs = {
    'energy_kcal': 2650, 'protein_g': 65, 'fat_g': 73.6, 'carbohydrate_g': 378.1,
    'calcium_mg': 800, 'iron_mg': 7.5, 'vitamin_c_mg': 100, 'vitamin_b1_mg': 1.4,
    'vitamin_b2_mg': 1.6, 'fiber_g': 21, 'sodium_mg': 2362
}

def analyze_meal(image, model, nutrition_df):
    detected_items_jp, detected_ids = [], []
    total_nutrition = pd.Series(0.0, index=nutrition_df.columns[3:])
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(image)
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            nutrition_id = class_id + 1
            detected_ids.append(nutrition_id)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{result.names[class_id]}'
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if nutrition_id in nutrition_df.index:
                item_name_jp = nutrition_df.loc[nutrition_id, 'food_name']
                detected_items_jp.append(item_name_jp)
                total_nutrition += nutrition_df.loc[nutrition_id].iloc[3:]
    return detected_items_jp, detected_ids, total_nutrition, img_bgr



# --- Streamlit アプリ ---
st.title('🥗 食事分析AI - 3食管理バージョン')
st.write('3食分の写真をアップロードして、1日の栄養素摂取状況を可視化・管理できます。')

if not os.path.isdir(IMAGE_BASE_PATH):
    st.error(f"画像フォルダ '{IMAGE_BASE_PATH}' が見つかりません。app.pyと同じ階層に配置してください。")
else:
    st.header("画像アップロード（3食分）")
    col1, col2, col3 = st.columns(3)
    with col1:
        breakfast_img = st.file_uploader("朝食画像", type=["jpg", "png", "jpeg"], key="breakfast")
    with col2:
        lunch_img = st.file_uploader("昼食画像", type=["jpg", "png", "jpeg"], key="lunch")
    with col3:
        dinner_img = st.file_uploader("夕食画像", type=["jpg", "png", "jpeg"], key="dinner")

    # 実行ボタン
    if st.button("3食分の食事を分析する"):
        meal_images = {
            "朝食": breakfast_img,
            "昼食": lunch_img,
            "夕食": dinner_img
        }
        meal_results = {}
        total_nutrition = pd.Series(0.0, index=nutrition_df.columns[3:])
        all_detected_ids = set()
        all_detected_items = []

        for meal_name, img_file in meal_images.items():
            if img_file is not None:
                image = Image.open(img_file)
                detected_items_jp, detected_ids, meal_nutrition, img_bgr = analyze_meal(image, model, nutrition_df)
                meal_results[meal_name] = {
                    "detected_items_jp": detected_items_jp,
                    "detected_ids": detected_ids,
                    "meal_nutrition": meal_nutrition,
                    "img_bgr": img_bgr
                }
                total_nutrition += meal_nutrition
                all_detected_ids.update(detected_ids)
                all_detected_items.extend(detected_items_jp)
            else:
                meal_results[meal_name] = None

        # --- 結果表示 ---
        st.header("3食の検出結果")
        for meal_name, result in meal_results.items():
            st.subheader(f"{meal_name}")
            if result is not None:
                st.image(cv2.cvtColor(result["img_bgr"], cv2.COLOR_BGR2RGB), caption=f'{meal_name}の検出', use_column_width=True)
                if result["detected_items_jp"]:
                    st.write(f"検出料理: **{', '.join(set(result['detected_items_jp']))}**")
                    disp = result["meal_nutrition"][daily_needs.keys()].copy()
                    disp.rename(index=nutrition_jp_map, inplace=True)
                    st.dataframe(disp.rename('摂取量').to_frame())
                else:
                    st.info("料理を検出できませんでした。")
            else:
                st.info("画像が未入力です。")

        # --- 合計・不足表示 ---
        st.header("1日合計の栄養素摂取量")
        disp_total = total_nutrition[daily_needs.keys()].copy()
        disp_total.rename(index=nutrition_jp_map, inplace=True)
        st.dataframe(disp_total.rename('摂取量').to_frame())

        st.header("1日の目標に対する不足栄養素")
        deficiency_data = {}
        for key, daily_value in daily_needs.items():
            meal_value = total_nutrition.get(key, 0)
            deficiency = daily_value - meal_value
            if deficiency > 0:
                jp_key = nutrition_jp_map.get(key, key)
                deficiency_data[jp_key] = {
                    "1日の目標": daily_value,
                    "摂取量合計": meal_value,
                    "不足分": deficiency
                }

        if deficiency_data:
            df_deficiency = pd.DataFrame.from_dict(deficiency_data, orient='index')
            st.warning("以下の栄養素が不足しています。")
            st.dataframe(df_deficiency.style.format('{:.2f}'))
            recommendations = recommend_foods(deficiency_data, nutrition_df, all_detected_ids)
            if recommendations:
                st.subheader("💡 不足分を補うおすすめメニュー")
                for nutrient, food_df in recommendations.items():
                    with st.expander(f"**「{nutrient}」**が豊富な料理TOP5"):
                        for index, row in food_df.iterrows():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                if row['image_path'] and os.path.exists(row['image_path']):
                                    st.image(row['image_path'])
                                else:
                                    st.text("画像なし")
                            with col2:
                                st.write(f"**{row['料理名']}**")
                                st.write(f"{nutrient}: {row[nutrient]:.2f}")
                            st.divider()
        else:
            st.success("素晴らしい！今日の3食で1日の主要な栄養素目標を達成できそうです。")