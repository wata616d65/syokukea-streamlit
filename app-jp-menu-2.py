import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
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

# 実行中のスクリプトのディレクトリを取得
SCRIPT_DIR = Path(__file__).resolve().parent
# スクリプトの場所を基準に画像フォルダへのパスを作成
IMAGE_BASE_PATH = SCRIPT_DIR / "UECFOOD256"

# 1日の推奨摂取量
daily_needs = {
    'energy_kcal': 2650, 'protein_g': 65, 'fat_g': 73.6, 'carbohydrate_g': 378.1,
    'calcium_mg': 800, 'iron_mg': 7.5, 'vitamin_c_mg': 100, 'vitamin_b1_mg': 1.4,
    'vitamin_b2_mg': 1.6, 'fiber_g': 21, 'sodium_mg': 2362
}

# --- ヘルパー関数 ---

def find_random_image(directory: Path) -> str | None:
    """指定されたディレクトリ内のランダムな画像ファイルパスを文字列で返す"""
    if not directory.is_dir():
        return None
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        return None
        
    # Pathオブジェクトを絶対パスの文字列に変換して返す
    return str(random.choice(image_files).resolve())

# 栄養素データの前処理を関数外で行う元のスタイルを維持
# ただし、読み込みはキャッシュ関数内で行うため、ここでは参照しない
# nutrition_df = pd.read_csv("master_natrition.csv")
# nutrition_df = nutrition_df.rename(columns={'food_name': '料理名'})

def recommend_foods(deficiency_data: dict, nutrition_df: pd.DataFrame, detected_ids: set, num_recommendations: int = 5) -> dict:
    """不足している栄養素を補う料理を推薦する（画像パス付き）"""
    jp_to_eng_map = {v: k for k, v in nutrition_jp_map.items()}
    recommendations = {}
    sorted_deficiencies = sorted(deficiency_data.items(), key=lambda item: item[1]['不足分'], reverse=True)
    
    for jp_nutrient, values in sorted_deficiencies[:3]:
        eng_nutrient_col = jp_to_eng_map.get(jp_nutrient)
        
        if eng_nutrient_col and eng_nutrient_col in nutrition_df.columns:
            recommend_df = nutrition_df[~nutrition_df.index.isin(detected_ids)]
            top_foods = recommend_df.sort_values(by=eng_nutrient_col, ascending=False).head(num_recommendations).copy()
            
            top_foods['image_path'] = top_foods.index.to_series().apply(
                lambda food_id: find_random_image(IMAGE_BASE_PATH / str(food_id))
            )
            
            # 注目: '料理名'列はload_nutrition_data関数内でリネーム済み
            result_df = top_foods[['料理名', eng_nutrient_col, 'image_path']].copy()
            result_df.rename(columns={eng_nutrient_col: jp_nutrient}, inplace=True)
            recommendations[jp_nutrient] = result_df
            
    return recommendations

# --- データとモデルの読み込み ---

@st.cache_resource
def load_yolo_model(path="best-2.pt"):
    """YOLOモデルをロード（キャッシュで高速化）"""
    try:
        # ユーザーの元のコードを維持
        model = YOLO(SCRIPT_DIR / path)
        return model
    except Exception as e:
        st.error(f"モデル '{path}' の読み込みに失敗しました: {e}")
        return None

@st.cache_data
def load_nutrition_data(path="master_natrition.csv"):
    """栄養素データベースをロード（キャッシュで高速化）"""
    try:
        # パスの扱いを堅牢に
        csv_path = SCRIPT_DIR / path
        if not csv_path.is_file():
            st.error(f"栄養素データベース '{path}' が見つかりません。")
            return None

        df = pd.read_csv(csv_path)
        # 'food_name'列を'料理名'にリネーム
        df.rename(columns={'food_name': '料理名'}, inplace=True)
        # 数値列の()や-を0に置換して数値型に変換
        numeric_cols = [col for col in df.columns if df[col].dtype != 'object' and col not in ['num']]
        numeric_cols.extend(['energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g', 'calcium_mg', 'iron_mg', 'vitamin_c_mg', 'vitamin_b1_mg', 'vitamin_b2_mg', 'fiber_g', 'sodium_mg'])
        
        for col in set(numeric_cols):
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\(\)-]', '0', regex=True), errors='coerce').fillna(0)

        df.set_index('num', inplace=True)
        return df
    except Exception as e:
        st.error(f"CSVファイルの読み込み中にエラーが発生しました: {e}")
        return None

# モデルとデータをロード（元の呼び出し方を維持）
model = load_yolo_model()
nutrition_df = load_nutrition_data()


# --- Streamlit アプリケーション画面 ---
st.title('🥗 食事分析AI')
st.write('食事の写真をアップロードすると、含まれる栄養素を分析し、1日の摂取基準に足りない栄養素と、それを補うメニューをお知らせします。')

if not IMAGE_BASE_PATH.is_dir():
    st.error(f"画像フォルダ '{IMAGE_BASE_PATH.name}' が見つかりません。app.pyと同じ階層に配置してください。")
else:
    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None and model is not None and nutrition_df is not None:
        image = Image.open(uploaded_file)
        results = model(image) 
        
        detected_items_jp, detected_ids = [], []
        # '料理名' を除外した栄養素列を取得
        nutrition_cols = [col for col in nutrition_df.columns if col != '料理名']
        total_nutrition = pd.Series(0.0, index=nutrition_cols) 
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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
                    # '料理名' 列を参照
                    detected_items_jp.append(nutrition_df.loc[nutrition_id, '料理名'])
                    total_nutrition += nutrition_df.loc[nutrition_id, nutrition_cols]

        st.subheader("📸 検出結果")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='検出された料理', use_column_width=True)
        
        if detected_items_jp:
            st.write(f"検出された料理: **{', '.join(set(detected_items_jp))}**")
            st.subheader("📊 この食事の栄養素")
            display_nutrition = total_nutrition.get(list(daily_needs.keys()), pd.Series(0.0, index=list(daily_needs.keys()))).copy()
            display_nutrition.rename(index=nutrition_jp_map, inplace=True)
            st.dataframe(display_nutrition.rename('摂取量').to_frame())

            st.subheader("💪 1日の目標に対する不足栄養素")
            deficiency_data = {}
            for key, daily_value in daily_needs.items():
                meal_value = total_nutrition.get(key, 0)
                deficiency = daily_value - meal_value
                if deficiency > 0:
                    jp_key = nutrition_jp_map.get(key, key)
                    deficiency_data[jp_key] = {
                        "1日の目標": daily_value, "この食事の摂取量": meal_value, "不足分": deficiency
                    }
            
            if deficiency_data:
                df_deficiency = pd.DataFrame.from_dict(deficiency_data, orient='index')
                st.warning("以下の栄養素が不足しています。")
                st.dataframe(df_deficiency.style.format('{:.2f}'))

                recommendations = recommend_foods(deficiency_data, nutrition_df, set(detected_ids))

                if recommendations:
                    st.subheader("💡 不足分を補うおすすめメニュー")
                    st.write("特に不足している栄養素を補うには、以下のような料理がおすすめです。")
                    
                    for nutrient, food_df in recommendations.items():
                        with st.expander(f"**「{nutrient}」**が豊富な料理TOP5", expanded=True):
                            for index, row in food_df.iterrows():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    # ★★★ここがメインの修正点★★★
                                    image_path = row.get('image_path')
                                    # パスが存在し、それが「ファイル」であることを確認してから表示
                                    if image_path and Path(image_path).is_file():
                                        st.image(image_path, use_column_width=True)
                                    else:
                                        st.text("画像なし")
                                with col2:
                                    # '料理名'列を正しく参照
                                    st.write(f"**{row['料理名']}**") 
                                    st.write(f"{nutrient}: {row[nutrient]:.2f}")
                                st.divider()
                
            else:
                st.success("素晴らしい！この食事で1日の主要な栄養素目標を達成できそうです。")
        else:
            st.info("写真から料理を検出できませんでした。別の画像を試してください。")