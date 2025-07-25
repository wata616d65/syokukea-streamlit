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

# 実行中のスクリプトのディレクトリを取得
SCRIPT_DIR = Path(__file__).resolve().parent
# UECFOOD256データセットへのベースパスを設定
IMAGE_BASE_PATH = SCRIPT_DIR / "UECFOOD256"


# --- ヘルパー関数 ---

def find_random_image(directory: Path):
    """
    指定されたディレクトリ内からランダムな画像ファイルのパス(Pathオブジェクト)を返す。
    
    Args:
        directory (Path): 画像を検索するディレクトリのPathオブジェクト。
        
    Returns:
        Path or None: 画像ファイルのPathオブジェクト。見つからない場合はNone。
    """
    # ディレクトリが存在しない場合はNoneを返す
    if not directory.is_dir():
        return None
    
    # ディレクトリ内の画像ファイル（jpg, jpeg, png）をリストアップ
    image_files = [
        file_path for file_path in directory.iterdir() 
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ]
    
    # 画像ファイルがあれば、その中からランダムに1つ選んで返す
    if image_files:
        return random.choice(image_files)
    
    # 画像ファイルがなければNoneを返す
    return None

def recommend_foods(deficiency_data, nutrition_df, detected_ids, num_recommendations=5):
    """
    不足している栄養素を補う料理を推薦する（画像パス付き）。
    
    Args:
        deficiency_data (dict): 栄養素の不足状況データ。
        nutrition_df (pd.DataFrame): 全食品の栄養素データフレーム。
        detected_ids (set): 既に食事に含まれている食品IDのセット。
        num_recommendations (int): 各栄養素ごとのおすすめ表示件数。
        
    Returns:
        dict: 栄養素ごとのおすすめ料理データフレームを格納した辞書。
    """
    jp_to_eng_map = {v: k for k, v in nutrition_jp_map.items()}
    recommendations = {}
    # 不足分が多い順に栄養素をソートし、上位3つを対象とする
    sorted_deficiencies = sorted(deficiency_data.items(), key=lambda item: item[1]['不足分'], reverse=True)
    
    for jp_nutrient, values in sorted_deficiencies[:3]:
        eng_nutrient_col = jp_to_eng_map.get(jp_nutrient)
        
        if eng_nutrient_col and eng_nutrient_col in nutrition_df.columns:
            # 既に検出された料理を除外して、推薦候補を絞り込む
            recommend_df = nutrition_df[~nutrition_df.index.isin(detected_ids)]
            # 該当の栄養素を多く含む順にソートし、上位N件を取得
            top_foods = recommend_df.sort_values(by=eng_nutrient_col, ascending=False).head(num_recommendations)
            
            # ★変更点: 各料理に対応する画像パスを取得する
            top_foods['image_path'] = top_foods.index.to_series().apply(
                lambda food_id: find_random_image(IMAGE_BASE_PATH / str(food_id))
            )
            
            # 結果を整形して辞書に格納
            result_df = top_foods[['food_name', eng_nutrient_col, 'image_path']].copy()
            result_df.rename(columns={'food_name': '料理名', eng_nutrient_col: jp_nutrient}, inplace=True)
            recommendations[jp_nutrient] = result_df
            
    return recommendations

# --- データとモデルの読み込み（キャッシュ利用） ---

@st.cache_resource
def load_yolo_model(path="best-2.pt"):
    """YOLOモデルをロード（キャッシュで高速化）"""
    model_path = SCRIPT_DIR / path
    if not model_path.exists():
        st.error(f"モデルファイル '{path}' が見つかりません。")
        return None
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"モデル '{path}' の読み込みに失敗しました: {e}")
        return None

@st.cache_data
def load_nutrition_data(path="master_natrition.csv"):
    """栄養素データベースをロード（キャッシュで高速化）"""
    csv_path = SCRIPT_DIR / path
    if not csv_path.exists():
        st.error(f"栄養素データベース '{path}' が見つかりません。")
        return None
    try:
        df = pd.read_csv(csv_path)
        # 数値でない可能性のある列を前処理
        for col in df.columns[4:]:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\(\)-]', '0', regex=True), errors='coerce').fillna(0)
        df.set_index('num', inplace=True)
        return df
    except Exception as e:
        st.error(f"CSVファイル '{path}' の読み込み中にエラーが発生しました: {e}")
        return None

# モデルとデータをロード
model = load_yolo_model()
nutrition_df = load_nutrition_data()

# 1日の推奨摂取量（成人男性30-49歳の身体活動レベルIIを想定）
daily_needs = {
    'energy_kcal': 2700, 'protein_g': 65, 'fat_g': 75, # 脂質はエネルギー比20-30%から計算
    'carbohydrate_g': 371, # 炭水化物はエネルギー比50-65%から計算
    'calcium_mg': 750, 'iron_mg': 7.5, 'vitamin_c_mg': 100, 
    'vitamin_b1_mg': 1.4, 'vitamin_b2_mg': 1.6, 'fiber_g': 21, 'sodium_mg': 3000 # 食塩相当量7.5g
}

# --- Streamlit アプリケーション画面 ---
st.title('🥗 食事分析AI')
st.write('食事の写真をアップロードすると、含まれる栄養素を分析し、1日の摂取基準に足りない栄養素と、それを補うメニューをお知らせします。')

# 必要なファイルやディレクトリが存在するかチェック
if model is None or nutrition_df is None:
    st.stop()

if not IMAGE_BASE_PATH.is_dir():
    st.error(f"画像フォルダ '{IMAGE_BASE_PATH.name}' が見つかりません。app.pyと同じ階層に配置してください。")
    st.stop()

# ファイルアップローダー
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # YOLOモデルで物体検出を実行
    results = model(image) 
    
    detected_items_jp, detected_ids = [], []
    total_nutrition = pd.Series(0.0, index=nutrition_df.columns[3:]) 
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # 検出結果の処理
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            # 栄養素DBのIDに合わせる (YOLOのclass_idが0から始まるため+1)
            nutrition_id = class_id + 1 
            detected_ids.append(nutrition_id)
            
            # 検出した物体を矩形で囲む
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{result.names[class_id]}'
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 検出した料理の栄養素を加算
            if nutrition_id in nutrition_df.index:
                item_name_jp = nutrition_df.loc[nutrition_id, 'food_name']
                detected_items_jp.append(item_name_jp)
                total_nutrition += nutrition_df.loc[nutrition_id].iloc[3:]

    st.subheader("📸 検出結果")
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='検出された料理', use_column_width=True)
    
    if detected_items_jp:
        st.write(f"検出された料理: **{', '.join(set(detected_items_jp))}**")
        st.subheader("📊 この食事の栄養素")
        display_nutrition = total_nutrition[daily_needs.keys()].copy()
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

            # 不足栄養素を補うメニューを推薦
            recommendations = recommend_foods(deficiency_data, nutrition_df, set(detected_ids))

            if recommendations:
                st.subheader("💡 不足分を補うおすすめメニュー")
                st.write("特に不足している栄養素を補うには、以下のような料理がおすすめです。")
                
                for nutrient, food_df in recommendations.items():
                    with st.expander(f"**「{nutrient}」**が豊富な料理TOP5"):
                        if food_df.empty:
                            st.write("おすすめの料理が見つかりませんでした。")
                            continue
                        
                        for index, row in food_df.iterrows():
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                # ★変更点: 画像パス(Pathオブジェクト)を処理し、存在すれば表示
                                image_path = row['image_path']
                                if image_path and image_path.exists():
                                    st.image(str(image_path), use_column_width=True)
                                else:
                                    # ★変更点: 画像がない場合はプレースホルダーを表示
                                    st.image("https://placehold.co/400x300/eee/ccc?text=画像なし", caption="画像なし", use_column_width=True)
                            with col2:
                                st.write(f"**{row['料理名']}**")
                                st.write(f"{nutrient}: {row[nutrient]:.2f}")
                            st.divider()
            
        else:
            st.success("素晴らしい！この食事で1日の主要な栄養素目標を達成できそうです。")
    else:
        st.info("写真から料理を検出できませんでした。別の画像を試してください。")
