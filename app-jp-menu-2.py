import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import random

# --- 定数定義 ---
# 実行中のスクリプトのディレクトリを基準にパスを設定
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "best-2.pt"
NUTRITION_DATA_PATH = SCRIPT_DIR / "master_natrition.csv"
IMAGE_BASE_DIR = SCRIPT_DIR / "UECFOOD256"

# 栄養素名（英語→日本語）のマッピング辞書
NUTRITION_JP_MAP = {
    'energy_kcal': 'エネルギー (kcal)', 'protein_g': 'タンパク質 (g)', 'fat_g': '脂質 (g)',
    'carbohydrate_g': '炭水化物 (g)', 'calcium_mg': 'カルシウム (mg)', 'iron_mg': '鉄 (mg)',
    'vitamin_c_mg': 'ビタミンC (mg)', 'vitamin_b1_mg': 'ビタミンB1 (mg)', 'vitamin_b2_mg': 'ビタミンB2 (mg)',
    'fiber_g': '食物繊維 (g)', 'sodium_mg': 'ナトリウム (mg)'
}

# 1日の推奨摂取量 (成人男性の例)
DAILY_NEEDS = {
    'energy_kcal': 2650, 'protein_g': 65, 'fat_g': 73.6, 'carbohydrate_g': 378.1,
    'calcium_mg': 800, 'iron_mg': 7.5, 'vitamin_c_mg': 100, 'vitamin_b1_mg': 1.4,
    'vitamin_b2_mg': 1.6, 'fiber_g': 21, 'sodium_mg': 2362
}


# --- ヘルパー関数 ---

def find_random_image(directory: Path) -> str | None:
    """
    指定されたディレクトリからランダムな画像ファイルの絶対パスを文字列で返す。
    
    Args:
        directory (Path): 画像が格納されているディレクトリのPathオブジェクト。

    Returns:
        str | None: 見つかった画像ファイルの絶対パス。見つからない場合はNone。
    """
    if not directory.is_dir():
        return None
    
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in image_extensions]
    
    if not image_files:
        return None
        
    # ランダムに選んだファイルの絶対パスを文字列として返す
    return str(random.choice(image_files).resolve())

def recommend_foods(deficiency_data: dict, nutrition_df: pd.DataFrame, detected_ids: set, num_recommendations: int = 5) -> dict:
    """
    不足している栄養素を補う料理を推薦する。
    
    Args:
        deficiency_data (dict): 不足している栄養素のデータ。
        nutrition_df (pd.DataFrame): 全食品の栄養素データフレーム。
        detected_ids (set): 検出された食品のIDセット。
        num_recommendations (int): 各栄養素ごとのおすすめ件数。

    Returns:
        dict: 栄養素ごとのおすすめ料理データフレームを格納した辞書。
    """
    jp_to_eng_map = {v: k for k, v in NUTRITION_JP_MAP.items()}
    recommendations = {}
    
    # 不足分が多い栄養素トップ3を取得
    sorted_deficiencies = sorted(deficiency_data.items(), key=lambda item: item[1]['不足分'], reverse=True)
    
    for jp_nutrient, _ in sorted_deficiencies[:3]:
        eng_nutrient_col = jp_to_eng_map.get(jp_nutrient)
        
        if eng_nutrient_col and eng_nutrient_col in nutrition_df.columns:
            # 検出された料理を除外
            recommend_df = nutrition_df[~nutrition_df.index.isin(detected_ids)]
            
            # 目的の栄養素が豊富な上位N件を取得し、コピーを作成
            top_foods = recommend_df.sort_values(by=eng_nutrient_col, ascending=False).head(num_recommendations).copy()
            
            # 各料理に対応する画像パスを取得
            top_foods['image_path'] = top_foods.index.to_series().apply(
                lambda food_id: find_random_image(IMAGE_BASE_DIR / str(food_id))
            )
            
            # 結果を整形
            result_df = top_foods[['料理名', eng_nutrient_col, 'image_path']].copy()
            result_df.rename(columns={eng_nutrient_col: jp_nutrient}, inplace=True)
            recommendations[jp_nutrient] = result_df
            
    return recommendations

# --- データとモデルの読み込み（キャッシュ機能付き） ---

@st.cache_resource
def load_yolo_model(path="best-2.pt"):
    """YOLOモデルをロード"""
    if not path.is_file():
        st.error(f"モデルファイルが見つかりません: {path}")
        return None
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"モデル '{path.name}' の読み込みに失敗しました: {e}")
        return None

@st.cache_data
def load_nutrition_data(path: Path):
    """栄養素データベースをロードし、前処理を行う"""
    if not path.is_file():
        st.error(f"栄養素データベースが見つかりません: {path}")
        return None
    try:
        df = pd.read_csv(path)
        # 'food_name'列を'料理名'にリネーム
        df.rename(columns={'food_name': '料理名'}, inplace=True)
        # 数値列の()や-を0に置換して数値型に変換
        # '料理名'列は文字列なので処理から除外
        numeric_cols = [col for col in df.columns[4:] if col != '料理名']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[\(\)-]', '0', regex=True), errors='coerce').fillna(0)
        df.set_index('num', inplace=True)
        return df
    except Exception as e:
        st.error(f"CSVファイル '{path.name}' の読み込みまたは処理中にエラーが発生しました: {e}")
        return None

# --- Streamlit アプリケーションのメイン処理 ---
def main():
    st.title('🥗 食事分析AI')
    st.write('食事の写真をアップロードすると、含まれる栄養素を分析し、1日の摂取基準に足りない栄養素と、それを補うメニューをお知らせします。')

    # 必須ファイルの存在チェック
    if not IMAGE_BASE_DIR.is_dir():
        st.error(f"画像フォルダ '{IMAGE_BASE_DIR.name}' が見つかりません。アプリケーションと同じ階層に配置してください。")
        return

    model = load_yolo_model(MODEL_PATH)
    nutrition_df = load_nutrition_data(NUTRITION_DATA_PATH)

    if model is None or nutrition_df is None:
        st.warning("必要なファイルの読み込みに失敗したため、処理を続行できません。")
        return

    uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model(image) 
        
        detected_items_jp, detected_ids = [], []
        # '料理名'列を除いた栄養素列でSeriesを初期化
        nutrition_cols = [col for col in nutrition_df.columns if col != '料理名' and col not in ['uec256_name', 'food_id']]
        total_nutrition = pd.Series(0.0, index=nutrition_cols) 
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # 検出結果の処理
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                nutrition_id = class_id + 1 
                detected_ids.append(nutrition_id)
                
                # バウンディングボックスとラベルを描画
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f'{result.names[class_id]}'
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # 栄養素を合計
                if nutrition_id in nutrition_df.index:
                    detected_items_jp.append(nutrition_df.loc[nutrition_id, '料理名'])
                    total_nutrition += nutrition_df.loc[nutrition_id, nutrition_cols]

        st.subheader("📸 検出結果")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption='検出された料理', use_column_width=True)
        
        if detected_items_jp:
            st.write(f"検出された料理: **{', '.join(sorted(list(set(detected_items_jp))))}**")
            
            # 摂取栄養素の表示
            st.subheader("📊 この食事の栄養素")
            display_nutrition = total_nutrition.get(list(DAILY_NEEDS.keys()), pd.Series(0.0, index=list(DAILY_NEEDS.keys()))).copy()
            display_nutrition.rename(index=NUTRITION_JP_MAP, inplace=True)
            st.dataframe(display_nutrition.rename('摂取量').to_frame())

            # 不足栄養素の計算と表示
            st.subheader("💪 1日の目標に対する不足栄養素")
            deficiency_data = {}
            for key, daily_value in DAILY_NEEDS.items():
                meal_value = total_nutrition.get(key, 0)
                deficiency = daily_value - meal_value
                if deficiency > 0:
                    jp_key = NUTRITION_JP_MAP.get(key, key)
                    deficiency_data[jp_key] = {
                        "1日の目標": daily_value, "この食事の摂取量": f"{meal_value:.2f}", "不足分": f"{deficiency:.2f}"
                    }
            
            if deficiency_data:
                df_deficiency = pd.DataFrame.from_dict(deficiency_data, orient='index')
                st.warning("以下の栄養素が不足しています。")
                st.dataframe(df_deficiency)

                # おすすめメニューの取得と表示
                recommendations = recommend_foods(deficiency_data, nutrition_df, set(detected_ids))

                if recommendations:
                    st.subheader("💡 不足分を補うおすすめメニュー")
                    st.write("特に不足している栄養素を補うには、以下のような料理がおすすめです。")
                    
                    for nutrient, food_df in recommendations.items():
                        with st.expander(f"**「{nutrient}」**が豊富な料理TOP5", expanded=True):
                            for _, row in food_df.iterrows():
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    image_path = row.get('image_path')
                                    # パスが存在し、それがファイルであることを確認
                                    if image_path and Path(image_path).is_file():
                                        st.image(image_path, use_column_width=True)
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

if __name__ == '__main__':
    main()
