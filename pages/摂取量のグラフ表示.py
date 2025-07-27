import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os

# --- フォントパスを指定（適宜パス変更）---
FONT_PATH = "NotoSansJP-Regular.ttf"
# フォントプロパティを作成
if os.path.exists(FONT_PATH):
    jp_font = fm.FontProperties(fname=FONT_PATH)
else:
    st.warning("NotoSansJP-Regular.ttfが見つかりません。グラフの日本語表示が崩れる可能性があります。")
    jp_font = None

# --- サンプルデータ ---
labels = ['エネルギー', 'たんぱく質', '脂質', '炭水化物', 'カルシウム', '鉄', 'ビタミンC', 'ビタミンB1', 'ビタミンB2', '食物繊維', 'ナトリウム']
units  = ['kcal', 'g', 'g', 'g', 'mg', 'mg', 'mg', 'mg', 'mg', 'g', 'mg']

breakfast = [403, 14, 12, 52, 49, 1.4, 7, 0.2, 0.1, 3, 800]
lunch     = [600, 21, 18, 70, 180, 2.1, 20, 0.4, 0.3, 7, 900]
dinner    = [650, 20, 23, 75, 210, 2.3, 30, 0.6, 0.5, 6, 600]
targets   = [2650, 65, 73.6, 378.1, 800, 7.5, 100, 1.4, 1.6, 21, 2362]

data = np.array([breakfast, lunch, dinner])  # (3, 11)
total = np.sum(data, axis=0)  # 1日合計

# --- グラフ描画 ---
fig, ax = plt.subplots(figsize=(6, 8))

bar_width = 0.6
x = np.arange(len(labels))

# 積み上げ棒グラフ（縦）
p1 = ax.bar(x, breakfast, bar_width, label='朝食', color='#94e0e4')
p2 = ax.bar(x, lunch, bar_width, bottom=breakfast, label='昼食', color='#e793ac')
p3 = ax.bar(x, dinner, bar_width, bottom=np.array(breakfast)+np.array(lunch), label='夕食', color='#53e0e4')

# 目標値に赤い横線（各項目ごとに1本ずつ）
for i, target in enumerate(targets):
    ax.hlines(target, i - bar_width/2, i + bar_width/2, color='red', linestyle='dashed', linewidth=2)

# 合計値/単位ラベル
for i, (t, u) in enumerate(zip(total, units)):
    ax.text(i, t + max(targets)*0.01, f'{t:.0f} {u}', ha='center', va='bottom', fontsize=10, color='black', fontproperties=jp_font)

# 項目名
ax.set_xticks(x)
if jp_font:
    ax.set_xticklabels(labels, fontproperties=jp_font, fontsize=12, rotation=30, ha='right')
else:
    ax.set_xticklabels(labels, fontsize=12, rotation=30, ha='right')
ax.set_ylabel('摂取量', fontproperties=jp_font)
ax.legend(prop=jp_font)
ax.set_title('1日の栄養素摂取状況（目標値に赤線）', fontproperties=jp_font)

fig.tight_layout()
st.pyplot(fig)