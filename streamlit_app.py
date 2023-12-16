import streamlit as st
import cv2
import numpy as np
from skimage import measure
from PIL import Image

# 画像を処理して細菌のコロニーを数える関数
def count_colonies(image):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # 画像を二値画像に閾値処理
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # 小さなノイズを取り除くために膨張と収縮を行う
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)

    # 二値画像に対して連結成分の解析を実行
    labels = measure.label(thresh, connectivity=2, background=0)

    # マスク領域の輪郭を見つける
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # 処理済みの画像を作成（元の画像をNumPy配列に変換）
    processed_image = np.array(image).copy()

    # コロニーの周りに赤い円を描画
    for c in cnts:
        ((x, y), r) = cv2.minEnclosingCircle(c)
        cv2.circle(processed_image, (int(x), int(y)), int(r), (255, 0, 0), 2)
    
    return processed_image, len(cnts)

# Streamlitアプリケーションの開始
st.title('細菌コロニーカウンター')

# ユーザーが自分の画像をアップロードできるようにする
uploaded_file = st.file_uploader("細菌コロニーが写った画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# ユーザーが画像をアップロードした場合に表示して処理する
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # PILイメージをRGBモードで開く
    st.image(image, caption='アップロードされた画像', use_column_width=True)
    
    # 画像を処理してコロニーの数を数える
    processed_image, colony_count = count_colonies(image)
    
    # 処理された画像とコロニーの数を表示
    st.image(processed_image, caption='コロニーを数えた処理画像', use_column_width=True)
    st.write(f'検出されたコロニーの数: {colony_count}')
