import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="Сравнение спутниковых снимков", page_icon="🛰️", layout="wide")

st.title("🛰️ Автоматическое сравнение спутниковых снимков")
st.caption("Проект №14 · Компьютерное зрение 1 · OpenCV + Streamlit")
st.info(
    "Загрузите снимки одной территории за разные даты. Для демонстрации в папке data "
    "есть синтетическая тестовая пара. Она имитирует спутниковую сцену, но не является "
    "реальными спутниковыми данными."
)

ROOT = Path(__file__).parent
DATA = ROOT / "data"

def read_uploaded_image(uploaded_file):
    """Преобразует загруженный файл в RGB-массив OpenCV."""
    pil_image = Image.open(uploaded_file).convert("RGB")
    return np.array(pil_image)

def load_sample(path):
    return np.array(Image.open(path).convert("RGB"))

def align_images(before_rgb, after_rgb):
    """
    Совмещает after с before с помощью SIFT + сопоставления дескрипторов
    + RANSAC-гомографии. При недостатке признаков использует исходное
    изображение после изменения размера и сообщает об этом пользователю.
    """
    h, w = before_rgb.shape[:2]
    after_resized = cv2.resize(after_rgb, (w, h), interpolation=cv2.INTER_AREA)

    before_gray = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after_resized, cv2.COLOR_RGB2GRAY)

    # CLAHE помогает сделать локальные детали более заметными при контрасте.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    before_eq = clahe.apply(before_gray)
    after_eq = clahe.apply(after_gray)

    sift = cv2.SIFT_create(nfeatures=3000)
    kp1, des1 = sift.detectAndCompute(before_eq, None)
    kp2, des2 = sift.detectAndCompute(after_eq, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return after_resized, 0, False, "Недостаточно ключевых точек; применено только изменение размера."

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    pairs = matcher.knnMatch(des2, des1, k=2)  # after -> before
    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 8:
        return after_resized, len(good), False, "Недостаточно надёжных совпадений; применено только изменение размера."

    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return after_resized, len(good), False, "Гомография не найдена; применено только изменение размера."

    aligned = cv2.warpPerspective(after_resized, H, (w, h), flags=cv2.INTER_LINEAR)
    inlier_count = int(inliers.sum()) if inliers is not None else 0
    return aligned, inlier_count, True, "Совмещение выполнено методом SIFT + Homography (RANSAC)."

def detect_changes(before_rgb, after_aligned_rgb, blur_size=5, threshold=28, min_area=120):
    """Фильтрация, разница яркости, бинарная сегментация и поиск контуров."""
    before_bgr = cv2.cvtColor(before_rgb, cv2.COLOR_RGB2BGR)
    after_bgr = cv2.cvtColor(after_aligned_rgb, cv2.COLOR_RGB2BGR)

    before_gray = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Фильтрация для снижения высокочастотного шума.
    before_blur = cv2.GaussianBlur(before_gray, (blur_size, blur_size), 0)
    after_blur = cv2.GaussianBlur(after_gray, (blur_size, blur_size), 0)

    # 2) Абсолютная разница между пикселями.
    diff = cv2.absdiff(before_blur, after_blur)

    # 3) Сегментация: пиксели выше порога считаются кандидатами на изменения.
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # 4) Морфология удаляет отдельные шумовые пиксели и соединяет близкие области.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area]

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, kept, -1, 255, thickness=cv2.FILLED)

    overlay = after_aligned_rgb.copy()
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    red = np.zeros_like(overlay_bgr)
    red[:, :] = (0, 0, 255)
    mask_bool = clean_mask > 0
    overlay_bgr[mask_bool] = cv2.addWeighted(
        overlay_bgr, 0.45, red, 0.55, 0
    )[mask_bool]
    overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    outlined = after_aligned_rgb.copy()
    outlined_bgr = cv2.cvtColor(outlined, cv2.COLOR_RGB2BGR)
    cv2.drawContours(outlined_bgr, kept, -1, (0, 0, 255), 2)
    outlined = cv2.cvtColor(outlined_bgr, cv2.COLOR_BGR2RGB)

    image_area = clean_mask.shape[0] * clean_mask.shape[1]
    changed_pixels = int(np.count_nonzero(clean_mask))
    percent = 100.0 * changed_pixels / image_area if image_area else 0.0

    return {
        "difference": diff,
        "mask": clean_mask,
        "overlay": overlay,
        "outlined": outlined,
        "contours": kept,
        "changed_pixels": changed_pixels,
        "changed_percent": percent,
    }

def as_png_download(rgb_array):
    buffer = io.BytesIO()
    Image.fromarray(rgb_array).save(buffer, format="PNG")
    return buffer.getvalue()

with st.sidebar:
    st.header("Настройки")
    source = st.radio("Источник изображений", ["Тестовая пара проекта", "Загрузить свои снимки"])
    threshold = st.slider("Порог разницы", min_value=5, max_value=80, value=28, step=1)
    min_area = st.slider("Минимальная площадь контура (пиксели)", min_value=20, max_value=1500, value=120, step=20)
    blur_size = st.select_slider("Размер Gaussian Blur", options=[3, 5, 7, 9], value=5)

if source == "Тестовая пара проекта":
    before_path = DATA / "satellite_before.png"
    after_path = DATA / "satellite_after.png"
    if not before_path.exists() or not after_path.exists():
        st.error("Не найдены тестовые изображения в папке data.")
        st.stop()
    before_rgb = load_sample(before_path)
    after_rgb = load_sample(after_path)
else:
    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Снимок раннего периода", type=["png", "jpg", "jpeg", "tif", "tiff"], key="before")
    with col2:
        after_file = st.file_uploader("Снимок позднего периода", type=["png", "jpg", "jpeg", "tif", "tiff"], key="after")
    if before_file is None or after_file is None:
        st.warning("Загрузите оба изображения, чтобы начать обработку.")
        st.stop()
    before_rgb = read_uploaded_image(before_file)
    after_rgb = read_uploaded_image(after_file)

if before_rgb.shape[:2] != after_rgb.shape[:2]:
    st.warning("Размеры снимков различаются. Второй снимок будет приведён к размеру первого.")

with st.spinner("Совмещаем изображения и ищем изменения..."):
    aligned, inlier_count, alignment_ok, alignment_message = align_images(before_rgb, after_rgb)
    result = detect_changes(before_rgb, aligned, blur_size=blur_size, threshold=threshold, min_area=min_area)

st.subheader("1. Исходные изображения")
c1, c2 = st.columns(2)
with c1:
    st.image(before_rgb, caption="Ранний период", use_container_width=True)
with c2:
    st.image(after_rgb, caption="Поздний период (до совмещения)", use_container_width=True)

st.subheader("2. Совмещение и промежуточные результаты")
st.caption(alignment_message)
st.caption(f"Количество внутренних согласованных соответствий RANSAC: {inlier_count}")
c1, c2, c3 = st.columns(3)
with c1:
    st.image(aligned, caption="Поздний снимок после совмещения", use_container_width=True)
with c2:
    st.image(result["difference"], caption="Абсолютная разница после Gaussian Blur", clamp=True, use_container_width=True)
with c3:
    st.image(result["mask"], caption="Бинарная маска изменений", clamp=True, use_container_width=True)

st.subheader("3. Итоговый результат")
c1, c2 = st.columns(2)
with c1:
    st.image(result["outlined"], caption=f"Обнаруженные области: {len(result['contours'])}", use_container_width=True)
with c2:
    st.image(result["overlay"], caption="Изменения выделены красным", use_container_width=True)

m1, m2, m3 = st.columns(3)
m1.metric("Найдено областей", len(result["contours"]))
m2.metric("Изменившиеся пиксели", f"{result['changed_pixels']:,}".replace(",", " "))
m3.metric("Площадь изменений", f"{result['changed_percent']:.2f}%")

st.download_button(
    "Скачать результат с контурами (PNG)",
    data=as_png_download(result["outlined"]),
    file_name="satellite_changes_contours.png",
    mime="image/png",
)
st.download_button(
    "Скачать маску изменений (PNG)",
    data=as_png_download(result["mask"]),
    file_name="satellite_changes_mask.png",
    mime="image/png",
)

with st.expander("Ограничения метода"):
    st.markdown("""
    - Метод обнаруживает различия пикселей, а не понимает семантически, что именно изменилось.
    - Тени, облака, сезонность, освещение и разные углы съёмки могут давать ложные срабатывания.
    - Homography хорошо работает при близких ракурсах и приблизительно плоской сцене; сильные параллакс и рельеф снижают качество.
    - Процент относится к пикселям тестового изображения, а не к географической площади в м²/км².
    - Для научного анализа реальные снимки обычно предварительно ортотрансформируют и приводят к единой геопривязке.
    """)
