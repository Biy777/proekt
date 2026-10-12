# Проект №14 — сравнение спутниковых снимков

## Возможности
- SIFT + Homography (RANSAC) для совмещения снимков.
- Gaussian Blur для подавления шума.
- Абсолютная разница пикселей и бинарная сегментация.
- Морфологическая фильтрация и поиск контуров.
- Streamlit-интерфейс, метрики и экспорт PNG.

## Запуск на Windows
```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```
Если PowerShell запрещает активацию окружения, можно не активировать его и запускать:
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

## Тестовые данные
`data/satellite_before.png` и `data/satellite_after.png` — синтетические изображения для демонстрации пайплайна. Это не реальные спутниковые фотографии. Для исследования реальных изменений загрузите два снимка одной территории с максимально похожими условиями съёмки.

## Ограничения
Пиксельная разница чувствительна к сезонности, облакам, теням и изменению освещения. Гомография предполагает, что геометрическое различие снимков можно приблизительно описать одной проективной трансформацией.
