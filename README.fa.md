# 🫀 سیستم پیش‌بینی ریسک بیماری قلبی

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0%2B-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi)
![Tests](https://img.shields.io/badge/تست‌ها-32_پاس-brightgreen)
![License](https://img.shields.io/badge/مجوز-MIT-yellow)

**یک سیستم یادگیری ماشین سطح تولید برای ارزیابی ریسک بیماری قلبی‌عروقی.**  
آموزش ۵ مدل +앙samble، توضیح پیش‌بینی‌ها با SHAP، و سرویس‌دهی نتایج از طریق REST API.

</div>

---

## ⚠️ سلب مسئولیت پزشکی

> این پروژه **صرفاً برای اهداف آموزشی و پژوهشی** است. به هیچ عنوان **نباید** جایگزین مشاوره پزشکی حرفه‌ای، تشخیص یا درمان شود. همیشه با یک متخصص بهداشت واجد شرایط مشورت کنید.

---

## ویژگی‌های پروژه

- **آموزش چند مدل**: رگرسیون لجستیک، جنگل تصادفی، XGBoost، LightGBM، SVM + Voting Ensemble
- **تنظیم هایپرپارامتر**: جستجوی تصادفی با اعتبارسنجی متقاطع k-fold
- **مدیریت عدم تعادل کلاس**: SMOTE فقط روی داده‌های آموزشی
- **مهندسی ویژگی**: ۹ ویژگی مشتق‌شده بر اساس دانش پزشکی
- **تفسیرپذیری**: مقادیر SHAP برای اهمیت سراسری و توضیحات هر بیمار
- **API تولیدی**: FastAPI با اعتبارسنجی Pydantic و مستندات OpenAPI
- **۳۲ تست خودکار** پوشش‌دهنده پیش‌پردازش، مدل‌ها، متریک‌ها و API
- **پشتیبانی Docker** برای استقرار کانتینری

---

## نتایج مدل‌ها

| مدل | دقت | صحت | فراخوانی | F1 | **AUC** |
|---|---|---|---|---|---|
| Logistic Regression | 0.880 | 0.891 | 0.891 | 0.891 | **0.963** |
| Ensemble (Top 3) | 0.875 | 0.870 | 0.909 | 0.889 | 0.960 |
| SVM | 0.860 | 0.860 | 0.891 | 0.875 | 0.954 |
| Random Forest | 0.850 | 0.851 | 0.882 | 0.866 | 0.951 |
| LightGBM | 0.895 | 0.916 | 0.891 | 0.903 | 0.947 |
| XGBoost | 0.875 | 0.883 | 0.891 | 0.887 | 0.947 |

اعتبارسنجی متقاطع AUC: **0.948 ± 0.013** (5-fold)

---

## شروع سریع

### پیش‌نیازها

- Python 3.10 یا بالاتر
- pip

### ۱. کلون کردن مخزن

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-predictor.git
cd heart-disease-predictor
```

### ۲. ساخت محیط مجازی

```bash
python -m venv venv

# لینوکس / مک
source venv/bin/activate

# ویندوز
venv\Scripts\activate
```

### ۳. نصب وابستگی‌ها

```bash
pip install -r requirements.txt
```

### ۴. اجرای pipeline کامل

```bash
# Pipeline کامل با تنظیم هایپرپارامتر (توصیه‌شده، ~۵ تا ۱۰ دقیقه)
python main.py

# حالت سریع — بدون تنظیم، مناسب برای تست (~۳۰ ثانیه)
python main.py --no-tune
```

این دستور:
1. دیتاست را تولید می‌کند (داده سینتتیک مبتنی بر UCI، 1000 بیمار)
2. پیش‌پردازش و مهندسی ویژگی انجام می‌دهد
3. تمام ۶ مدل را آموزش می‌دهد و ارزیابی می‌کند
4. توضیحات SHAP تولید می‌کند
5. همه نمودارها را در `reports/figures/` ذخیره می‌کند
6. مدل‌های آموزش‌دیده را در `models/saved/` ذخیره می‌کند

### ۵. راه‌اندازی سرور API

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

به **http://localhost:8000/docs** بروید تا رابط Swagger تعاملی را ببینید.

### ۶. اجرای تست‌ها

```bash
pytest tests/ -v
```

---

## معماری پروژه

```
heart-disease-predictor/
│
├── main.py                        # هماهنگ‌کننده pipeline
├── requirements.txt               # وابستگی‌های Python
├── .gitignore
│
├── src/
│   ├── data/
│   │   └── preprocess.py          # پاک‌سازی، مهندسی ویژگی، تقسیم داده، SMOTE
│   ├── models/
│   │   ├── train.py               # تنظیم مدل، آموزش، CV، ارزیابی، ensemble
│   │   └── explain.py             # مقادیر SHAP و نمودارهای تفسیرپذیری
│   ├── visualization/
│   │   └── plots.py               # EDA، منحنی ROC، ماتریس confusion
│   └── api/
│       └── app.py                 # برنامه FastAPI با endpoint های پیش‌بینی
│
├── tests/
│   └── test_pipeline.py           # ۳۲ تست pytest
│
├── data/
│   ├── raw/                       # CSV خام (در اولین اجرا تولید می‌شود)
│   └── processed/                 # CSV با ویژگی‌های مهندسی‌شده
│
├── models/
│   └── saved/                     # مدل‌های سریالایزشده، scaler، متادیتا
│
├── reports/
│   └── figures/                   # تمام نمودارهای تولیدشده (PNG)
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## مهندسی ویژگی‌ها

علاوه بر ۱۳ ویژگی اصلی UCI، ۹ ویژگی مبتنی بر دانش پزشکی استخراج می‌شود:

| ویژگی | توضیح | پایه پزشکی |
|---|---|---|
| `hr_age_ratio` | حداکثر ضربان ÷ سن | ظرفیت ضربان قلب تنظیم‌شده با سن |
| `expected_max_hr` | 220 − سن | فرمول استاندارد کاردیولوژی |
| `hr_reserve_pct` | (حداکثر HR / انتظاری) × 100 | درصد ذخیره ضربان قلب استفاده‌شده |
| `chol_age_interaction` | کلسترول × سن / 1000 | ریسک مرکب قلبی‌عروقی |
| `hypertension_risk` | فشار ≥ 130 mmHg | آستانه فشار خون ACC/AHA |
| `st_depression_severe` | افسردگی ST ≥ 2.0 | نشانگر ایسکمی قابل توجه |
| `multi_vessel` | تعداد رگ ≥ 2 | بیماری چند رگ کرونری |
| `clinical_risk_score` | ترکیب وزن‌دار ۸ عامل | طبقه‌بندی ریسک بالینی |
| `cp_risk` | درد قفسه سینه بازنگاری‌شده | آسیمپتوماتیک = بالاترین ریسک |

---

## استفاده از API

### پیش‌بینی تک بیمار

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 60,
    "sex": 1,
    "cp": 0,
    "trestbps": 150,
    "chol": 280,
    "fbs": 0,
    "restecg": 1,
    "thalach": 130,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 2,
    "thal": 3
  }'
```

**پاسخ:**

```json
{
  "prediction": 1,
  "probability": 0.8734,
  "risk_level": "HIGH",
  "risk_description": "ریسک بالای بیماری قلبی. توصیه می‌شود فوراً به متخصص مراجعه کنید.",
  "confidence": "Very High",
  "key_risk_factors": [
    "سن بالای ۵۵ سال (۶۰ ساله)",
    "جنس مذکر (ریسک پایه بالاتر)",
    "الگوی درد قفسه سینه آسیمپتوماتیک",
    "فشار خون بالا (150 mmHg)",
    "آنژین ناشی از ورزش",
    "افسردگی ST قابل توجه (2.5)",
    "بیماری چند رگ (2 رگ)",
    "نقص تالاسمی برگشت‌پذیر"
  ],
  "recommendations": [
    "🚨 مراجعه فوری به متخصص قلب توصیه می‌شود",
    "تست استرس و تصویربرداری کرونری را در نظر بگیرید",
    "داروهای فعلی را با پزشک مرور کنید",
    "تغییرات سبک زندگی: ترک سیگار، رژیم غذایی، ورزش",
    "پایش منظم فشار خون و کلسترول"
  ],
  "model_version": "Logistic Regression",
  "timestamp": "2026-05-29T12:00:00"
}
```

### پیش‌بینی دسته‌ای

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"patients": [ {...}, {...} ]}'
```

### همه endpoint ها

| متد | مسیر | توضیح |
|---|---|---|
| GET | `/` | اطلاعات API |
| GET | `/health` | بررسی سلامت سرویس |
| GET | `/docs` | رابط Swagger تعاملی |
| GET | `/model/info` | متادیتای مدل |
| GET | `/features` | توضیحات ویژگی‌ها |
| POST | `/predict` | پیش‌بینی یک بیمار |
| POST | `/predict/batch` | پیش‌بینی دسته‌ای (تا ۱۰۰) |

---

## استقرار با Docker

```bash
cd docker
docker-compose up --build
# API در http://localhost:8000 در دسترس خواهد بود
```

---

## خروجی‌های تولیدشده

بعد از اجرای pipeline، فایل‌های زیر ساخته می‌شوند:

```
reports/figures/
├── eda_overview.png              # داشبورد EDA با ۸ نمودار
├── roc_comparison.png            # منحنی‌های ROC + نمودار مقایسه متریک
├── confusion_matrices.png        # ماتریس confusion برای همه مدل‌ها
├── shap_feature_importance.png   # ۱۵ ویژگی برتر بر اساس |SHAP| میانگین
└── shap_summary.png              # نمودار beeswarm برای SHAP

models/saved/
├── best_model.pkl                # بهترین مدل سریالایزشده
├── scaler.pkl                    # StandardScaler تنظیم‌شده
├── ensemble.pkl                  # Voting Ensemble
├── results.json                  # تمام متریک‌ها برای همه مدل‌ها
└── metadata.json                 # متادیتای آموزش
```

---

## دیتاست

بر اساس **دیتاست بیماری قلبی UCI** (Cleveland Clinic Foundation).

- **منبع اصلی**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **۱۳ ویژگی بالینی** + ۱ هدف (دودویی: بیماری / بدون بیماری)
- **۱۰۰۰ بیمار** (تولید سینتتیک با حفظ توزیع‌های آماری UCI)
- **~۵۴٪ شیوع بیماری** (مطابق دیتاست اصلی)

---

## ساختار تست‌ها

```
tests/test_pipeline.py
├── TestDataCleaning          (6 تست) — پاک‌سازی داده، مقادیر گمشده، انواع داده
├── TestFeatureEngineering    (7 تست) — ویژگی‌های جدید، فرمول‌ها، صحت مقادیر
├── TestDatasetPreparation    (5 تست) — تقسیم‌بندی، scaler، اندازه‌ها
├── TestModelPipeline         (4 تست) — آموزش مدل، AUC، احتمالات
├── TestMetrics               (3 تست) — محاسبه متریک، کلیدها، بازه مقادیر
├── TestPatientDataValidation (5 تست) — اعتبارسنجی API، مهندسی ویژگی
└── TestEndToEnd              (1 تست) — pipeline کامل یکپارچه
```

---


## مشارکت

Pull request ها خوش‌آمد هستند. برای تغییرات عمده، لطفاً ابتدا یک issue باز کنید تا درباره آنچه می‌خواهید تغییر دهید بحث شود. قبل از ارسال، مطمئن شوید همه تست‌ها پاس می‌شوند:

```bash
pytest tests/ -v --tb=short
```
