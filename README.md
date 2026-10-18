# Eksperimen_SML_siswa

Repositori eksperimen & preprocessing untuk AI4I 2020 Predictive Maintenance.

```
.github/workflows/preprocessing.yml     # preprocessing otomatis saat data berubah
ai4i2020_raw/ai4i2020.csv               # taruh dataset mentah di sini (10.000 baris)
preprocessing/
├── Eksperimen_siswa.ipynb              # eksplorasi + preprocessing (template MSML)
├── automate_siswa.py                   # versi otomatis dari notebook
└── ai4i2020_preprocessing/             # hasil: ai4i2020_preprocessing.csv + scaler.joblib
```

```bash
pip install pandas scikit-learn joblib
python preprocessing/automate_siswa.py --input ai4i2020_raw/ai4i2020.csv \
       --output-dir preprocessing/ai4i2020_preprocessing
```
