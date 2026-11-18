import os
import io
import base64
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Wajib agar Matplotlib berjalan di server tanpa GUI
import matplotlib.pyplot as plt
from PIL import Image
from google import genai
from django.conf import settings

MODEL_PATH = os.path.join(settings.BASE_DIR, "model_kos.joblib")
DATA_PATH = os.path.join(settings.BASE_DIR, "kos_madiun.csv")

model_ml = joblib.load(MODEL_PATH)
df_dataset = pd.read_csv(DATA_PATH)

def evaluasi_fuzzy_kos(estimasi_ml, harga_tawaran, jarak_km, rating):
    """
    Logika Fuzzy mengevaluasi 'Selisih Harga (%)' dan 'Kenyamanan (Jarak & Rating)'
    untuk menentukan kelayakan kos (skor 0 - 100).
    """
    selisih_persen = ((estimasi_ml - harga_tawaran) / estimasi_ml) * 100

    # 1. Fuzzifikasi Selisih Harga
    if selisih_persen >= 10:
        mu_murah, mu_wajar, mu_mahal = 1.0, 0.0, 0.0
    elif 0 <= selisih_persen < 10:
        mu_murah = selisih_persen / 10.0
        mu_wajar = 1.0 - mu_murah
        mu_mahal = 0.0
    elif -15 <= selisih_persen < 0:
        mu_murah = 0.0
        mu_wajar = (15.0 + selisih_persen) / 15.0
        mu_mahal = 1.0 - mu_wajar
    else:
        mu_murah, mu_wajar, mu_mahal = 0.0, 0.0, 1.0

    # 2. Fuzzifikasi Kenyamanan (Dekat kampus & Rating tinggi)
    skor_jarak = max(0.0, min(1.0, (5.0 - jarak_km) / 4.7))
    skor_rating = max(0.0, min(1.0, (rating - 3.0) / 2.0))
    skor_nyaman = (skor_jarak * 0.5) + (skor_rating * 0.5)

    # 3. Inferensi & Defuzzifikasi Weighted Average
    bobot_sangat_layak = mu_murah * (0.7 + 0.3 * skor_nyaman)
    bobot_wajar = mu_wajar * (0.6 + 0.2 * skor_nyaman)
    bobot_kemahalan = mu_mahal * 0.25

    total_mu = mu_murah + mu_wajar + mu_mahal
    skor_akhir = ((bobot_sangat_layak * 95) + (bobot_wajar * 70) + (bobot_kemahalan * 35)) / total_mu

    if skor_akhir >= 78:
        kategori = "Sangat Layak (Recommended)"
    elif skor_akhir >= 55:
        kategori = "Wajar / Sesuai Pasar"
    else:
        kategori = "Kemahalan (Overpriced)"

    return round(skor_akhir, 1), kategori

def buat_grafik_kos(nama_kos, ac, harga_tawaran, estimasi_ml):
    subset_ac = df_dataset[df_dataset["ac"] == ac]
    rata_pasar_ac = subset_ac["harga"].mean() if not subset_ac.empty else df_dataset["harga"].mean()

    label_ac = "Ber-AC" if ac == "ya" else "Non-AC"
    kategori = ["Harga Pemilik Kos", "Prediksi Wajar ML", f"Rata-rata Kos {label_ac}"]
    nilai = [harga_tawaran, estimasi_ml, rata_pasar_ac]
    warna = ["#ef4444" if harga_tawaran > estimasi_ml else "#10b981", "#3b82f6", "#64748b"]

    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar(kategori, nilai, color=warna, width=0.5)
    ax.set_ylabel("Harga Sewa (Ribu Rp / Bulan)")
    ax.set_title(f"Perbandingan Harga Sewa: {nama_kos}")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 8, f"Rp {yval:.0f} rb", ha="center", fontweight="bold")

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.seek(0)
    pil_image = Image.open(buf)
    return img_base64, pil_image

def jelaskan_grafik_gemini(pil_image, nama_kos, harga_tawaran, estimasi_ml, kategori_fuzzy, skor_fuzzy):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or "GANTI_DENGAN" in api_key:
        return "API Key Gemini belum diatur di file .env."

    try:
        client = genai.Client(api_key=api_key)
        prompt = (
            f"Anda adalah asisten pencari kos mahasiswa di Madiun. Jelaskan grafik perbandingan harga untuk '{nama_kos}' ini "
            f"secara singkat (maksimal 2 paragraf pendek). Data: Harga Tawaran = Rp {harga_tawaran:.0f} ribu/bulan, "
            f"Prediksi Wajar ML = Rp {estimasi_ml:.0f} ribu/bulan, Evaluasi Fuzzy = {kategori_fuzzy} (Skor {skor_fuzzy}/100). "
            f"Berikan saran praktis apakah kos ini layak diambil oleh mahasiswa."
        )
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=[pil_image, prompt]
        )
        return response.text
    except Exception as e:
        return f"Gagal memuat analisis AI: {e}"