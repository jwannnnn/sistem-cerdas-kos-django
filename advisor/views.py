import pandas as pd
from django.shortcuts import render
from .models import RiwayatKos
from .intelligent_engine import (
    model_ml,
    evaluasi_fuzzy_kos,
    buat_grafik_kos,
    jelaskan_grafik_gemini
)

def index(request):
    konteks = {}
    if request.method == "POST":
        nama_kos = request.POST.get("nama_kos")
        jarak_km = float(request.POST.get("jarak_km"))
        rating = float(request.POST.get("rating"))
        ac = request.POST.get("ac")
        harga_tawaran = float(request.POST.get("harga_tawaran"))

        # 1. Prediksi ML
        input_df = pd.DataFrame([{
            "jarak_km": jarak_km,
            "rating": rating,
            "ac_num": 1 if ac == "ya" else 0
        }])
        estimasi_ml = round(float(model_ml.predict(input_df)[0]), 1)

        # 2. Evaluasi Fuzzy
        skor_fuzzy, status_fuzzy = evaluasi_fuzzy_kos(estimasi_ml, harga_tawaran, jarak_km, rating)

        # 3. Grafik Matplotlib
        grafik_b64, pil_img = buat_grafik_kos(nama_kos, ac, harga_tawaran, estimasi_ml)

        # 4. Penjelasan Grafik Gemini AI
        analisis_ai = jelaskan_grafik_gemini(
            pil_img, nama_kos, harga_tawaran, estimasi_ml, status_fuzzy, skor_fuzzy
        )

        # 5. Simpan ke Database SQLite
        RiwayatKos.objects.create(
            nama_kos=nama_kos,
            jarak_km=jarak_km,
            rating=rating,
            ac=ac,
            harga_tawaran=harga_tawaran,
            estimasi_ml=estimasi_ml,
            status_fuzzy=status_fuzzy,
            skor_fuzzy=skor_fuzzy,
            analisis_ai=analisis_ai
        )

        konteks.update({
            "hasil": True,
            "nama_kos": nama_kos,
            "harga_tawaran": harga_tawaran,
            "estimasi_ml": estimasi_ml,
            "skor_fuzzy": skor_fuzzy,
            "status_fuzzy": status_fuzzy,
            "grafik_b64": grafik_b64,
            "analisis_ai": analisis_ai,
        })

    konteks["riwayat"] = RiwayatKos.objects.order_by("-created_at")[:5]
    return render(request, "advisor/index.html", konteks)