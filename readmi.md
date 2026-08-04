# Klasifikasi Citra Bunga dengan Multi-Layer Perceptron (MLP)
Proyek Ujian Akhir Semester (UAS) mata kuliah **Pengolahan Citra Digital (SIF202)**.
Sistem ini mengklasifikasikan citra bunga ke dalam **5 kelas** (Daisy, Dandelion, Roses, Sunflowers, Tulips) menggunakan algoritma **Multi-Layer Perceptron (MLP)** dari scikit-learn.

| **Nama**           |Riana                          |
| **NIM**            | 24146012                           |
| **Mata Kuliah**    | Pengolahan Citra Digital (SIF202).  |
| **Tahun Ajaran**   | Genap 2025/2026                     |
| **Dosen Pengampu** | Teuku Rizky Noviandy, S.Kom., M.Kom.|
| **Prodi**          | Sistem Informasi                    |
| **Universitas**    | Abulyatama Aceh                     |

---


## Tentang Proyek
Proyek ini membangun model *machine learning* yang mampu mengenali jenis bunga dari sebuah foto/citra secara otomatis. Prosesnya dimulai dari membaca ribuan foto bunga, mengubahnya menjadi format yang bisa "dibaca" oleh model, melatih model untuk mengenali pola tiap jenis bunga, lalu mengujinya untuk melihat seberapa akurat model dalam menebak jenis bunga yang belum pernah dilihat sebelumnya.

Semua tahapan (mulai dari eksplorasi data sampai evaluasi hasil) ada di dalam satu notebook:
**`UAS_PCD_24146012_Riana.ipynb`**


## Dataset
Dataset **Flowers Recognition** berisi 3.670 citra bunga yang terbagi ke dalam 5 kelas, disimpan pada folder `dataset/flower_photos/`:

| Kelas      | Jumlah Citra |
| Daisy      | 633          |
| Dandelion  | 898          |
| Roses      | 641          |
| Sunflowers | 699          |
| Tulips     | 799          |
| **Total**  | **3.670** |

Struktur folder dataset harus seperti berikut agar notebook dapat menemukannya secara otomatis:

```
dataset/
└── flower_photos/
    ├── daisy/
    ├── dandelion/
    ├── roses/
    ├── sunflowers/
    └── tulips/
```


## Alur Kerja (Pipeline)
Secara sederhana, begini cara kerja sistem ini, langkah demi langkah:

1. **Eksplorasi Data (EDA)** — menghitung jumlah citra tiap kelas dan menampilkan contoh gambarnya.
2. **Preprocessing citra**
   - *Resize* semua citra menjadi ukuran seragam **64x64 piksel**.
   - Konversi warna dari BGR (format bawaan OpenCV) ke RGB.
   - *Normalisasi* nilai piksel dari rentang 0–255 menjadi 0–1.
3. **Ekstraksi fitur (flatten)** — citra 3 dimensi (64x64x3) diratakan menjadi vektor 1 dimensi berisi **12.288 fitur**, karena MLP hanya menerima input berbentuk vektor.
4. **Encoding label** — nama kelas (teks) diubah menjadi angka menggunakan `LabelEncoder`.
5. **Split data** — 80% untuk data latih, 20% untuk data uji, dengan `random_state = NIM (24146012)` agar hasilnya konsisten setiap kali dijalankan.
6. **Feature scaling** — fitur distandarisasi menggunakan `StandardScaler` (hanya *fit* pada data latih, untuk mencegah kebocoran data).
7. **Melatih model MLP** — menggunakan `MLPClassifier` dengan 2 hidden layer (256 dan 128 neuron).
8. **Evaluasi model** — dihitung akurasi, precision, recall, F1-score, dan confusion matrix.
9. **Visualisasi hasil prediksi** — menampilkan contoh gambar asli berdampingan dengan label prediksi model.


## Konfigurasi Model
```python
MLPClassifier(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=64,
    max_iter=100,
    random_state=24146012,   # NIM
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
)
```


## Hasil
Model diuji menggunakan 734 citra data uji (20% dari total dataset) dan memperoleh:

- **Akurasi keseluruhan: 48,91%**

| Kelas      | Precision  | Recall     | F1-Score   |
| Daisy      | 0.4068     | 0.3780     | 0.3918     |
| Dandelion  | 0.4735     | 0.5978     | 0.5284     |
| Roses      | 0.4214     | 0.4609     | 0.4403     |
| Sunflowers | **0.6471** | **0.6286** | **0.6377** |
| Tulips     | 0.5000     | 0.3563     | 0.4161     |

**Poin penting dari hasil:**
- Kelas **Sunflowers** paling mudah dikenali oleh model (F1-score tertinggi).
- Kelas **Daisy** dan **Tulips** paling sering tertukar dengan kelas lain, kemungkinan karena kemiripan warna/bentuk pada citra mentah.
- Akurasi yang belum terlalu tinggi (~49%) wajar terjadi karena model hanya menggunakan piksel mentah sebagai fitur, tanpa ekstraksi fitur khusus (seperti HOG atau fitur dari CNN).


## Struktur Proyek
```
├── UAS_PCD_24146012_Riana.ipynb          # Notebook utama (seluruh kode & hasil)
├── dataset/
│   └── flower_photos/                       # Dataset citra bunga (5 subfolder kelas)
├── Laporan_UAS_PCD_24146012_Riana.docx   # Laporan lengkap (Word)
└── README.md                                # Dokumentasi proyek (file ini)
```


## Cara Menjalankan
1. Pastikan Python 3 sudah terpasang, lalu install pustaka yang dibutuhkan:
   ```bash
   pip install numpy pandas opencv-python matplotlib scikit-learn
   ```
2. Letakkan folder `dataset/flower_photos` sejajar dengan file notebook.
3. Buka `UAS_PCD_24146012_Riana.ipynb` menggunakan Jupyter Notebook, JupyterLab, atau VS Code.
4. Jalankan seluruh sel (*Run All*) secara berurutan dari atas ke bawah.


## Pustaka yang Digunakan
| Pustaka        | Fungsi                                    |
| OpenCV (`cv2`) | Membaca, resize, dan konversi warna citra |
| NumPy          | Operasi numerik dan array                 |
| Pandas         | Pengolahan data tabular                   |
| Matplotlib     | Visualisasi grafik dan gambar             |
| scikit-learn   | Split data, scaling, model MLP, evaluasi  |


## Saran Pengembangan
- Menggunakan ekstraksi fitur yang lebih baik (HOG, color histogram, atau fitur dari CNN pre-trained).
- Menambahkan augmentasi data untuk memperbanyak variasi data latih.
- Melakukan tuning hyperparameter (jumlah neuron, learning rate, dll.) dengan Grid Search/Random Search.
- Mencoba arsitektur Convolutional Neural Network (CNN) yang umumnya lebih unggul untuk klasifikasi citra.


## Lisensi & Catatan
Proyek ini dibuat untuk keperluan akademik (Ujian Akhir Semester) mata kuliah Pengolahan Citra Digital. Dataset **Flowers Recognition** bersumber dari domain publik dan digunakan hanya untuk keperluan pembelajaran.
