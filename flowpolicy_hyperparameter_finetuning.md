# Hyperparameter Fine-Tuning FlowPolicy
> Panduan konfigurasi dan eksplorasi hyperparameter untuk implementasi **State-Based FlowPolicy** pada lingkungan **Franka Kitchen** (Long Horizon Manipulation Task)

---

## Daftar Isi
1. [Konfigurasi Default (Baseline)](#1-konfigurasi-default-baseline)
2. [Rentang Fine-Tuning](#2-rentang-fine-tuning)
3. [Deskripsi Setiap Hyperparameter](#3-deskripsi-setiap-hyperparameter)
4. [Strategi Eksplorasi: Random Search](#4-strategi-eksplorasi-random-search)
5. [Metrik Evaluasi](#5-metrik-evaluasi)
6. [Prosedur Eksperimen](#6-prosedur-eksperimen)
   - [6.5 Early Stopping](#65-early-stopping)
7. [Catatan Teknis dari Kode (Hydra Config)](#7-catatan-teknis-dari-kode-hydra-config)

---

## 1. Konfigurasi Default (Baseline)

Konfigurasi awal mengacu pada penelitian orisinal FlowPolicy (Q. Zhang et al., 2025) yang sebelumnya dioptimasi untuk **short horizon task** berbasis point cloud.

| Hyperparameter          | Nilai Default | Keterangan                          |
|-------------------------|---------------|-------------------------------------|
| `epoch`                 | 3000          | Dari `training.num_epochs`          |
| `learning_rate`         | 1e-4          | AdamW optimizer                     |
| `batch_size`            | 128           | Train & validation                  |
| `seed`                  | 0, 42, 101    | Tiga seed eksperimen (lihat §4)     |
| `hidden_dim`            | 512           | Jumlah hidden dimension MLP         |
| `time_embedding_dim`    | 256           | Dimensi time embedding              |
| `num_segments` (K)      | 2             | Jumlah segmen Consistency FM        |
| `epsilon` (eps)         | 1e-2          | Dipakai di training & inferensi     |
| `delta_t`               | 1e-2          | Granularitas langkah waktu          |
| `action_horizon`        | 4             | Panjang sekuens aksi per inferensi  |
| `observation_horizon`   | 2             | Jumlah frame observasi historis     |

> **Catatan:** Nilai `num_segments` dan parameter Consistency FM lain (boundary, alpha) berasal dari `config/flowpolicy.yaml`. Pastikan override lewat CLI jika diperlukan.

---

## 2. Rentang Fine-Tuning

Rentang eksplorasi ditetapkan secara simetris di sekitar nilai default untuk memastikan landasan empiris yang valid.

| Hyperparameter          | Rentang Eksplorasi              | Tipe     |
|-------------------------|---------------------------------|----------|
| `epoch`                 | [500, 1000, 3000, 5000]         | Diskrit  |
| `learning_rate`         | [1e-3, 5e-4, 1e-4, 1e-5]       | Kontinu  |
| `batch_size`            | [64, 128, 256, 512]             | Diskrit  |
| `seed`                  | [0, 42, 101] (tidak di-tune)    | Fixed    |
| `hidden_dim`            | [128, 256, 512, 1024]           | Diskrit  |
| `time_embedding_dim`    | [128, 256, 512, 1024]           | Diskrit  |
| `num_segments` (K)      | [1, 2, 3, 4]                    | Diskrit  |
| `epsilon`               | [1e-4, 1e-3, 1e-2, 1]          | Kontinu  |
| `delta_t`               | [1e-4, 1e-3, 1e-2, 1]          | Kontinu  |
| `action_horizon`        | [2, 4, 6, 8]                    | Diskrit  |
| `observation_horizon`   | [4, 6, 8, 16]                   | Diskrit  |

---

## 3. Deskripsi Setiap Hyperparameter

### 3.1 Kelompok Transformasi Distribusi (Consistency FM)

#### `num_segments` (K)
- Menentukan berapa kali medan vektor dievaluasi selama transformasi distribusi dari noise menuju distribusi aksi target.
- **K besar** → lintasan lebih halus dan akurat, tetapi **inference latency meningkat linear**.
- **K kecil** → latensi rendah, tetapi risiko lintasan transformasi kurang presisi.
- Pada inferensi: total waktu ∝ K (setiap segmen = 1 forward pass jaringan).

```
Lat = Σ (t_akhir_forward_pass_i - t_awal_forward_pass_i), i = 1..K
```

#### `delta_t` (Δt)
- Granularitas interval waktu antara dua titik evaluasi dalam satu segmen.
- **Δt terlalu besar** → lompatan transformasi kasar, konsistensi antarsegmen sulit dipertahankan, training tidak konvergen.
- **Δt terlalu kecil** → gradien tidak signifikan, model butuh waktu sangat lama untuk belajar.
- Nilai yang direkomendasikan untuk eksplorasi awal: `1e-2`.

#### `epsilon` (ε)
- Batas waktu minimum pada sampling timestep `t` saat training dan inferensi.
- Dipakai di `compute_loss` maupun `predict_action`.
- **Pada inferensi:** hardcoded menggunakan nilai `self.eps` (sama dengan training).

#### `num_inference_step`
- Jumlah langkah inferensi pada `ConsistencyFM` (`sample_N`).
- Default: `1` (single-step inference).
- Meningkatkan nilai ini meningkatkan kualitas aksi namun menambah latensi.

---

### 3.2 Kelompok Optimasi Model

#### `learning_rate`
- Laju pembelajaran AdamW optimizer.
- Terlalu besar → training tidak stabil.
- Terlalu kecil → konvergensi lambat.
- Scheduler: **cosine** dengan warmup 500 steps (`num_cycles=0.5`).

#### `batch_size`
- Jumlah sampel per iterasi training.
- Batch kecil → update lebih sering, gradient noise tinggi.
- Batch besar → estimasi gradient lebih stabil, kebutuhan VRAM meningkat.
- Perhatikan batas VRAM: GTX 1080 (8 GB).

#### `epoch`
- Jumlah total epoch pelatihan (batas atas / budget maksimum).
- Dataset kecil (19 episode) → risiko overfitting pada epoch tinggi.
- Nilai ini berfungsi sebagai **batas atas**; training dapat berhenti lebih awal jika kriteria early stopping terpenuhi (lihat §6.5).
- Early stopping **tidak aktif sebelum epoch 1000** agar cosine LR scheduler dan flow matching memiliki waktu cukup untuk konvergen.

#### `hidden_dim`
- Ukuran layer tersembunyi pada Multi Layer Perceptron (MLP).
- MLP menerima input: `[t, a_t, s]` dengan total dimensi `1 + 9 + d = 10 + d`.
- **Terlalu kecil** → model tidak mampu mempelajari dinamika kompleks.
- **Terlalu besar** → risiko overfitting, terutama pada dataset kecil.

#### `time_embedding_dim`
- Dimensi representasi embedding untuk variabel waktu `t`.
- Mempengaruhi kemampuan model dalam membedakan tahap transformasi.

---

### 3.3 Kelompok Horizon

#### `action_horizon`
- Jumlah langkah aksi yang diprediksi model dalam satu inferensi.
- **Besar** → robot bergerak lebih "terencana", tetapi aksi awal mungkin sudah stale saat dieksekusi.
- **Kecil** → respons lebih reaktif terhadap lingkungan.

#### `observation_horizon`
- Jumlah frame observasi historis yang digunakan sebagai konteks.
- **Besar** → konteks temporal lebih kaya, tetapi meningkatkan dimensi input.
- **Kecil** → input lebih ringkas, risiko kehilangan konteks temporal penting.

> **Hubungan dengan `horizon` (panjang window trajectory):**
> ```
> horizon = 4 * ((max(n_obs_steps + n_action_steps - 1, 4) + 3) // 4)
> ```
> Contoh default: `max(2+4-1, 4) = 5` → `(5+3)//4 = 2` → `4*2 = 8`

---

## 4. Strategi Eksplorasi: Random Search

Penelitian ini menggunakan **Random Search** (Bergstra & Bengio, 2012) karena:
- Lebih efisien dari Grid Search untuk ruang hyperparameter berdimensi tinggi.
- Cocok ketika hanya sebagian kecil hyperparameter yang dominan memengaruhi performa.
- Tidak membuat asumsi struktural tertentu tentang bentuk ruang performa.

### Alur Eksperimen Lengkap (Baseline → Random Search)

```
Fase 1 — Baseline (hyperparameter default §1)
  Untuk setiap seed s ∈ {0, 42, 101}:
    1. Latih FlowPolicy dengan konfigurasi baseline + early stopping (§6.5)
    2. Eval 50 episode × 3 seed eval {0, 42, 101}
    3. Catat ke baseline_results.jsonl

Fase 2 — Random Search
  Untuk setiap trial:
    1. Sampel acak satu kombinasi hyperparameter (Tabel 2)
    2. Untuk setiap seed s ∈ {0, 42, 101}:
         Latih dari awal + early stopping
    3. Eval setiap checkpoint: 50 episode × 3 seed eval
    4. Agregasi mean ± std; catat ke results.jsonl
    5. Update best_config.json menurut trade_off_mean

Fase 3 — Visualisasi
  python scripts/kitchen_plot_results.py
  └── Scatter success_rate vs latency, trade-off vs val loss, subtask SR, dll.
```

Jalankan otomatis:
```bash
bash scripts/kitchen_run_experiment.sh 10 0   # baseline + 10 trial search + plots
```

### Pengendalian Seed
Tiga seed **`[0, 42, 101]`** dipakai untuk:
- Inisialisasi bobot awal neural network (training)
- Pengacakan data batch saat training
- Sampling noise pada inferensi
- Reset state awal lingkungan simulasi (eval)

Hasil dilaporkan sebagai **rata-rata ± simpangan baku** dari ketiga seed (training dan/atau eval, sesuai fase).

---

## 5. Metrik Evaluasi

### 5.1 Success Rate

#### Success Rate Total
```
success_rate = (N_success / N_total) × 100%
```

#### Success Rate Per Sub-Tugas (k ∈ {1,2,3,4})
```
success_rate_k = (N_success_k / N_total) × 100%
```

| k | Sub-Tugas yang Diselesaikan Secara Berurutan                          |
|---|-----------------------------------------------------------------------|
| 1 | Membuka pintu microwave                                               |
| 2 | Membuka microwave + memutar knob lampu kompor                         |
| 3 | Membuka microwave + knob + menaruh teko di kompor                     |
| 4 | Keempat sub-tugas lengkap (+ membuka slide cabinet)                   |

- Batas maksimum: **280 langkah per episode**
- Keberhasilan bersifat **biner** berdasarkan threshold konfigurasi objek

### 5.2 Inference Latency

```
Lat = Σ (t_akhir_forward_pass_i - t_awal_forward_pass_i), i = 1..K
```

- Latency berskala **linear** terhadap `K` (jumlah segmen)
- Gunakan **dummy pass** (GPU warm-up) sebelum pengukuran untuk menghilangkan bias CUDA initialization

### 5.3 Trade-Off Score

```
trade_off = success_rate / Lat
```

Nilai trade-off **lebih tinggi** = keseimbangan performa dan efisiensi komputasi **lebih baik**.

---

## 6. Prosedur Eksperimen

### 6.0 Urutan Eksperimen

| Langkah | Perintah | Early stopping |
|---------|----------|----------------|
| Preprocess | `bash scripts/preprocess_kitchen.sh` | — |
| Baseline × 3 seed | `kitchen_run_experiment.sh 0 0` atau `--phase baseline` | Ya |
| Random search | `kitchen_run_experiment.sh N 0` atau `kitchen_random_search.sh` | Ya (per trial) |
| Plot hasil | `python scripts/kitchen_plot_results.py` | — |

Early stopping aktif untuk **baseline** dan **setiap trial random search** (`training.use_early_stopping=true`).

### 6.1 Preprocessing Dataset
- Dataset: `Kitchen-Complete-v2` (19 episode, 4.209 timestep total)
- Split: **15 train / 3 validasi / 1 test** (pada level episode)
- Sliding window dengan stride = 1
- Augmentasi: Gaussian noise pada vektor observasi (hanya data train)
  - `obs_noise_std: 0.01`
  - `action_noise_std: 0.0` (aksi tidak diaugmentasi)

### 6.2 Modifikasi Arsitektur (State-Based)
- PointNet++ encoder **dihapus**
- Diganti dengan **state encoder** (linear projection ℝ⁵⁹ → ℝᵈ)
- Input MLP: `[t ∈ ℝ¹, a_t ∈ ℝ⁹, s ∈ ℝᵈ]` → total `10 + d` dimensi
- Output MLP: vektor aksi ℝ⁹

### 6.3 Konfigurasi Inferensi (Hardcoded)
| Parameter       | Nilai        |
|-----------------|--------------|
| `noise_scale`   | 1.0          |
| `sigma_var`     | 0.0          |
| `ode_tol`       | 1e-5         |
| `ode_sampler`   | `rk45`       |

### 6.4 Perangkat Keras
| Komponen     | Spesifikasi             |
|--------------|-------------------------|
| GPU          | NVIDIA GTX 1080 Ti      |
| VRAM         | 8 GB                    |
| CPU          | Intel Core i7           |
| RAM          | 16 GB                   |
| OS           | Ubuntu 20.04 LTS        |

---

### 6.5 Early Stopping

Early stopping pada eksperimen ini dirancang **konservatif** karena dua alasan utama: (1) dataset sangat kecil (hanya 3 episode validasi) sehingga validation loss bisa bersifat noisy, dan (2) flow matching model sering menunjukkan fase plateau panjang sebelum akhirnya konvergen ke solusi yang lebih baik. Strategi ini menggunakan dua sinyal berbeda yang harus **keduanya** terpenuhi sebelum training dihentikan.

#### Parameter Early Stopping

| Parameter                    | Nilai   | Keterangan                                                           |
|------------------------------|---------|----------------------------------------------------------------------|
| `min_epochs`                 | 1000    | Epoch minimum sebelum early stopping dapat aktif                     |
| `val_loss_patience`          | 400     | Epoch tanpa perbaikan val loss sebelum sinyal pertama aktif          |
| `val_loss_min_delta`         | 5e-5    | Selisih minimum untuk dihitung sebagai "perbaikan"                   |
| `val_loss_ema_alpha`         | 0.05    | Smoothing factor EMA pada val loss (~window 20 epoch)                |
| `success_rate_check_interval`| 500     | Interval epoch untuk evaluasi success rate ringan                    |
| `success_rate_patience`      | 2       | Jumlah check berturut-turut tanpa perbaikan untuk sinyal kedua aktif |
| `success_rate_min_delta`     | 2.0     | Selisih minimum (pp) untuk dihitung sebagai "perbaikan"              |
| `success_rate_eval_episodes` | 20      | Jumlah episode evaluasi ringan (1 seed)                              |

#### Dua Sinyal yang Harus Keduanya Terpenuhi

Early stopping hanya dipicu jika **Sinyal 1 DAN Sinyal 2** aktif secara bersamaan. Hal ini memastikan training tidak berhenti hanya karena plateau sementara pada satu metrik saja.

**Sinyal 1 — Validation Loss Stagnan (monitor setiap epoch)**

Validation loss diukur setiap akhir epoch, lalu dihaluskan dengan Exponential Moving Average (EMA) untuk meredam noise yang muncul akibat hanya 3 episode validasi:

```
val_loss_ema[e] = α × val_loss[e] + (1 - α) × val_loss_ema[e-1]
```

Sinyal 1 aktif jika `val_loss_patience` epoch berturut-turut tidak ada perbaikan:

```
best_val_loss_ema - val_loss_ema[e] < val_loss_min_delta  selama 400 epoch berturut-turut
```

**Sinyal 2 — Success Rate Stagnan (monitor setiap 500 epoch, mulai epoch 1000)**

Setiap 500 epoch, jalankan evaluasi ringan (20 episode, 1 seed) untuk memperkirakan success rate. Sinyal 2 aktif jika 2 pemeriksaan berturut-turut (≙ 1000 epoch) tidak ada perbaikan:

```
best_success_rate - success_rate_check[c] < success_rate_min_delta  selama 2 check berturut-turut
```

> **Mengapa tidak hanya mengandalkan val loss?** Pada tugas long-horizon seperti Franka Kitchen, penurunan val loss tidak selalu berkorelasi langsung dengan peningkatan success rate (Zhang et al., 2025). Sebaliknya, mengandalkan success rate saja terlalu mahal dan noisy. Dua sinyal ini saling melengkapi.

#### Alur Keputusan Early Stopping

```
Setiap epoch e:
├── e < 1000 → lanjut (fase warmup, early stopping nonaktif)
└── e ≥ 1000 →
    ├── Hitung val_loss_ema[e]
    ├── Update sinyal 1 (val loss patience counter)
    ├── Jika e % 500 == 0:
    │   ├── Jalankan evaluasi ringan (20 ep × 1 seed)
    │   └── Update sinyal 2 (success rate patience counter)
    └── Jika sinyal 1 AKTIF dan sinyal 2 AKTIF:
        ├── Hentikan training
        └── Load checkpoint dengan val_loss_ema terbaik
```

#### Contoh Skenario

| Situasi                                              | Keputusan                                        |
|------------------------------------------------------|--------------------------------------------------|
| Val loss stagnan 400 epoch, success rate masih naik  | **Lanjut** — Sinyal 2 belum aktif                |
| Success rate stagnan 2 check, val loss masih turun   | **Lanjut** — Sinyal 1 belum aktif                |
| Val loss stagnan 400 epoch, success rate stagnan 2 check | **Stop** — Kedua sinyal aktif               |
| Epoch mencapai `num_epochs` (batas atas)             | **Stop** — Budget habis                          |

#### Checkpoint dan Restore

- Simpan checkpoint setiap kali `val_loss_ema` mencapai nilai terbaru terbaik.
- Saat training berhenti (baik karena early stopping maupun budget habis), **load checkpoint terbaik** — bukan bobot epoch terakhir.
- Nama file checkpoint: `best_val_loss_epoch{e}.ckpt`

> **Catatan implementasi:** Karena val loss pada dataset 3 episode bisa bervariasi signifikan, `val_loss_min_delta = 5e-5` sengaja dibuat kecil agar threshold tidak terlalu ketat. Nilai ini dapat disesuaikan jika pada run awal terlihat bahwa val loss bergerak dalam skala yang berbeda.

### 6.6 Visualisasi Hasil

Setelah baseline dan random search selesai, jalankan:

```bash
cd FlowPolicy
python scripts/kitchen_plot_results.py
```

Plot yang dihasilkan (`data/hparam_search/plots/`):

| File | Isi |
|------|-----|
| `success_rate_vs_latency.png` | Scatter + error bar: performa vs kecepatan inferensi |
| `tradeoff_vs_val_loss.png` | Trade-off vs best validation loss (EMA) |
| `subtask_success_rates.png` | Success rate k=1..4 (baseline vs rata-rata search) |
| `tradeoff_vs_hparams.png` | Trade-off vs `num_segments` dan `hidden_dim` |
| `latency_vs_num_segments.png` | Latensi vs K (ukuran marker ∝ success rate) |

---

## 7. Catatan Teknis dari Kode (Hydra Config)

Berikut nilai aktual dari `flowpolicy.yaml` dan `kitchen_complete.yaml` yang perlu diperhatikan:

```yaml
# Training
training:
  num_epochs: 3000
  lr_scheduler: cosine
  lr_warmup_steps: 500
  ema_decay: 0.95  # ⚠️ TIDAK dipakai di kode, hanya ada di YAML

# Optimizer (AdamW)
optimizer:
  lr: 1.0e-4
  betas: [0.95, 0.999]
  eps: 1.0e-8
  weight_decay: 1.0e-6

# EMA (yang benar-benar dipakai)
ema:
  inv_gamma: 1.0
  power: 0.75
  min_value: 0.0
  max_value: 0.9999
  update_after_step: 0

# Consistency FM
policy:
  num_segments: 2
  boundary: 1
  delta: 1.0e-2
  alpha: 1.0e-5
  eps: 1.0e-2
  num_inference_step: 1

# Shape
shape_meta:
  obs:
    agent_pos:
      shape: [70]   # dimensi observasi per timestep
  action:
    shape: [9]      # dimensi aksi per timestep
```

> **Perhatian:** `training.ema_decay` hanya ada di config dan **tidak terhubung** ke `EMAModel`. EMA yang benar-benar dipakai menggunakan `inv_gamma`, `power`, `min_value`, `max_value`.

---

## Referensi

- Zhang, Q., et al. (2025). *FlowPolicy: Enabling Fast and Robust 3D Flow-Based Policy via Consistency Flow Matching for Robot Manipulation.* AAAI 2025.
- Yang, L., et al. (2024). *Consistency Flow Matching: Defining Straight Flows with Velocity Consistency.* arXiv.
- Bergstra, J., & Bengio, Y. (2012). *Random search for hyper-parameter optimization.* JMLR.
- Chi, C., et al. (2025). *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* IJRR.
- Gupta, A., et al. (2019). *Relay Policy Learning.* arXiv.
