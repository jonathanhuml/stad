# STAD implementation outline

## Goal

Implement the method from **“Generative AI Enables EEG Super-Resolution via Spatio-Temporal Adaptive Diffusion Learning”** as a PyTorch training pipeline that reconstructs **high-resolution EEG (up to 256 channels)** from **low-resolution EEG** by combining:

1. a **pretrained DreamDiffusion-style EEG MAE** for HR EEG latent representations,
2. a **Spatio-Temporal Condition (STC)** module that encodes LR EEG plus electrode positions,
3. a **Multi-scale Transformer Denoiser (MTD)** trained as a **latent diffusion** noise predictor.

Given the paper’s wording and citations, the safest implementation assumption is that **STAD reuses the DreamDiffusion EEG MAE backbone (or a very close variant)** and places most of its novelty in the **conditioning pathway and latent denoiser**, not in a brand-new autoencoder.

DreamDiffusion is implemented here: https://github.com/bbaaii/DreamDiffusion

And the DreamDiffusion paper is here: https://arxiv.org/pdf/2306.16934

The STAD paper is here: https://arxiv.org/pdf/2407.03089

Make sure to check everything in this document against these papers, all of this could be incorrect, slightly or majorly.

Be careful and thorough in spotting any differences.

This outline is deliberately practical rather than purely descriptive. It includes unresolved decisions, simplifications we may choose initially, and places where the paper is underspecified.

---

## What the paper actually specifies

### Data and task
- Dataset: **Localize-MI**
- HR target EEG: **256 channels**
- LR inputs are created by channel downsampling to different montages:
  - scale 2 -> 128 channels
  - scale 4 -> 64 channels
  - scale 8 -> 32 channels
  - scale 16 -> 16 channels
- Preprocessing in the paper:
  - high-pass filtering
  - notch filtering at **50, 100, 150, 200 Hz**
  - segment into **350 ms epochs**
  - original HR recorded at **8000 Hz**
- Final paired dataset size in the paper: **23,930 samples**
- Split: **80% train / 10% val / 10% test**

### Model structure
The full STAD system has three pieces:

#### 1. Pretrained EEG autoencoder
- They use a **DreamDiffusion-style EEG Masked Autoencoder (MAE)** as the HR EEG encoder-decoder backbone.
- HR EEG is encoded to latent vectors `z0`.
- The decoder maps denoised latent vectors back to reconstructed SR EEG.
- The STAD paper points to **DreamDiffusion** for encoder architecture details rather than fully specifying a new MAE, so we should treat DreamDiffusion as the default backbone unless later evidence shows otherwise.

#### 2. STC (Spatio-Temporal Condition module)
Inputs:
- LR EEG time series
- LR electrode spatial positions

Output:
- condition vector / condition tokens `c`

Described pieces:
- spatial position embedding
- 1D convolution block
- Transformer block

#### 3. MTD (Multi-scale Transformer Denoising module)
Inputs:
- noisy latent `z_t`
- timestep `t`
- condition `c`

Output:
- predicted diffusion noise `eps_theta(z_t, t, c)`

Described pieces:
- position encoding
- multi-scale 1D convolutions with kernel sizes **3, 5, 7, 9**
- diffusion Transformer blocks
- each block uses:
  - layer norm
  - self-attention
  - cross-attention to condition `c`
  - feed-forward network

### Training objective
They train the diffusion model with the standard simplified latent-diffusion noise-prediction loss:

`L_DM = E || eps - eps_theta(z_t, t, tau_theta(x)) ||^2`

where:
- `x` = LR EEG
- `tau_theta(x)` = STC output
- `z0` comes from encoding HR EEG through the pretrained MAE encoder

### Training hyperparameters reported
- framework: PyTorch
- optimizer: Adam
- learning rate: `2e-4`
- epochs: `300`
- batch size: `32`
- diffusion steps: `1000`
- noise schedule: paper says **sine or cosine**
- attention heads: `16`
- hidden dimension: `64`

---

## Important implementation reality check

The paper is **not** fully reproducible from text alone. The biggest missing pieces are now narrower than before, because the MAE backbone is likely inherited from DreamDiffusion. The remaining gaps are:

1. which parts of the **DreamDiffusion EEG MAE** are kept exactly versus lightly modified,
2. the exact tensor shapes through STC and MTD,
3. the exact latent arrangement used for diffusion,
4. whether STAD keeps DreamDiffusion’s exact patching / masking hyperparameters,
5. the exact noise schedule implementation,
6. the exact train/val/test split protocol at the subject/session level,
7. whether the MAE is frozen or jointly fine-tuned during diffusion training.

So the right engineering plan is:

- first build a **faithful paper-inspired baseline**,
- then tighten it toward the paper where possible,
- and explicitly log all assumptions.

---

## Step-by-step recipe

## Step 0: Decide scope of the first implementation

Given what we now know, we should assume the STAD autoencoder is **DreamDiffusion’s EEG MAE or a very close derivative**.

### Option A: “DreamDiffusion-backed STAD”
Use:
- a **DreamDiffusion-style EEG MAE** as the HR latent autoencoder,
- latent diffusion over encoder latents,
- STC + MTD modules matching the STAD paper structurally.

Pros:
- best match to the paper
- removes one major ambiguity
- lets us focus implementation effort on the truly new STAD pieces

Cons:
- DreamDiffusion still leaves some hyperparameter choices open for our data

### Option B: “simplified replacement MAE”
Use:
- our own simpler EEG MAE,
- same STAD diffusion pipeline around it.

Pros:
- easier to customize
- useful as an ablation

Cons:
- less faithful to STAD

**Recommendation:** make **DreamDiffusion-backed STAD** the main path, and keep the simpler MAE only as an optional ablation.

---

## Step 1: Define file structure

Suggested project structure:

```text
project/
  configs/
    stad_localize_mi_scale4.yaml
    stad_localize_mi_scale8.yaml
    stad_localize_mi_scale16.yaml
  data/
    localize_mi.py
    preprocess_localize_mi.py
    montages.py
  models/
    eeg_mae.py
    stc.py
    mtd.py
    diffusion.py
    stad.py
  training/
    train_mae.py
    train_stad.py
    losses.py
    metrics.py
    scheduler.py
  eval/
    eval_reconstruction.py
    eval_classification.py
    eval_localization.py
  utils/
    positions.py
    checkpointing.py
    logging.py
    seed.py
  outline.md
```

---

## Step 2: Build the preprocessing pipeline

### 2.1 Load Localize-MI
We need a dataset loader that returns:
- HR EEG: shape likely `[C_hr, T] = [256, T]`
- stimulus / event annotations
- electrode positions for all 256 channels
- optional session/subject metadata

### 2.2 Reproduce paper preprocessing
Per epoch:
1. load raw continuous HR EEG
2. high-pass filter
3. notch filter at 50, 100, 150, 200 Hz
4. cut fixed **350 ms** epochs using annotations
5. store resulting HR epochs

### 2.3 Decide sampling rate handling
This is one of the biggest design choices.

The paper says:
- acquisition at **8000 Hz**
- epochs of **350 ms**

That would imply `T = 2800` samples per epoch if no temporal downsampling is applied.

But that is expensive for Transformers and diffusion. The paper does not clearly state whether they further downsample in time after segmentation.

So we need to choose:

#### Design choice A: keep 8000 Hz
- input shape per epoch: `[256, 2800]`
- closest to text
- much heavier

#### Design choice B: resample after filtering
- e.g. to 1000 Hz or 500 Hz
- makes patching and training much easier
- probably more realistic for a first implementation

**Recommendation for v1:** resample after preprocessing to a manageable rate, but record this as a deviation.

### 2.4 Create LR-HR pairs
For each HR epoch:
- select subsets of channels corresponding to:
  - 128-channel montage
  - 64-channel montage
  - 32-channel montage
  - 16-channel montage
- LR input = subset of channels from HR
- HR target = full 256-channel epoch

We should save:
- `x_lr`: `[C_lr, T]`
- `y_hr`: `[256, T]`
- `pos_lr`: `[C_lr, 3]` or `[C_lr, 2]`
- `pos_hr`: `[256, 3]` or `[256, 2]`
- metadata: subject, session, epoch id, scale factor

### 2.5 Normalize
Need another explicit design decision:
- per-channel z-score per epoch?
- per-channel dataset-wide normalization?
- global scaling?

**Recommendation for v1:** per-channel z-score using training statistics only, and preserve a reversible normalization path for evaluation.

---

## Step 3: Use DreamDiffusion MAE as the default EEG autoencoder

The paper says it follows DreamDiffusion for the EEG MAE. That changes the default implementation assumption: we should **start from DreamDiffusion’s EEG MAE design**, then layer STAD on top.

### 3.1 What DreamDiffusion contributes
DreamDiffusion’s EEG MAE is best understood as a **temporal masked-signal MAE**:
- EEG is treated as a signal with channels and time,
- the model performs **time-domain tokenization** rather than a generic image-style 2D patching scheme,
- patch embedding is implemented with **1D convolution**,
- the architecture is an **asymmetric MAE** with Transformer encoder/decoder.

So for STAD, the default backbone assumption should be:
- **time-domain tokenization first**
- Transformer latent sequence
- decoder able to reconstruct HR EEG from latent tokens

### 3.2 What STAD appears to keep versus change
Likely kept from DreamDiffusion:
- MAE encoder/decoder backbone
- temporal tokenization
- Transformer latent sequence representation
- masked reconstruction pretraining logic

Likely changed in STAD:
- the MAE is used as a **latent EEG autoencoder** for super-resolution rather than EEG-to-image generation,
- DreamDiffusion’s image-generation / CLIP-alignment pieces are irrelevant and should be removed,
- the new method-specific modules are **STC** and **MTD**, not a radically new MAE.

### 3.3 Representation we should assume for implementation
Use an EEG tensor:
- `[B, C, T]`

Then follow DreamDiffusion-style temporal patching:
- split along time into fixed-length windows / tokens
- project with a 1D conv patch embed
- produce latent token sequence:
```text
z0: [B, L, D]
```

This latent token sequence is what the diffusion model will denoise.

### 3.4 Do we need 2D channel-time patching?
Probably not for the first faithful implementation.

Even though STAD informally describes EEG as a 2D array with rows = channels and columns = time, its wording about splitting each channel into fixed-length windows remains compatible with DreamDiffusion’s **time-domain tokenization**. So we should **not invent a 2D patching scheme unless we find stronger evidence**.

### 3.5 Freeze or finetune?
Paper wording still suggests:
- use a pretrained MAE,
- encode HR EEG to latent vectors,
- train diffusion in latent space.

**Recommendation for v1:**
- initialize from DreamDiffusion-style MAE pretraining,
- freeze encoder and decoder during STAD diffusion training,
- optionally test later:
  - freeze encoder, finetune decoder
  - finetune all modules jointly

### 3.6 Practical implementation consequence
The engineering target is no longer:
- “design a generic EEG MAE”

It is now:
- “implement or port the DreamDiffusion EEG MAE cleanly, strip away image-generation-specific extras, and expose:
  - `encode(y_hr) -> z0`
  - `decode(z_hat0) -> y_hat_sr`”

## Step 4: Implement or port the DreamDiffusion MAE phase

### 4.1 First decision: reuse weights or just reuse architecture?
There are two realistic paths:

#### Path A: reuse DreamDiffusion architecture only
- implement the DreamDiffusion EEG MAE structure,
- pretrain it on our HR EEG data,
- then use it inside STAD.

#### Path B: initialize from existing DreamDiffusion weights if compatible
- only possible if the training data format, channel layout, and sampling assumptions are compatible enough.

**Recommendation:** assume **architecture reuse** first. Weight reuse is a bonus, not a dependency.

### 4.2 Strip out non-STAD pieces
DreamDiffusion as a full system includes EEG-to-image machinery. For STAD we only need the EEG MAE backbone, so remove or ignore:
- image-generation components,
- CLIP/image alignment paths,
- any auxiliary image-feature losses not needed for EEG reconstruction.

### 4.3 Pretraining dataset
Use only HR EEG epochs from Localize-MI.

### 4.4 MAE objective
Use DreamDiffusion-style masked EEG reconstruction:
- temporal token masking
- reconstruction loss on masked portions

### 4.5 Initial hyperparameters
If we can verify DreamDiffusion defaults, use them as the initial reference point; otherwise document every deviation explicitly.

Key hyperparameters to pin down:
- temporal patch size
- embedding dimension
- encoder depth
- decoder depth
- number of heads
- mask ratio

### 4.6 Deliverable
At the end of this stage we want:
- `dreamdiffusion_mae_encoder.pt`
- `dreamdiffusion_mae_decoder.pt`
- latent shape documentation
- reconstruction sanity plots on HR EEG

### 4.7 Sanity checks
- can the MAE reconstruct HR EEG reasonably?
- do latent embeddings vary across examples?
- does the latent token sequence have the shape expected by STAD’s diffusion model?
- does stripping the image-generation extras leave a clean standalone EEG autoencoder?

## Step 5: Define the latent diffusion space

This is a critical choice.

The paper says:
- encode HR EEG into latent vectors `z0`
- add Gaussian noise in latent space
- denoise with MTD
- decode denoised latent to SR EEG

But it does not precisely define whether `z0` is:
- a flat latent vector,
- a sequence of latent tokens,
- or a channel-time latent grid.

### Recommended latent design
Use MAE encoder output as a **sequence of latent tokens**:

```text
z0: [B, L, D]
```

where:
- `L` = number of HR EEG patches/tokens
- `D` = latent dim

This matches Transformer denoising naturally.

Alternative:
- reshape to `[B, D, L]` for multi-scale 1D convolutions, then transpose back.

---

## Step 6: Implement STC

The paper’s STC takes:
- LR EEG time series
- LR electrode positions

and outputs condition `c`.

### 6.1 Input format
Use:
- `x_lr`: `[B, C_lr, T]`
- `pos_lr`: `[B, C_lr, P]`, where `P` is 2 or 3

### 6.2 Spatial position embedding
Implement a small MLP:
- `Linear(P, d_model) -> GELU -> Linear(d_model, d_model)`

Output:
- `pos_emb`: `[B, C_lr, d_model]`

### 6.3 Temporal feature extractor
Paper says:
- partition input time series into patches
- process with 1D convolution block

Implement:
- per-channel 1D conv over time
- BN + ReLU
- maybe stride equal to patch size or use unfold + linear projection

Two valid v1 designs:

#### Design 1: Conv patch embed
```python
Conv1d(in_channels=C_lr, out_channels=d_model, kernel_size=p_t, stride=p_t)
```
This mixes channels early.

#### Design 2: Channel-wise temporal encoder
Apply conv independently per channel, then tokenise.

**Recommendation:** Design 2 is more faithful to the text.

### 6.4 Transformer block
Combine:
- temporal token features
- channel identity or channel position embeddings
- optional sinusoidal time-position encoding over patch index

Then run a Transformer encoder.

### 6.5 Output format for condition
We should output condition tokens:
```text
c: [B, Lc, Dc]
```

Best case:
- set `Dc = D` so cross-attention is simple
- allow `Lc` to differ from diffusion token length

### 6.6 Open question
How should LR channels and HR latent tokens align?

Possible answers:
1. no explicit alignment; let cross-attention learn it,
2. interpolate condition tokens to HR token count,
3. build channel-aware tokenization so HR and LR share geometry.

**Recommendation for v1:** no hard alignment; just use cross-attention from diffusion tokens to STC tokens.

---

## Step 7: Implement MTD

The MTD predicts diffusion noise from `(z_t, t, c)`.

### 7.1 Input / output
- input latent: `z_t` with shape `[B, L, D]`
- timestep embedding: `[B, D]`
- condition tokens: `c` with shape `[B, Lc, D]`
- output predicted noise: same shape as `z_t`

### 7.2 Time embedding
Use standard sinusoidal diffusion timestep embedding followed by MLP:
```python
t_emb = MLP(sinusoidal_embedding(t))
```

### 7.3 Positional encoding
Add learned or sinusoidal token positional encodings to `z_t`.

### 7.4 Multi-scale conv block
Paper says:
- use 1D convs with kernel sizes `3, 5, 7, 9`
- concatenate outputs along feature dimension

Implementation plan:
1. transpose latent tokens to `[B, D, L]`
2. run four `Conv1d(D, D_branch, kernel_size=k, padding=k//2)` branches
3. apply BN and activation
4. concatenate to `[B, D_concat, L]`
5. project back to `[B, D, L]`
6. transpose back to `[B, L, D]`

### 7.5 Diffusion Transformer block
Each block should contain:
1. LayerNorm
2. Multi-head self-attention over `z_t`
3. residual
4. LayerNorm
5. cross-attention: query from latent tokens, key/value from condition tokens
6. residual
7. LayerNorm
8. MLP / FFN
9. residual

### 7.6 Where to inject time embedding
Options:
- add `t_emb` to token embeddings at input,
- add it before every block,
- use AdaLN-style modulation.

The paper only states that timestep embedding is concatenated / combined as conditioning.

**Recommendation for v1:** add projected `t_emb` to every latent token before the first block, then optionally again inside blocks.

### 7.7 Number of blocks
Not specified.

**Recommendation:** start with 4 to 8 blocks.

### 7.8 Hidden dimension issue
The paper says hidden dimension is **64**. That is quite small.

Possible interpretations:
- token dim `D = 64`
- inner hidden dim of some block = 64
- condition dim = 64

**Recommendation:** begin with `D = 64` for a paper-faithful baseline, but be prepared to increase to 128 or 256 if capacity is too low.

---

## Step 8: Implement the diffusion process

### 8.1 Forward diffusion
Given `z0`:
- sample timestep `t`
- sample Gaussian noise `eps`
- form
```text
z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps
```

### 8.2 Noise schedule
Paper says:
- `T = 1000`
- sine or cosine schedule

This is ambiguous.

**Recommendation for v1:** use a standard cosine beta schedule.

### 8.3 Reverse model target
Predict `eps` from `(z_t, t, c)`.

### 8.4 Training loss
Use:
```text
loss = mse(pred_eps, eps)
```

### 8.5 Sampling
At inference:
1. encode LR EEG to condition `c`
2. initialize `z_T ~ N(0, I)`
3. iteratively denoise to `z_0`
4. decode `z_0` with MAE decoder to get SR EEG

### 8.6 DDPM or DDIM?
Paper text gives standard DDPM-style reverse equations.

**Recommendation for v1:** train with DDPM objective, sample first with DDPM, then add DDIM for speed later.

---

## Step 9: Train in phases

### Phase 1: MAE pretraining
Train MAE on HR EEG only.

### Phase 2: STAD diffusion training
For each batch:
1. load `(x_lr, y_hr, pos_lr)`
2. encode `y_hr -> z0` using frozen MAE encoder
3. compute `c = STC(x_lr, pos_lr)`
4. sample `t ~ Uniform({1,...,T})`
5. sample `eps ~ N(0, I)`
6. build noisy latent `z_t`
7. predict `eps_hat = MTD(z_t, t, c)`
8. optimize MSE between `eps_hat` and `eps`

### Phase 3: Sampling and evaluation
1. sample latent from noise conditioned on LR EEG
2. decode to SR EEG
3. compare against HR EEG

---

## Step 10: Reproduce the paper’s evaluation

### 10.1 Reconstruction metrics
Implement:
- PCC
- NMSE
- SNR
- MAE

Need to define:
- per-channel then average?
- per-sample global?
- averaged over all test examples?

The paper does not fully specify aggregation.

**Recommendation:** report both:
- macro average over channels and samples
- global aggregate

### 10.2 Scaling-factor experiments
Run at:
- 128 -> 256
- 64 -> 256
- 32 -> 256
- 16 -> 256

### 10.3 Downstream classification
The paper trains **EEGNet** on LR EEG versus reconstructed SR EEG for abnormal-vs-normal classification.

We should:
- reproduce the same binary labels if available from stimulation events
- train a separate EEGNet classifier on:
  - LR input
  - SR reconstructed input
- compare ACC / precision / recall / F1

### 10.4 Frequency-domain classification
Paper also computes PSD features over:
- delta, theta, alpha, beta, gamma, all bands

Then uses classification over `c x 5` features.

This is optional for first pass but useful for paper alignment.

### 10.5 Source localization
This is the most complex downstream task and can wait until the reconstruction pipeline is stable.

---

## Step 11: Track all unresolved design decisions explicitly

We should make a section in config files or logs that records:

1. temporal resampling rate after raw preprocessing
2. patch length for MAE
3. MAE mask ratio
4. latent token count and latent dim
5. whether MAE is frozen
6. STC tokenization scheme
7. 2D or 3D electrode coordinates
8. beta schedule exact formula
9. DDPM or DDIM sampling
10. data split strategy:
   - random epoch split
   - session split
   - subject split

This matters because random epoch splitting can leak subject/session information and inflate performance.

---

## Step 12: Minimal viable implementation path

If we want the fastest route to a working file:

### MVP order
1. preprocess Localize-MI into paired LR/HR epoch tensors
2. implement a simple EEG MAE on HR data
3. pretrain MAE
4. implement STC
5. implement MTD
6. implement latent DDPM training
7. generate SR EEG
8. evaluate PCC / NMSE / SNR / MAE

### Simplifications allowed in MVP
- freeze MAE
- use cosine schedule only
- use one scaling factor first, probably **64 -> 256** or **32 -> 256**
- use only reconstruction metrics before downstream tasks
- skip source localization initially

---

## Step 13: Practical PyTorch tensor plan

A concrete tensor plan for v1:

### MAE
- input HR EEG: `[B, 256, T]`
- patchify along time only, following the DreamDiffusion EEG MAE assumption
- encoder outputs:
  - `z0: [B, L_hr, D]`

### STC
- input LR EEG: `[B, C_lr, T]`
- input positions: `[B, C_lr, 3]`
- output:
  - `c: [B, L_cond, D]`

### MTD
- input:
  - `z_t: [B, L_hr, D]`
  - `t: [B]`
  - `c: [B, L_cond, D]`
- output:
  - `eps_hat: [B, L_hr, D]`

### Decoder
- input:
  - `z_hat0: [B, L_hr, D]`
- output:
  - `y_hat_sr: [B, 256, T]`

This is internally coherent and maps well to the paper.

---

## Step 14: Suggested first hyperparameter set

These are proposed defaults for a first run, not paper-guaranteed values.

```yaml
data:
  sample_rate: 1000           # design choice for v1, not paper-guaranteed
  epoch_ms: 350
  scales: [4]                 # start with 64 -> 256
  normalize: per_channel_zscore

mae:
  patch_len: 25
  embed_dim: 128
  encoder_depth: 6
  decoder_depth: 4
  num_heads: 8
  mask_ratio: 0.75

diffusion:
  timesteps: 1000
  beta_schedule: cosine
  objective: pred_noise

stc:
  d_model: 128
  conv_kernel: 7
  transformer_depth: 4
  num_heads: 8
  pos_dim: 3

mtd:
  d_model: 128
  num_blocks: 6
  num_heads: 8
  conv_kernels: [3, 5, 7, 9]
  mlp_ratio: 4.0

train:
  batch_size: 32
  lr: 2e-4
  epochs: 300
  optimizer: adam
  grad_clip: 1.0
```

If we want closer paper alignment, reduce `d_model` and set attention heads to 16.

---

## Step 15: Concrete implementation order by file

### `data/preprocess_localize_mi.py`
- raw data loading
- filtering
- epoch extraction
- channel subset generation
- save paired LR/HR dataset

### `data/montages.py`
- define 256/128/64/32/16 channel subsets
- store channel names and coordinates

### `models/eeg_mae.py`
- DreamDiffusion-style 1D patch embed
- temporal masking
- encoder
- decoder
- stripped-down standalone EEG MAE interface
- `encode()` and `decode()` methods

### `models/stc.py`
- position embedding MLP
- temporal conv block
- Transformer encoder
- output condition tokens

### `models/mtd.py`
- timestep embedding
- latent positional encoding
- multi-scale conv module
- diffusion Transformer blocks with self-attn + cross-attn

### `models/diffusion.py`
- beta schedule
- q-sample
- loss target creation
- ancestral/DDIM sampler

### `models/stad.py`
- wraps:
  - frozen MAE encoder/decoder
  - STC
  - MTD
- training forward and sampling forward

### `training/train_mae.py`
- HR-only MAE pretraining

### `training/train_stad.py`
- latent diffusion training loop

### `eval/eval_reconstruction.py`
- PCC, NMSE, SNR, MAE
- plots and qualitative examples

### `eval/eval_classification.py`
- LR vs SR EEGNet experiment

---

## Step 16: High-risk areas

These are the places most likely to break or silently mismatch the paper:

### 16.1 Data split leakage
Epoch-level random splitting may leak subject/session structure.

### 16.2 Latent decoder mismatch
If the decoder cannot reconstruct cleanly from latent tokens, diffusion quality will be capped.

### 16.3 Condition/token geometry mismatch
LR condition tokens and HR latent tokens may have very different structure.

### 16.4 Time resolution too high
Transformers over long EEG sequences become expensive quickly.

### 16.5 Poor electrode embedding
Simple coordinate MLPs may not capture meaningful scalp geometry.

---

## Step 17: Smart ablations we should run

Once the baseline works, ablate:

1. **No position embedding**
2. **No cross-attention**, condition concatenated instead
3. **Single-scale conv** vs multi-scale conv
4. **No MAE latent space**, diffuse directly in signal space
5. **Freeze vs finetune MAE**
6. **Different LR scales**
7. **Cosine vs linear beta schedule**
8. **Per-subject vs pooled training**
9. **2D scalp coords vs 3D coords**
10. **time-only patching vs 2D patching**

---

## Step 18: What I would personally implement first

A very concrete first pass:

1. preprocess Localize-MI
2. start only with **64 -> 256**
3. resample to a manageable temporal rate
4. port or reimplement the DreamDiffusion-style time-patching EEG MAE
5. freeze MAE after pretraining
6. represent `z0` as latent token sequence
7. build STC with:
   - channel-wise temporal conv
   - 3D coordinate embedding
   - Transformer encoder
8. build MTD with:
   - timestep embedding
   - multi-scale conv branch
   - 6 diffusion Transformer blocks
9. train latent DDPM with noise-prediction MSE
10. evaluate reconstruction metrics
11. only then add EEGNet downstream classification

That gets us to a working research baseline quickly.

---

## Step 19: Questions we should answer before coding

1. Do we want a **strict reproduction** or a **paper-inspired baseline**?
2. Do we have direct access to **Localize-MI** in a usable format?
3. Are we okay **resampling in time** for tractability?
4. Should the first version support **all scaling factors** or just one?
5. Do we want to **port DreamDiffusion MAE closely**, or reimplement a cleaner equivalent that preserves its temporal-token MAE behavior?
6. Should evaluation be **subject-independent**?

---

## Step 20: Definition of done

A good first milestone is:

- trainable STAD pipeline in PyTorch,
- saved checkpoints for DreamDiffusion-style MAE and diffusion model,
- generated SR EEG examples,
- reconstruction metric report on test set,
- explicit note of deviations from the paper.

A stronger milestone is:

- plus EEGNet downstream improvement on SR vs LR.

---

## Brief pseudo-code

```python
# Phase 1: MAE pretraining
for y_hr in hr_loader:
    loss = mae.masked_reconstruction_loss(y_hr)
    loss.backward()
    opt.step()

# Phase 2: STAD training
freeze(mae_encoder)
freeze(mae_decoder)

for x_lr, y_hr, pos_lr in paired_loader:
    with torch.no_grad():
        z0 = mae_encoder(y_hr)              # [B, L, D]

    c = stc(x_lr, pos_lr)                  # [B, Lc, D]

    t = sample_timesteps(B, T=1000)
    eps = torch.randn_like(z0)
    z_t = q_sample(z0, t, eps)

    eps_hat = mtd(z_t, t, c)
    loss = mse(eps_hat, eps)

    loss.backward()
    opt.step()

# Inference
c = stc(x_lr, pos_lr)
z = torch.randn(B, L, D)
for t in reversed(range(T)):
    eps_hat = mtd(z, t, c)
    z = p_sample(z, t, eps_hat)
y_hat = mae_decoder(z)
```

---

## Final note

The paper gives enough to build a serious implementation, but not enough for guaranteed exact reproduction. The right target is a transparent, well-documented implementation with explicit assumptions, then iterative tightening toward the paper.
