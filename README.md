# Semantic Fashion Image Search

A text-to-image fashion search application. Describe an item in natural language, and the app uses CLIP to encode the query, FAISS to retrieve the closest product vectors, and the local product-image collection to display results.

## Contents

- [Overview](#overview)
- [How Search Works](#how-search-works)
- [Project Layout](#project-layout)
- [Data And Artifacts](#data-and-artifacts)
- [Install And Run](#install-and-run)
- [Rebuild The Search Index](#rebuild-the-search-index)
- [Troubleshooting](#troubleshooting)
- [GitHub And Dataset Notes](#github-and-dataset-notes)

## Overview

The project contains a Streamlit user interface and a notebook-based workflow for generating the search index.

| Component | Responsibility |
| --- | --- |
| Streamlit | Accept a text query and number of results; display matching product images. |
| CLIP (`openai/clip-vit-base-patch32`) | Convert text into a 512-dimensional feature vector in the same feature space as the indexed images. |
| FAISS | Search the saved image vectors using inner product; normalized vectors make this equivalent to cosine similarity ranking. |
| Pandas metadata table | Map each result position returned by FAISS to a product row and image filename. |
| Product image directory | Supply the image files shown in the results. |

### System Architecture

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#d9f2ff","primaryTextColor":"#152b3c","primaryBorderColor":"#168aad","lineColor":"#526777","secondaryColor":"#e8f7df","tertiaryColor":"#fff0cf","background":"#ffffff","fontFamily":"Segoe UI, sans-serif"}}}%%
flowchart LR
	visitor([Visitor]) -->|fashion description| ui[Streamlit search interface]
	ui --> processor[CLIP text processor]
	processor --> encoder[CLIP text encoder]
	encoder --> normalize[Normalize query vector]
	normalize --> search[FAISS nearest-neighbor search]
	search -->|row positions| metadata[(cleaned_data.csv)]
## How Search Works

### System Architecture
	imageResolver --> images[(images/<product-id>.jpg)]
	images -->|ranked product photos| ui

	classDef user fill:#fff0cf,stroke:#e09f3e,color:#422b0b,stroke-width:2px;
	classDef interface fill:#d9f2ff,stroke:#168aad,color:#123044,stroke-width:2px;
	classDef model fill:#eadfff,stroke:#805ad5,color:#2d1c53,stroke-width:2px;
	classDef index fill:#ffe1e1,stroke:#db5a5a,color:#4b1e1e,stroke-width:2px;
	classDef data fill:#e8f7df,stroke:#58a44c,color:#1d3a1b,stroke-width:2px;
	class visitor user;
	class ui interface;
	imageResolver --> images[(Local product JPEGs)]
	class search index;
	class metadata,imageResolver,images data;
```

### Search Request

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#e4f4ff","primaryTextColor":"#17324d","primaryBorderColor":"#2583b8","lineColor":"#526777","secondaryColor":"#eef9e8","tertiaryColor":"#fff1dd","actorBkg":"#d9f2ff","actorBorder":"#168aad","actorTextColor":"#123044","signalColor":"#466477","signalTextColor":"#203849","noteBkgColor":"#fff0cf","noteTextColor":"#422b0b","noteBorderColor":"#e09f3e","activationBkgColor":"#eadfff","activationBorderColor":"#805ad5"}}}%%
sequenceDiagram
	autonumber
	actor User
	participant UI as Streamlit UI
	participant CLIP as CLIP text encoder
	participant FAISS as FAISS index
	participant CSV as Metadata CSV
	participant Disk as Image directory
	User->>UI: Enter query and choose result count
	UI->>CLIP: Tokenize and encode query
	CLIP-->>UI: Projected text feature vector
	Note over UI,CLIP: Normalize the feature vector before search
	UI->>FAISS: Search vector for top-k neighbors
	FAISS-->>UI: Matching row positions and scores
	loop For each matching row
		UI->>CSV: Resolve row position to image filename
		CSV-->>UI: Product metadata and image path
		UI->>Disk: Open local image
		Disk-->>UI: Product image
	end
	UI-->>User: Display ranked results
```

## Project Layout

```text
.
|-- README.md
|-- requirements.txt
|-- archive (13)/
|   |-- app.py
|   |-- cleaned_data.csv
|   |-- clip_index.faiss
|   |-- images/
|   |   |-- 10000.jpg
|   |   `-- <product-id>.jpg
|   |-- styles.csv
|   |-- quries.txt
|   `-- DSA_FINAL_PROJECT.ipynb
`-- .git/
```

The folder name `archive (13)` is part of the current project layout. Commands below quote it because it contains a space and parentheses.

## Data And Artifacts

### Required To Run The App

Keep these files together under `archive (13)`:

| Path | Purpose |
| --- | --- |
| `archive (13)/app.py` | Streamlit application entry point. |
| `archive (13)/cleaned_data.csv` | Product rows. The app uses row order to map FAISS results to images. |
| `archive (13)/clip_index.faiss` | Prebuilt FAISS image-vector index used for retrieval. |
| `archive (13)/images/` | Product JPEGs displayed in search results. Filenames are based on product IDs. |

The app reconstructs each image location from the filename in the CSV and the local `images/` directory. It does not require the original laptop's absolute paths to exist.

### Retained For Rebuilding Or Reference

- `archive (13)/DSA_FINAL_PROJECT.ipynb` documents the CLIP embedding and FAISS index workflow. It reads `cleaned_data.csv` and the local images, then writes the embeddings and index.
- `archive (13)/styles.csv` is the original product metadata source. It is useful if the cleaned metadata needs to be regenerated.
- `archive (13)/quries.txt` contains example search prompts; the app does not read it.
- `requirements.txt` lists the packages needed by the Streamlit app.

### Generated And Optional Files

The notebook creates `clip_image_embeddings.npy` as an intermediate file before writing `clip_index.faiss`. The running app does not load that embeddings file. It can be omitted from a runtime-only distribution, but it will be regenerated when the notebook rebuilds the index.

The app downloads the CLIP model and processor from Hugging Face on first run and uses the local Hugging Face cache afterward. Local copies of model weights, tokenizer files, notebook checkpoints, and the old ResNet artifacts are not required by the current app.

### Index And Metadata Must Stay Aligned

FAISS returns positions, and the app uses each position with `df.iloc[position]`. The CSV row order must therefore match the order used when the FAISS index was built. If you reorder, filter, or replace the metadata after index creation, rebuild the index before searching; otherwise results can point at the wrong product images.

### Index Generation Flow

```mermaid
%%{init: {"theme":"base","themeVariables":{"primaryColor":"#e4f4ff","primaryTextColor":"#17324d","primaryBorderColor":"#2583b8","lineColor":"#526777","secondaryColor":"#eef9e8","tertiaryColor":"#fff1dd","fontFamily":"Segoe UI, sans-serif"}}}%%
flowchart TD
	csv[(cleaned_data.csv)] --> rows[Load and validate metadata rows]
	folder[(images/<product-id>.jpg)] --> rows
	rows --> model[Load CLIP image encoder]
	model --> batch[Encode images in batches]
	batch --> vectors[Normalize image feature vectors]
	vectors --> intermediate[(clip_image_embeddings.npy)]
	vectors --> faiss[Build FAISS IndexFlatIP]
	faiss --> artifact[(clip_index.faiss)]
	artifact --> app[Streamlit app loads index]
	rows --> app

	classDef source fill:#e8f7df,stroke:#58a44c,color:#1d3a1b,stroke-width:2px;
	classDef process fill:#d9f2ff,stroke:#168aad,color:#123044,stroke-width:2px;
	classDef model fill:#eadfff,stroke:#805ad5,color:#2d1c53,stroke-width:2px;
	classDef generated fill:#fff0cf,stroke:#e09f3e,color:#422b0b,stroke-width:2px;
	classDef retrieval fill:#ffe1e1,stroke:#db5a5a,color:#4b1e1e,stroke-width:2px;
	class csv,folder source;
	class rows,batch process;
	class model,vectors model;
	class intermediate,artifact generated;
	class faiss,app retrieval;
```

## Install And Run

### Prerequisites

- Windows, macOS, or Linux.
- Python and `pip`. This workspace was run with Python 3.12.
- Internet access on the first run to download `openai/clip-vit-base-patch32` from Hugging Face.
- The required CSV, FAISS index, and product images listed above.
- A GPU is optional. The app uses CUDA when PyTorch detects it; otherwise it runs on CPU, which may take longer.

### Windows PowerShell

Run from the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run "archive (13)/app.py"
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

### macOS Or Linux

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run "archive (13)/app.py"
```

### Using The Search Page

1. Enter a description such as `red floral dress`, `white sneakers`, or `navy blue suit`.
2. Set the number of results from 5 to 50.
3. Select **Search** to see the nearest products.

Results are ranked by similarity in CLIP's shared image/text embedding space. The app displays product images when their files are present in `images/`.

## Rebuild The Search Index

Rebuilding is optional when the included CSV and index are already available. It can take substantial time and compute, especially on CPU.

1. Ensure `cleaned_data.csv` and all referenced images are present.
2. Open `archive (13)/DSA_FINAL_PROJECT.ipynb` in Jupyter or VS Code.
3. Run the dependency-install cell, then execute the notebook cells in order.
4. The notebook creates `clip_image_embeddings.npy` and writes a new `clip_index.faiss` next to the app.
5. Restart or rerun Streamlit so it loads the rebuilt index.

The notebook uses batches of 64 by default. Reduce `BATCH_SIZE` in its configuration cell if GPU memory is limited. The notebook's interactive search example also uses `ipywidgets`; install `ipywidgets` in the notebook environment if that example reports a missing package.

The notebook regenerates the CLIP image index; it does not regenerate `cleaned_data.csv` from `styles.csv`. If the cleaned CSV must be rebuilt from raw metadata, repeat the project's data-cleaning step and ensure it writes image paths/IDs compatible with the local `images/` directory before building embeddings.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| `FileNotFoundError` for `cleaned_data.csv` or `clip_index.faiss` | The file is missing or not inside `archive (13)`. | Restore the required file to the documented folder. |
| Results appear but product photos are missing | The image file is absent or its filename does not match the product ID in the CSV. | Check that `images/<product-id>.jpg` exists. |
| `ModuleNotFoundError` | Dependencies were installed into a different Python environment. | Activate `.venv`, then rerun `python -m pip install -r requirements.txt`. |
| Hugging Face download or connection error | The model is not cached and the machine cannot reach Hugging Face. | Restore network access or configure Transformers to use a valid local model cache. |
| App starts slowly | CLIP is loading for the first time, or inference is running on CPU. | Allow the model download/load to finish; use a supported CUDA-enabled PyTorch setup if available. |
| FAISS returns unrelated or mismatched product rows | The metadata was reordered or changed after the index was built. | Rebuild the index from the current CSV in its current row order. |
| Notebook reports missing `ipywidgets` | The app requirements do not include notebook-only widget packages. | Install `ipywidgets` in the active notebook environment. |

## GitHub And Dataset Notes

The full `images/` directory and generated model/index data are large. Do not commit large binaries to ordinary Git history without first choosing a distribution strategy. Options include Git LFS, a release or artifact store, or a separate data download with setup instructions. A clone that does not receive `cleaned_data.csv`, `clip_index.faiss`, and the matching `images/` directory cannot run the full local search.

GitHub rejects individual Git files larger than 100 MB. Removing a file from the working tree does not remove copies already present in earlier commits; this repository's existing history has previously contained large model and dataset files. Before pushing a cleaned version, inspect the Git history and decide whether to use Git LFS or publish a fresh source-only history/repository. Do not rewrite shared history without coordinating with other collaborators.

Before redistributing the product images or metadata, verify that their source license permits redistribution. Keep the original source and attribution details with the project where required.
