# NusaInsight — Data & AI Studio

NusaInsight is a Streamlit learning platform for dataset exploration, machine learning, lightweight deep learning, community discussion, and AI assistance.

## Run locally

```bash
python -m venv .venv
```

Activate the environment (Windows: `.venv\Scripts\activate`; macOS/Linux: `source .venv/bin/activate`), then run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens on **Home**. Select another page from the sidebar.

## Configuration

Copy `.streamlit/secrets.example.toml` to `.streamlit/secrets.toml` and enter your own values. Do not commit or share the real Secrets file.

```toml
mongo_uri = "mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority"
mongo_database = "datasphere"
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

See [MONGODB_SETUP.md](MONGODB_SETUP.md) and [GROQ_SETUP.md](GROQ_SETUP.md) for detailed setup instructions.

## Main features

- **Home** — dataset and algorithm summary, MongoDB/community counts, dataset popularity, data exploration, service status, and recent activity.
- **Dataset Explorer** — built-in scikit-learn and UCI datasets, overview, sample data, feature distributions, correlation, data quality, and custom CSV upload.
- **Machine Learning Lab** — Regression, Classification, Clustering, Deep Learning (PyTorch MLP), and MongoDB & Retraining.
- **Community** — MongoDB-backed posts, replies, reactions, and newest-first discussions.
- **AI Assistant** — Groq-powered data-science Q&A.
- **About & Methodology** — project overview, workflow, architecture, and acknowledgements.

## Custom CSV datasets

Open **Dataset Explorer** and scroll to **Upload & Test a New Dataset**. Upload a CSV file (up to 20 MB, 50,000 rows, and 100 columns), provide a name, and choose Classification, Regression, or Clustering. Classification and Regression require a target column. Clustering is unsupervised and requires at least two numeric features. Review the preview and validation results, then save to MongoDB.

Saved custom datasets are stored in `custom_datasets` and a dedicated `custom_data_<name>_<id>` collection. They become available to Dataset Explorer, Machine Learning Lab, Deep Learning, Home, and MongoDB & Retraining.

## Database collections

| Collection | Purpose |
|---|---|
| `community_posts` | Community comments, replies, and reactions |
| `training_logs` | ML/deep-learning training history and evaluation |
| `dataset_uploads` | Uploaded dataset metadata |
| `custom_datasets` | Custom dataset catalog |
| `custom_data_<name>_<id>` | Rows for each uploaded dataset |
| `model_artifacts.files` / `model_artifacts.chunks` | Model files stored with GridFS |

The MongoDB database name remains `datasphere`. The app creates required collections on the first write.

## Project structure

```text
NusaInsight/
├── app.py                         # Main app, routing, datasets, Community and ML tabs
├── app_modules/                    # Application modules
│   ├── __init__.py
│   ├── home_dashboard.py            # Home dashboard, realtime weather and service status
│   ├── custom_dataset_manager.py    # CSV validation, MongoDB storage and custom loader
│   ├── mongodb_retraining.py        # ML model retraining and GridFS persistence
│   ├── deep_learning.py             # Lightweight PyTorch tabular neural network
│   └── ai_assistant.py              # Groq AI chat interface
├── requirements.txt                # Python dependencies
├── README.md                       # Project guide
├── MONGODB_SETUP.md                # MongoDB setup guide
├── GROQ_SETUP.md                   # Groq setup guide
├── .streamlit/
│   └── secrets.example.toml         # Safe Secrets template (no live credentials)
├── assets/                         # Team images and project media
├── archive/legacy/                 # Historical app snapshots; not imported at runtime
└── tests/                          # Lightweight project checks
```

## Fresh UI state and data safety

Sidebar navigation and tab changes use Streamlit's native rerun lifecycle without globally clearing widget state, reducing unnecessary redraws and flicker. A browser refresh starts a new Streamlit session. MongoDB documents, uploaded datasets, GridFS model artifacts, and training logs remain persistent. Realtime weather uses approximate IP geolocation and the Open-Meteo current-conditions API; it refreshes from cache every ten minutes.

## Security and operations

- MongoDB URI and Groq API key are read server-side from Streamlit Secrets or environment variables.
- Allow only the outbound IP address of the machine/server in MongoDB Atlas. Avoid leaving `0.0.0.0/0` enabled.
- Use a dedicated MongoDB database user with `readWrite` access to `datasphere`.
- Rotate credentials if a live Secrets file has ever been shared or committed.
- Do not upload private or restricted data.
- Install dependencies from `requirements.txt`; do not package `.venv` or real `secrets.toml`.

## Validation

Run the lightweight neural-network smoke tests with:

```bash
python -m unittest discover -s tests -v
```

The project can also be checked for syntax with `python -m compileall -q .`. Full browser rendering, live MongoDB Atlas operations, and Groq API calls require your own configured environment and credentials.
