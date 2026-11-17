# MongoDB Atlas Setup for NusaInsight

## 1. Create an Atlas cluster and database user

1. Open https://www.mongodb.com/atlas and sign in.
2. Create a project and a cluster. The Free tier is suitable for development and coursework.
3. Open **Security → Network Access → Add IP Address**. Add the outbound IP address of the computer or server running Streamlit. Do not keep `0.0.0.0/0` as a permanent rule.
4. Open **Security → Database Access → Add New Database User**. Create an application-specific username and password, then grant `readWrite` access to the `datasphere` database. This database user is different from your Atlas website account.
5. On the cluster page, choose **Connect → Drivers → Python**. Copy the `mongodb+srv://...` connection string and replace the username, password, and cluster host with the values provided by Atlas.

## 2. Store the connection string in Streamlit Secrets

Create `.streamlit/secrets.toml` beside `app.py` using `.streamlit/secrets.example.toml` as a template:

```toml
mongo_uri = "mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/?retryWrites=true&w=majority"
mongo_database = "datasphere"
```

Do not put the URI in `app.py`, commit it to Git, or send `secrets.toml` to anyone. The file is excluded by `.gitignore`. URL-encode special password characters such as `@`, `:`, `/`, `#`, or `%`.

## 3. Start the application and test MongoDB

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open **Machine Learning Lab → MongoDB & Retraining**. The retraining workspace uses the same server-side database connection as Home, Community, and custom dataset uploads. If the database is unavailable, model artifacts and training history cannot be saved.

## 4. Collections used by the application

| Collection | Purpose |
|---|---|
| `community_posts` | Community comments, replies, and reactions |
| `training_logs` | Training and retraining history, parameters, and metrics |
| `dataset_uploads` | Metadata for uploaded CSV files |
| `custom_datasets` | Catalog of user-uploaded datasets |
| `custom_data_<name>_<id>` | Rows for an individual custom dataset upload |
| `model_artifacts.files` and `model_artifacts.chunks` | Model files stored through GridFS |

MongoDB creates collections on the first write; they do not need to be created manually.

## 5. Retraining workflow

The **MongoDB & Retraining** tab focuses on model retraining. Choose **Regression**, **Classification**, or **Clustering**, select a compatible dataset and model, set the random seed and test size (when applicable), then run retraining. The app saves the fitted model artifact in GridFS and the evaluation metrics in `training_logs`. All history records remain stored in MongoDB without a retention limit; the UI shows a compact scrollable view. When a previous result exists for the same task and dataset, the workspace compares the latest metrics with that saved result.

Custom CSV datasets uploaded in **Dataset Explorer → Upload & Test a New Dataset** are stored in `custom_datasets` and their corresponding `custom_data_<name>_<id>` collection. The same catalog is read by Home, Dataset Explorer, Machine Learning Lab, Deep Learning, and MongoDB & Retraining. Classification and Regression require a target column. Clustering is unsupervised and requires at least two numeric features.

## 6. Troubleshooting

| Message or issue | What to check |
|---|---|
| Authentication failed | Use an Atlas Database User and verify the password. |
| IP access denied | Add the actual outbound IP of the machine or server running Streamlit. |
| DNS/SRV error | Use the Atlas `mongodb+srv://` URI and install `pymongo[srv]`. |
| Secrets not found | Put `secrets.toml` in `.streamlit/` beside `app.py`, then restart Streamlit. |
| GridFS save failed | Check `readWrite` permissions and available cluster storage. |

## Security notes

- MongoDB credentials are read by the Python server and are never rendered in the browser.
- Use a database user with only the permissions required for `datasphere`; do not use an Atlas administrator account.
- Only load model artifacts from a trusted database. Joblib/pickle deserialization can execute code.
- Do not include `.venv` or `secrets.toml` when sharing the project. Reinstall packages from `requirements.txt`.
