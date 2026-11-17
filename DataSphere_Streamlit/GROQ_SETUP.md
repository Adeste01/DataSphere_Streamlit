# Groq API Setup for NusaInsight

## Why does Home show “API key available”?

A populated `GROQ_API_KEY` only confirms that a value exists in Secrets. It does not prove that the key is valid, the network can reach Groq, the project can access the selected model, or a chat-completion request succeeds.

## 1. Create an API key

1. Open https://console.groq.com/keys.
2. Sign in to Groq and create an API key.
3. Copy the key when it is shown. Never share it in chat, screenshots, source code, or Git commits.

## 2. Store the key in Streamlit Secrets

For local development, create or edit `.streamlit/secrets.toml` beside `app.py`:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
```

For Streamlit Community Cloud, open the app settings, choose **Secrets**, and paste the same TOML entries. Restart the app after changing Secrets.

`GROQ_MODEL` is optional. The default is `openai/gpt-oss-120b`, which is also the AI Assistant default. If you choose another model, make sure it is available to your Groq project.

## 3. Test the connection from Home

1. Run `streamlit run app.py`.
2. Open **Home → System Status**.
3. Select **Test Groq connection**.
4. The app sends one short chat-completion request. The result is kept in session state so the API is not called again on every rerun.

## 4. Understand the test result

- **API active**: Groq returned a response from the selected model.
- **401 AuthenticationError**: The key is wrong, revoked, or invalid. Create a new key and update Secrets.
- **403 PermissionDeniedError**: The project or organization cannot use the selected model. Check model permissions or choose an allowed model.
- **404 NotFoundError**: The model ID was not found. Check `GROQ_MODEL` against the current model catalog.
- **429 RateLimitError**: The request limit or quota was reached. Wait for the limit to reset or check usage in the Groq Console.
- **APIConnectionError**: The app server cannot reach Groq. Check internet access, proxy, firewall, and DNS.
- **Missing library**: Run `pip install -r requirements.txt`.

Official references:
- https://console.groq.com/docs/quickstart
- https://console.groq.com/docs/api-reference
- https://console.groq.com/docs/models
