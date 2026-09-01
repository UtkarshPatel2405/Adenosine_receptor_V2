# Deploying to Hugging Face Spaces

This guide walks you through deploying the **Adeno_Advance** FastAPI + HTML/JS application to Hugging Face Spaces.

## 🚀 Steps to Deploy

### 1. Create a New Space on Hugging Face
1. Go to your Hugging Face account: [huggingface.co/Utkarsh2405](https://huggingface.co/Utkarsh2405)
2. Click on **New** (or "+" icon) in the top-right and select **Space**.
3. Fill in the following details:
   * **Space Name**: `Adeno_Advance` (or any name you prefer)
   * **License**: `mit` (or leave blank)
   * **SDK**: Select **Gradio** (This is 100% Free! Even though it is labeled Gradio, our root `app.py` will route traffic to your FastAPI server).
   * **Gradio Template**: Select **Blank** (do not select chatbot, text-to-image, etc.)
   * **Space Hardware**: **CPU basic** (Free, 16GB RAM)
   * **Visibility**: **Public** (or Private if you prefer)
4. Click **Create Space**.

---

### 2. Connect and Push from GitHub (Automatic Sync)
Since your code is already pushed to GitHub at [UtkarshPatel2405/Adeno_Advance](https://github.com/UtkarshPatel2405/Adeno_Advance), you can set up GitHub Actions to automatically deploy to Hugging Face:

#### Step A: Generate a Hugging Face Write Token
1. Go to your Hugging Face settings: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **Create new token**.
3. Set **Type** to `Write`.
4. Name the token (e.g., `github-deploy`) and copy it.

#### Step B: Add Secret to GitHub
1. Go to your GitHub Repository: `https://github.com/UtkarshPatel2405/Adeno_Advance`
2. Go to **Settings** -> **Secrets and variables** -> **Actions**.
3. Click **New repository secret**.
4. Name: `HF_TOKEN`
5. Value: *Paste the Hugging Face Write Token you copied*.
6. Click **Add secret**.

#### Step C: Add GitHub Actions Workflow
We have configured a GitHub Actions workflow in `.github/workflows/deploy.yml` that automatically pushes your code to Hugging Face whenever you push to GitHub's `main` branch.

---

### 3. Verification
Once pushed, Hugging Face will automatically read the `Dockerfile`, build the container, and run it. The FastAPI app (configured with the custom HTML/JS frontend) will be live!
