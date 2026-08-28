# FABINTEL Deployment Guide

This guide details how to deploy the FABINTEL Semiconductor Defect & Process Intelligence Platform using a standard **GitHub → Docker → Render** pipeline.

## 1. Prerequisites
- A GitHub account.
- A Render account (https://render.com).

## 2. Push to GitHub
First, you must commit and push this repository to GitHub.

1. Open your terminal in the project root (`d:\ALK` locally).
2. Initialize and push the repository:
```bash
git init
git add .
git commit -m "Initial commit for FABINTEL deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main
```
*(Make sure not to push unnecessary large files. The `.gitignore` should match the provided `.dockerignore` for best results).*

## 3. Deploy on Render
Render natively supports Dockerfile deployments from GitHub repositories.

1. Log in to Render.
2. Click **New +** and select **Web Service**.
3. Connect your GitHub account (if not already connected).
4. Select the repository you just pushed.
5. In the configuration screen, apply the following settings:
   - **Name:** `fabintel-dashboard` (or your preferred name)
   - **Region:** Choose the region closest to your users.
   - **Branch:** `main`
   - **Runtime:** `Docker` (Render should auto-detect the `Dockerfile` at the root).
   - **Instance Type:** Choose at least the **Starter** tier (512MB RAM). *Note: Due to TensorFlow and XGBoost, a tier with 1GB+ RAM is highly recommended to prevent out-of-memory errors during startup.*
6. Click **Create Web Service**.

## 4. Environment Variables (Optional but Recommended)
Render dynamically assigns a `PORT` environment variable to your container. The provided `Dockerfile` automatically handles this:
`CMD streamlit run src/dashboard/app.py --server.port="${PORT:-10000}" --server.address="0.0.0.0"`

No additional environment variables are strictly required to run the dashboard.

## 5. Wait for Build
Render will now pull the repository, build the Docker container using the `Dockerfile`, install the dependencies from `requirements.txt`, and deploy the container. 
- You can monitor the build progress in the Render Logs tab.
- Once the deployment shows **Live**, you can click the generated Render URL (e.g., `https://fabintel-dashboard.onrender.com`) to access your industrial dashboard.
