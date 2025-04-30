#!/bin/bash
set -eu

PROJECT_ID=$(gcloud config get-value project)
REGION=us-east4
REPO_NAME=reranker
TAG=kagi-hn-reranker
IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${TAG}:latest

# one-time auth
gcloud auth configure-docker ${REGION}-docker.pkg.dev
gcloud services enable artifactregistry.googleapis.com

echo "🔍 Checking if repo '${REPO_NAME}' exists in '${REGION}'…"
if ! gcloud artifacts repositories describe "${REPO_NAME}" --location="${REGION}" >/dev/null 2>&1; then
  echo "📦 Repository not found — creating it now."
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Images for reranker service"
else
  echo "✅ Repository already exists."
fi

# Build and push
docker buildx create --use --name reranker_builder >/dev/null 2>&1 || true
docker buildx build \
  --platform linux/amd64 \
  --provenance=false \
  --tag ${IMAGE_URI} \
  --push .

gcloud run deploy ${TAG} \
  --image ${IMAGE_URI} \
  --region us-east4 \
  --cpu 8 --memory 8Gi \
  --cpu-boost \
  --no-cpu-throttling \
  --execution-environment gen1 \
  --min-instances 1 --max-instances 1
