#!/usr/bin/env bash
# Deploy the PyTorch inference API to Google Cloud Run
#
# Usage:
#   ./deploy.sh <GCP_PROJECT_ID> [REGION]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Cloud Run API enabled: gcloud services enable run.googleapis.com
#   - Cloud Build API enabled: gcloud services enable cloudbuild.googleapis.com

set -euo pipefail

GCP_PROJECT="${1:?Usage: ./deploy.sh <GCP_PROJECT_ID> [REGION]}"
REGION="${2:-us-central1}"
SERVICE_NAME="pytorch-workshop-api"
IMAGE="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}"

echo "=== Deploying ${SERVICE_NAME} ==="
echo "Project: ${GCP_PROJECT}"
echo "Region:  ${REGION}"
echo "Image:   ${IMAGE}"
echo ""

# Build from the repo root (need both serve/ and src/)
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "1. Building container image..."
gcloud builds submit \
    --project "${GCP_PROJECT}" \
    --tag "${IMAGE}" \
    --timeout 900 \
    -f serve/Dockerfile \
    .

echo ""
echo "2. Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
    --project "${GCP_PROJECT}" \
    --image "${IMAGE}" \
    --platform managed \
    --region "${REGION}" \
    --port 8000 \
    --memory 2Gi \
    --cpu 2 \
    --timeout 60 \
    --concurrency 10 \
    --min-instances 0 \
    --max-instances 5 \
    --allow-unauthenticated

echo ""
echo "=== Deployment complete ==="
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project "${GCP_PROJECT}" \
    --region "${REGION}" \
    --format "value(status.url)")

echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test with:"
echo "  curl ${SERVICE_URL}/health"
echo "  curl -X POST ${SERVICE_URL}/generate -H 'Content-Type: application/json' -d '{\"prompt\": \"Hello\", \"max_tokens\": 30}'"
