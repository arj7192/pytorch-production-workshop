#!/usr/bin/env bash
# Deploy the PyTorch inference API to Google Cloud Run
#
# Usage:
#   ./deploy.sh <GCP_PROJECT_ID> [--gpu] [--region REGION]
#
# Options:
#   --gpu         Deploy with NVIDIA L4 GPU (requires GPU quota)
#   --region      GCP region (default: us-central1)
#
# Prerequisites:
#   - Docker installed
#   - gcloud CLI installed and authenticated
#   - Cloud Run API enabled: gcloud services enable run.googleapis.com

set -euo pipefail

GCP_PROJECT=""
USE_GPU=false
REGION="us-central1"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)    USE_GPU=true; shift ;;
        --region) REGION="$2"; shift 2 ;;
        -*)       echo "Unknown flag: $1"; exit 1 ;;
        *)        GCP_PROJECT="$1"; shift ;;
    esac
done

if [[ -z "${GCP_PROJECT}" ]]; then
    echo "Usage: ./deploy.sh <GCP_PROJECT_ID> [--gpu] [--region REGION]"
    exit 1
fi

SERVICE_NAME="pytorch-workshop-api"
IMAGE="gcr.io/${GCP_PROJECT}/${SERVICE_NAME}"

echo "=== Deploying ${SERVICE_NAME} ==="
echo "Project: ${GCP_PROJECT}"
echo "Region:  ${REGION}"
echo "GPU:     ${USE_GPU}"
echo "Image:   ${IMAGE}"
echo ""

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

echo "1. Building and pushing container image..."
docker build -f serve/Dockerfile -t "${IMAGE}" .
gcloud auth configure-docker --quiet 2>/dev/null
docker push "${IMAGE}"

echo ""
echo "2. Deploying to Cloud Run..."

if ${USE_GPU}; then
    gcloud run deploy "${SERVICE_NAME}" \
        --project "${GCP_PROJECT}" \
        --image "${IMAGE}" \
        --platform managed \
        --region "${REGION}" \
        --port 8000 \
        --memory 4Gi \
        --cpu 4 \
        --gpu 1 \
        --gpu-type nvidia-l4 \
        --timeout 60 \
        --concurrency 10 \
        --min-instances 0 \
        --max-instances 3 \
        --allow-unauthenticated
else
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
fi

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
