# Deployment Readiness Checklist

Use this before deploying a model to production.

---

## Model Readiness

- [ ] **Model in eval mode**: `model.eval()` -  disables dropout, batchnorm uses running stats
- [ ] **No grad context**: All inference wrapped in `torch.no_grad()` or `torch.inference_mode()`
- [ ] **Deterministic output**: Same input → same output (set seeds, disable dropout)
- [ ] **Input validation**: Handle malformed inputs gracefully (empty strings, too-long sequences)
- [ ] **Output bounds**: Verify outputs are reasonable (e.g., probabilities sum to 1)
- [ ] **Performance baseline**: Know your P50/P95/P99 latency targets

## Export & Optimization

- [ ] **Export format chosen**: TorchScript for C++/mobile, ONNX for cross-platform
- [ ] **Export verified**: Output matches original model within tolerance
- [ ] **Quantization** (if CPU): Dynamic quantization for Linear-heavy models
- [ ] **Model size**: Acceptable for deployment target (container, mobile, edge)
- [ ] **Cold start time**: Model loads in < 10s (or use warm instances)

## API Design

- [ ] **Health endpoint**: `GET /health` returns model status (for load balancer probes)
- [ ] **Input schema**: Validated with Pydantic or equivalent
- [ ] **Error responses**: Structured JSON errors with appropriate HTTP status codes
- [ ] **Timeouts**: Request timeout shorter than gateway timeout
- [ ] **Rate limiting**: Prevent abuse (especially for GPU-backed inference)
- [ ] **CORS**: Configured appropriately for frontend consumers

## Container

- [ ] **Non-root user**: Never run as root in production
- [ ] **Minimal image**: Multi-stage build, no dev tools in runtime image
- [ ] **Health check**: Docker HEALTHCHECK or Kubernetes liveness probe
- [ ] **Graceful shutdown**: Handle SIGTERM properly
- [ ] **No secrets in image**: Use environment variables or secret managers
- [ ] **Reproducible build**: Pin all dependency versions

## Cloud Deployment (Cloud Run / Kubernetes)

- [ ] **Resource limits**: CPU and memory limits set appropriately
- [ ] **Scaling**: Min/max instances configured
- [ ] **Concurrency**: Set based on model's throughput capacity - CPU inference: concurrency = 10-50 - GPU inference: concurrency = 1-4
- [ ] **Startup probe**: Allow time for model loading
- [ ] **Region**: Deploy close to users

## Monitoring & Observability

- [ ] **Request logging**: Log request ID, latency, input size, output size
- [ ] **Error alerting**: Alert on error rate > 1%
- [ ] **Latency alerting**: Alert on P99 > threshold
- [ ] **Memory monitoring**: Alert on > 80% memory usage
- [ ] **Model version tracking**: Know which model version is serving
- [ ] **Dashboards**: Latency, error rate, throughput, resource usage

## Security

- [ ] **Authentication**: API key, OAuth, or service account
- [ ] **Input sanitization**: Prevent injection attacks
- [ ] **Output filtering**: Remove sensitive information from responses
- [ ] **Network policy**: Restrict ingress/egress
- [ ] **Dependency scanning**: No known vulnerabilities in dependencies

---

## Cloud Run Quick Reference

```bash
# Deploy
gcloud run deploy SERVICE_NAME \
  --image gcr.io/PROJECT/IMAGE \
  --platform managed \
  --region us-central1 \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 60 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 10

# Check logs
gcloud run services logs read SERVICE_NAME --region us-central1

# Update traffic split (canary deployment)
gcloud run services update-traffic SERVICE_NAME \
  --to-revisions REVISION_1=90,REVISION_2=10
```

## Performance Tuning Guide

| Scenario | Recommended settings |
|----------|---------------------|
| Low-latency API | `min-instances=1`, `concurrency=10`, `cpu=2` |
| Batch processing | `min-instances=0`, `timeout=300`, `memory=4Gi` |
| GPU inference | `concurrency=1`, `memory=8Gi`, GPU accelerator |
| Cost-optimized | `min-instances=0`, `max-instances=3` |
