#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Setup GCP Project, Artifact Registry, Service Account & Workload Identity
# Federation (WIF) for GitHub Actions CI/CD deployment of Pulsar MCP.
# ==============================================================================

PROJECT_ID="${1:-pulsar-mcp-prod}"
REGION="${2:-us-central1}"
GITHUB_REPO="${3:-Krv-Labs/pulsar}"
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
SA_NAME="pulsar-mcp-deployer"
ARTIFACT_REPO="pulsar-mcp"
SERVICE_NAME="pulsar-mcp"

echo "======================================================================"
echo "Initializing GCP setup for project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "GitHub Repo: ${GITHUB_REPO}"
echo "======================================================================"

# 1. Create GCP Project if it doesn't exist
if ! gcloud projects describe "${PROJECT_ID}" &>/dev/null; then
  echo "--> Creating new GCP project: ${PROJECT_ID}..."
  gcloud projects create "${PROJECT_ID}" --name="Pulsar MCP Production"
else
  echo "--> GCP project ${PROJECT_ID} already exists."
fi

gcloud config set project "${PROJECT_ID}"

# 2. Enable Required GCP APIs
echo "--> Enabling required Google Cloud APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  iam.googleapis.com

# 3. Create Artifact Registry Docker repository
if ! gcloud artifacts repositories describe "${ARTIFACT_REPO}" --location="${REGION}" &>/dev/null; then
  echo "--> Creating Artifact Registry Docker repository: ${ARTIFACT_REPO}..."
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Pulsar MCP Docker Container Repository"
else
  echo "--> Artifact Registry repository ${ARTIFACT_REPO} already exists."
fi

# 4. Create Service Account for GitHub Actions deployment
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" &>/dev/null; then
  echo "--> Creating Service Account: ${SA_EMAIL}..."
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Pulsar MCP Deployment Service Account"
else
  echo "--> Service Account ${SA_EMAIL} already exists."
fi

# Grant IAM Roles to Service Account
echo "--> Granting IAM roles to Service Account..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/artifactregistry.writer" --condition=None

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.developer" --condition=None

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/iam.serviceAccountUser" --condition=None

# 5. Create Workload Identity Pool
if ! gcloud iam workload-identity-pools describe "${POOL_NAME}" --location="global" &>/dev/null; then
  echo "--> Creating Workload Identity Pool: ${POOL_NAME}..."
  gcloud iam workload-identity-pools create "${POOL_NAME}" \
    --location="global" \
    --display-name="GitHub Actions Pool"
else
  echo "--> Workload Identity Pool ${POOL_NAME} already exists."
fi

WORKLOAD_IDENTITY_POOL_ID=$(gcloud iam workload-identity-pools describe "${POOL_NAME}" \
  --location="global" \
  --format="value(name)")

# 6. Create Workload Identity Provider for GitHub OIDC
if ! gcloud iam workload-identity-pools providers describe "${PROVIDER_NAME}" \
  --workload-identity-pool="${POOL_NAME}" \
  --location="global" &>/dev/null; then
  echo "--> Creating Workload Identity Provider: ${PROVIDER_NAME}..."
  gcloud iam workload-identity-pools providers create-oidc "${PROVIDER_NAME}" \
    --location="global" \
    --workload-identity-pool="${POOL_NAME}" \
    --display-name="GitHub Provider" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'" \
    --issuer-uri="https://token.actions.githubusercontent.com"

else
  echo "--> Workload Identity Provider ${PROVIDER_NAME} already exists."
fi

WIF_PROVIDER_RESOURCE_NAME=$(gcloud iam workload-identity-pools providers describe "${PROVIDER_NAME}" \
  --workload-identity-pool="${POOL_NAME}" \
  --location="global" \
  --format="value(name)")

# 7. Bind Service Account to GitHub Repo OIDC Assertion
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
MEMBER_EXPRESSION="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL_NAME}/attribute.repository/${GITHUB_REPO}"

echo "--> Binding WIF policy to Service Account for repository ${GITHUB_REPO}..."
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --role="roles/iam.workloadIdentityUser" \
  --member="${MEMBER_EXPRESSION}"

echo "======================================================================"
echo "GCP Configuration Complete!"
echo "======================================================================"
echo "Add the following secrets/variables to your GitHub Repository Settings:"
echo ""
echo "  GCP_PROJECT_ID:               ${PROJECT_ID}"
echo "  GCP_SERVICE_ACCOUNT:          ${SA_EMAIL}"
echo "  GCP_WORKLOAD_IDENTITY_PROVIDER: ${WIF_PROVIDER_RESOURCE_NAME}"
echo "  GCP_REGION:                   ${REGION}"
echo "======================================================================"
