#!/bin/bash
set -e

if [ -z "${WANDB_API_KEY}" ]; then
    echo "❌ WANDB_API_KEY est vide ou absent — impossible de charger le modèle depuis W&B."
    echo "   Définissez la variable dans votre .env ou via docker compose."
    exit 1
fi

echo "========================================="
echo " Inference API — démarrage"
echo "========================================="
exec uvicorn radiocovid.inference.api:app --host 0.0.0.0 --port 8000
