FROM public.ecr.aws/lambda/python:3.11

# System deps for torch CPU
RUN dnf install -y gcc-c++ && dnf clean all

# Install Python deps (no CUDA — Lambda is CPU-only)
COPY requirements-lambda.txt .
RUN pip install --no-cache-dir -r requirements-lambda.txt

# Copy handler
COPY handler.py ${LAMBDA_TASK_ROOT}/

# Copy quantized model (baked into image for fast cold start)
COPY model/ ${LAMBDA_TASK_ROOT}/model/

# Point handler at baked-in model
ENV MODEL_DIR=${LAMBDA_TASK_ROOT}/model

CMD ["handler.lambda_handler"]
