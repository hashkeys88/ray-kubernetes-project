# Ray on Kubernetes Custom Image
FROM rayproject/ray:2.8.0-py310

# Install additional dependencies
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scikit-learn==1.3.0 \
    prometheus-client==0.17.1 \
    psutil==5.9.5

# Create working directory
WORKDIR /app

# Copy application code
COPY ray-apps/ /app/ray-apps/
COPY monitoring/ /app/monitoring/
COPY scripts/ /app/scripts/

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Expose ports
EXPOSE 8265 8000 8080

# Default command
CMD ["ray", "start", "--head", "--port=6379", "--dashboard-host=0.0.0.0", "--dashboard-port=8265"]
