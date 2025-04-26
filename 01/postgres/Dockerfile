FROM postgres:15

RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev

# Install Python packages and Clean up
RUN pip3 install --break-system-packages --no-cache-dir psycopg2-binary jupyter pandas pycodestyle numpy && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*
