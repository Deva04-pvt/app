#!/bin/bash

# Alternative startup script using uvicorn directly
export PORT=${PORT:-8000}

# Start the server with uvicorn
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
