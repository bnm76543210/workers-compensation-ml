#!/bin/bash
# ClearML Serving deployment script
# Run this AFTER experiment2.py has completed and registered the model in ClearML

set -e

echo "=== ClearML Serving Deployment ==="
echo ""

# Serving service and model IDs (created 2026-04-10)
SERVICE_ID="f4898d5556a942c19f571692a97737a6"
MODEL_ID="c69c03966cad4898bcc6cdd8de3d61db"

# Step 1: Install clearml-serving (if not already installed)
pip install clearml-serving

# Step 2: Register model endpoint (already done, kept for reference)
# clearml-serving --id "$SERVICE_ID" model add \
#   --engine sklearn \
#   --endpoint "workers_compensation" \
#   --preprocess "clearml_serving/preprocess.py" \
#   --model-id "$MODEL_ID"

# Step 3: Start inference container
echo "Starting inference container..."
docker run -d \
  --name clearml-serving-inference \
  -v ~/clearml.conf:/root/clearml.conf \
  -p 8080:8080 \
  -e CLEARML_SERVING_TASK_ID="$SERVICE_ID" \
  -e CLEARML_SERVING_POLL_FREQ=5 \
  clearml/clearml-serving:latest

echo ""
echo "=== Deployment complete! ==="
echo "Serving service ID: $SERVICE_ID"
echo "Model ID:           $MODEL_ID"
echo "Inference endpoint: http://localhost:8080/serve/workers_compensation"
echo ""
echo "Test with:"
echo 'curl -X POST "http://localhost:8080/serve/workers_compensation" \'
echo '  -H "Content-Type: application/json" \'
echo '  -d '"'"'{"Age": 35, "WeeklyPay": 500, "InitialCaseEstimate": 5000,
  "HoursWorkedPerWeek": 40, "DaysWorkedPerWeek": 5,
  "Gender": "M", "MaritalStatus": "S",
  "DependentChildren": 0, "DependentsOther": 0,
  "PartTimeFullTime": "F",
  "Accident_Year": 2010, "Accident_Month": 6, "Accident_DayOfWeek": 2,
  "Reported_Year": 2010, "Reported_Month": 7, "Reported_DayOfWeek": 1,
  "ReportDelay_Days": 30}'"'"
