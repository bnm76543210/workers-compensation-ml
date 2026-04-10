#!/bin/bash
# ClearML Serving deployment script
# Run this AFTER experiment2.py has completed and registered the model in ClearML

set -e

echo "=== ClearML Serving Deployment ==="
echo ""

# Step 1: Install clearml-serving
pip install clearml-serving

# Step 2: Create serving service (run once)
echo "Creating ClearML Serving service..."
clearml-serving create --name "Workers Compensation Serving"

# Step 3: Get the service ID from output above, then:
# Replace <SERVICE_ID> with the actual ID printed above
SERVICE_ID=${1:-"<SERVICE_ID>"}

echo ""
echo "Step 3: Registering model endpoint..."
clearml-serving --id "$SERVICE_ID" model add \
  --engine sklearn \
  --endpoint "workers_compensation" \
  --preprocess "clearml_serving/preprocess.py" \
  --name "Workers Comp XGBoost" \
  --project "Workers Compensation"

echo ""
echo "Step 4: Starting inference container..."
docker run -d \
  --name clearml-serving-inference \
  -v ~/clearml.conf:/root/clearml.conf \
  -p 8080:8080 \
  -e CLEARML_SERVING_TASK_ID="$SERVICE_ID" \
  -e CLEARML_SERVING_POLL_FREQ=5 \
  clearml/clearml-serving:latest

echo ""
echo "=== Deployment complete! ==="
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
