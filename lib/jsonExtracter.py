import json

def jsonExtractor(data):

#   "Error": [
#     {
#       "timestamp": "2025-04-12T16:35:21Z",
#       "namespace": "default",
#       "pod_name": "webapp-5f4b9cbb9d-xyz12",
#       "container_name": "webapp",
#       "node_name": "worker-node-1",
#       "error_type": "NetworkError",
#       "message": "Readiness probe failed: Get http://10.42.0.15:8080/health: dial tcp 10.42.0.15:8080: connect: connection refused",
#       "source": "kubelet",
#       "event_reason": "Unhealthy",
#       "probe_type": "Readiness",
#       "target_ip": "10.42.0.15",
#       "target_port": 8080,
#       "protocol": "HTTP"
#     }
#   ],
#   "solution_function": "solution_func()",
#   "rollback_function": "rollback_func()"
# }

# Extract values
    solution = data.get("solution_function")
    rollback = data.get("rollback_function")

    print("Solution Function:", solution)
    print("Rollback Function:", rollback)
    return solution, rollback
def solution_implementation(solution, rollback):
    if solution = "network_failure"
        network_failure()

def network_failure():
    print("Entered Network failure")
