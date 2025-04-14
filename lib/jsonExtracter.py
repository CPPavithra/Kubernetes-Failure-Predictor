import json
from datetime import datetime, timedelta
import time
from kubernetes import client, config
try 
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


def generate_patch_from_pod_json(pod_json, memory_request=None, memory_limit=None):
    """
    Extracts container names and their current memory requests and limits from a pod JSON.
    If new memory values are provided, it will update them in the patch body.
    """
    containers = pod_json['spec']['containers']
    patch_containers = []

    for container in containers:
        container_name = container['name']

        # Extract current memory requests and limits
        current_memory_request = container['resources']['requests'].get('memory', '256Mi')
        current_memory_limit = container['resources']['limits'].get('memory', '512Mi')

        # If new memory values are provided, use those
        memory_request = memory_request or current_memory_request
        memory_limit = memory_limit or current_memory_limit

        patch_containers.append({
            "name": container_name,
            "resources": {
                "requests": {
                    "memory": memory_request
                },
                "limits": {
                    "memory": memory_limit
                }
            }
        })

    patch_body = {
        "spec": {
            "template": {
                "spec": {
                    "containers": patch_containers
                }
            }
        }
    }

    return patch_body



def diagnose_and_fix_pod(pod_name, namespace, patch_body):
    apps_v1.patch_namespaced_deployment("webapp", namespace, patch_body)
    print("Patched deployment with higher memory to avoid OOM.")
    return


def fix_image_pull_error(json_input):
    config.load_kube_config()
    v1 = client.CoreV1Api()
    apps_v1 = client.AppsV1Api()

# {
#   "deployment_name": "my-deployment",
#   "namespace": "default",
#   "correct_image": "nginx:latest",
#   "image_pull_secrets": ["my-docker-secret"]
# }
    name = json_input['deployment_name']
    namespace = json_input['namespace']
    correct_image = json_input['correct_image']
    image_pull_secrets = json_input.get('image_pull_secrets', [])

    # Get deployment
    deployment = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)

    # Patch image and imagePullSecrets
    container_name = deployment.spec.template.spec.containers[0].name

    patch_body = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": container_name,
                        "image": correct_image
                    }],
                    "imagePullSecrets": [{"name": secret} for secret in image_pull_secrets]
                }
            }
        }
    }

    # Apply patch
    apps_v1.patch_namespaced_deployment(
        name=name,
        namespace=namespace,
        body=patch_body
    )



def solution_implementation(solution, rollback):
    if solution = "OOMKilled":
        #printing the error
        logs = v1.read_namespaced_pod_log(pod_name, namespace, previous=True, tail_lines=50)
        print(logs)
        patch_body = generate_patch_from_pod_json(pod_json)
        diagnose_and_fix_pod(patch_body)
        return 
    if solution = "failed to pull image: access denied"
        fix_image_pull_error(json_input)
    if solution = "scale_deployment":
        patch_namespaced_deployment()
        return
    if solution = "CPU_limit":
        patch_namespaced_deployment()
        return
