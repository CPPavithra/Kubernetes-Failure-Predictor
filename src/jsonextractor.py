import json
from datetime import datetime, timedelta
import time
from kubernetes import client, config
from kubernetes.client.rest import ApiException

# Load Kubernetes config
config.load_kube_config()
v1 = client.CoreV1Api()
apps_v1 = client.AppsV1Api()


def jsonExtractor(data):
    solution = data.get("solution_function")
    rollback = data.get("rollback_function")
    print("Solution Function:", solution)
    print("Rollback Function:", rollback)
    return solution, rollback

def get_first_pod_name_from_deployment(deployment_name, namespace):
    try:
        # Fetch deployment details
        deployment = apps_v1.read_namespaced_deployment(deployment_name, namespace)
        selector = deployment.spec.selector.match_labels
        label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])
        # List pods based on label selector
        pods = v1.list_namespaced_pod(namespace=namespace, label_selector=label_selector)
        if pods.items:
            pod_name = pods.items[0].metadata.name  # Correctly reference the first pod's name
            print(f"Found pod name: {pod_name}")  # Add logging to confirm pod found
            return pod_name
        else:
            print(f"No pods found for deployment '{deployment_name}' in namespace '{namespace}'")
            return None
    except ApiException as e:
        print(f" Error fetching pods for deployment: {e}")
        return None


def generate_patch_from_pod_json(pod_json, memory_request=None, memory_limit=None, pod_name=None, namespace=None):
    if not pod_json and pod_name and namespace:
        try:
            pod_json = v1.read_namespaced_pod(pod_name, namespace).to_dict()
        except ApiException as e:
            if e.status == 404:
                print(f" Pod '{pod_name}' not found in namespace '{namespace}'. Cannot proceed.")
                return None
            else:
                raise

    if not pod_json or 'spec' not in pod_json or 'containers' not in pod_json['spec']:
        raise ValueError("Invalid or missing pod JSON data.")

    containers = pod_json['spec']['containers']
    patch_containers = []

    for container in containers:
        container_name = container['name']

        # Ensure resources is not None before accessing it
        resources = container.get('resources', {})
        requests = resources.get('requests', {})
        limits = resources.get('limits', {})

        # Use default values if resources, requests, or limits are not set
        current_memory_request = requests.get('memory', '256Mi') if requests else '256Mi'
        current_memory_limit = limits.get('memory', '512Mi') if limits else '512Mi'

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

def diagnose_and_fix_pod(deployment_name, namespace, patch_body):
    if patch_body is None:
        print("Patch body is missing. Cannot proceed.")
        return

    try:
        apps_v1.patch_namespaced_deployment(deployment_name, namespace, patch_body)
        print("✅ Patched deployment with updated resource settings.")
    except ApiException as e:
        print(f"Failed to patch deployment: {e}")

def fix_image_pull_error(json_input):
    name = json_input['deployment_name']
    namespace = json_input['namespace']
    correct_image = json_input['correct_image']
    image_pull_secrets = json_input.get('image_pull_secrets', [])

    try:
        deployment = apps_v1.read_namespaced_deployment(name=name, namespace=namespace)
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

        apps_v1.patch_namespaced_deployment(name=name, namespace=namespace, body=patch_body)
        print("✅ Fixed image pull error by updating image and secrets.")
    except ApiException as e:
        print(f" Failed to patch image or secrets: {e}")


def scale_deployment(deployment_name, namespace, replicas):
    scale = {"spec": {"replicas": replicas}}
    try:
        apps_v1.patch_namespaced_deployment_scale(deployment_name, namespace, scale)
        print(f"✅ Scaled deployment {deployment_name} to {replicas} replicas.")
    except ApiException as e:
        print(f"Failed to scale deployment: {e}")


def delete_pod(pod_name, namespace):
    try:
        v1.delete_namespaced_pod(name=pod_name, namespace=namespace)
        print(f"♻️ Deleted pod {pod_name} for restart.")
    except ApiException as e:
        print(f"Failed to delete pod: {e}")


# 🔍 Natural Language Matcher
ACTION_KEYWORDS = {
    "high memory usage": "adjust_memory_limits",
    "memory limit": "adjust_memory_limits",
    "container logs": "print_logs",
    "restart": "restart_container",
    "scale up": "scale_deployment",
    "image pull": "fix_image_pull_error",
    "access denied": "fix_image_pull_error",
    "cpu limit": "adjust_cpu_limits",
    "container resource limits": "adjust_resource_limits",
    "node resources": "increase_node_resources",
    "network connectivity": "check_network_connectivity",
    "pod events": "inspect_pod_events",
    "liveness readiness": "check_liveness_readiness",
    "rebuild image": "rebuild_and_redeploy_image",
    "rollback": "rollback_changes",
    "increase resource limits (memory)": "increase_memory_limits"
}


def solution_implementation(solution_steps, deployment_name, namespace, pod_name="demo-deployment-6d6c8487f6-d2bw9", pod_json=None, json_input=None):
    # Instead of dynamically fetching the pod name, we now use the hardcoded name
    if not pod_name:
        print("🚫 No pod found to act on. Skipping solution.")
        return

    if isinstance(solution_steps, str):
        solution_steps = [solution_steps]

    for step in solution_steps:
        action = None
        for keyword, mapped_action in ACTION_KEYWORDS.items():
            if keyword in step.lower():
                action = mapped_action
                break

        if action == "adjust_memory_limits":
            patch_body = generate_patch_from_pod_json(pod_json, pod_name=pod_name, namespace=namespace)
            diagnose_and_fix_pod(deployment_name, namespace, patch_body)

        elif action == "adjust_cpu_limits":
            patch_body = generate_patch_from_pod_json(pod_json, memory_request="512Mi", memory_limit="1Gi")
            diagnose_and_fix_pod(deployment_name, namespace, patch_body)

        elif action == "print_logs":
            try:
                logs = v1.read_namespaced_pod_log(pod_name, namespace, previous=True, tail_lines=50)
                print("🔍 Recent logs:")
                print(logs)
            except ApiException as e:
                print(f"Could not fetch logs: {e}")

        elif action == "restart_container":
            delete_pod(pod_name, namespace)

        elif action == "scale_deployment":
            scale_deployment(deployment_name, namespace, replicas=3)
        elif action == "increase_memory_limits":


            print("Increasing memory limits for deployment...")
            patch_body = generate_patch_from_pod_json(
                pod_json, 
                memory_request="512Mi", 
                memory_limit="1Gi"
            )
            diagnose_and_fix_pod(deployment_name, namespace, patch_body)

        elif action == "fix_image_pull_error":
            fix_image_pull_error(json_input)

        elif action == "adjust_resource_limits":
            patch_body = generate_patch_from_pod_json(pod_json, pod_name=pod_name, namespace=namespace)
            diagnose_and_fix_pod(deployment_name, namespace, patch_body)

        elif action == "increase_node_resources":
            print("Considering increasing node resources (CPU/Memory). Adjusting settings as necessary.")
            # Add code to modify node resources here

        elif action == "check_network_connectivity":
            print("Check for network connectivity issues, especially if the container has issues pulling images or communicating with other services.")

        elif action == "inspect_pod_events":
            print("Inspect Kubernetes events for the failing pods to gather more info.")
            # Add code to inspect pod events here

        elif action == "check_liveness_readiness":
            print("Review and adjust liveness and readiness probes for better health checks.")
            # Add code to inspect and modify probes here

        elif action == "rebuild_and_redeploy_image":
            print("Rebuilding and redeploying the container image.")
            # Add code to rebuild and redeploy image here

        elif action == "rollback_changes":
            print("Rolling back to a previous version of the deployment.")
            # Add rollback logic here

        else:
            print(f" ")

