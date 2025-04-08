import os
import yaml
import json
import logging
from typing import Dict, Any, Optional
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from app.models.model_schemas import ModelMetadata, DeploymentRequest

logger = logging.getLogger("kubernetes-service")

class KubernetesService:
    def __init__(self, namespace: str):
        """Initialize Kubernetes client for model deployment."""
        self.namespace = namespace

        # Try to load configuration from service account or kubeconfig
        try:
            # When running in a pod
            config.load_incluster_config()
        except config.ConfigException:
            try:
                # When running locally
                config.load_kube_config()
            except config.ConfigException:
                logger.warning("Could not configure Kubernetes client")

        self.core_api = client.CoreV1Api()
        self.apps_api = client.AppsV1Api()
        self.custom_api = client.CustomObjectsApi()

    async def check_health(self) -> Dict[str, Any]:
        """Check Kubernetes service health."""
        try:
            # Test API connection by listing namespaces
            namespaces = self.core_api.list_namespace()
            return {
                "status": "healthy",
                "namespace_exists": any(ns.metadata.name == self.namespace for ns in namespaces.items),
                "connected": True
            }
        except Exception as e:
            logger.error(f"Kubernetes health check failed: {str(e)}")
            return {"status": "unhealthy", "error": str(e)}

    async def deploy_model(self, model: ModelMetadata, deployment_params: DeploymentRequest) -> Dict[str, Any]:
        """Deploy model to Triton Inference Server."""
        try:
            # Create model config map
            model_config = self._generate_model_config(model, deployment_params)
            config_map_name = f"{model.name}-{deployment_params.version}-config"

            # Create or update the ConfigMap
            try:
                self._create_model_config_map(config_map_name, model_config)
            except ApiException as e:
                if e.status == 409:  # Already exists
                    # Update existing config map
                    self._update_model_config_map(config_map_name, model_config)
                else:
                    raise

            # Create Triton Inference Server model deployment if not using CRD
            # In a more advanced setup, you'd use the TritonServer Custom Resource
            if not self._check_triton_crd_exists():
                self._create_or_update_triton_deployment(model, deployment_params)
            else:
                # Use TritonServer CRD if it exists
                self._create_or_update_triton_server_resource(model, deployment_params)

            # Return deployment information
            return {
                "model_name": model.name,
                "version": deployment_params.version,
                "config_map": config_map_name,
                "service_name": f"triton-inference-server",
                "status": "deploying",
                "resource": "TritonServer CRD" if self._check_triton_crd_exists() else "Deployment",
                "namespace": self.namespace,
                "timestamp": str(datetime.now())
            }
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            raise

    def _generate_model_config(self, model: ModelMetadata, params: DeploymentRequest) -> Dict[str, Any]:
        """Generate Triton model configuration."""
        # Base configuration structure
        config = {
            "name": model.name,
            "platform": self._map_framework_to_platform(model.framework),
            "max_batch_size": params.max_batch_size,
            "input": [],  # Should be filled with model-specific inputs
            "output": [],  # Should be filled with model-specific outputs
            "dynamic_batching": {} if params.dynamic_batching else None,
            "instance_group": [
                {
                    "count": params.instance_count,
                    "kind": "KIND_GPU" if params.instance_type == "gpu" else "KIND_CPU"
                }
            ],
            "version_policy": {
                "specific": {
                    "versions": [params.version]
                }
            }
        }

        # Add model-specific configs from metadata
        if model.metadata.get("inputs"):
            config["input"] = model.metadata.get("inputs")

        if model.metadata.get("outputs"):
            config["output"] = model.metadata.get("outputs")

        # Configure dynamic batching if enabled
        if params.dynamic_batching:
            config["dynamic_batching"] = {
                "preferred_batch_size": [1, 4, 8],
                "max_queue_delay_microseconds": 50000
            }

        # Cache configuration if specified
        if params.cache_config:
            config["cache_config"] = params.cache_config

        return config

    def _map_framework_to_platform(self, framework: str) -> str:
        """Map model framework to Triton platform name."""
        mapping = {
            "pytorch": "pytorch_libtorch",
            "tensorflow": "tensorflow_savedmodel",
            "onnx": "onnxruntime_onnx",
            "tensorrt": "tensorrt_plan"
        }
        return mapping.get(framework.lower(), framework.lower())

    def _create_model_config_map(self, name: str, config_data: Dict[str, Any]) -> None:
        """Create a ConfigMap for the model configuration."""
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(
                name=name,
                namespace=self.namespace,
                labels={
                    "app": "triton-inference-server",
                    "component": "model-config"
                }
            ),
            data={
                "config.pbtxt": self._format_triton_config(config_data)
            }
        )

        self.core_api.create_namespaced_config_map(
            namespace=self.namespace,
            body=config_map
        )

    def _update_model_config_map(self, name: str, config_data: Dict[str, Any]) -> None:
        """Update an existing ConfigMap with new model configuration."""
        config_map = self.core_api.read_namespaced_config_map(
            name=name,
            namespace=self.namespace
        )

        config_map.data = {
            "config.pbtxt": self._format_triton_config(config_data)
        }

        self.core_api.replace_namespaced_config_map(
            name=name,
            namespace=self.namespace,
            body=config_map
        )

    def _format_triton_config(self, config_data: Dict[str, Any]) -> str:
        """Format the configuration data as a Triton config file."""
        # This is a simplified version - a real implementation would generate
        # the proper Triton configuration format
        config_lines = []

        def format_value(value):
            if isinstance(value, str):
                return f'"{value}"'
            elif isinstance(value, bool):
                return "true" if value else "false"
            elif isinstance(value, (int, float)):
                return str(value)
            elif value is None:
                return ""
            elif isinstance(value, list):
                return "[" + ", ".join(format_value(v) for v in value) + "]"
            elif isinstance(value, dict):
                return "{ " + ", ".join(f"{k}: {format_value(v)}" for k, v in value.items()) + " }"
            return str(value)

        def format_dict(d, indent=0):
            lines = []
            for key, value in d.items():
                if value is None:
                    continue

                if isinstance(value, dict):
                    lines.append(" " * indent + key + " {")
                    lines.extend(format_dict(value, indent + 2))
                    lines.append(" " * indent + "}")
                elif isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    for item in value:
                        lines.append(" " * indent + key + " {")
                        lines.extend(format_dict(item, indent + 2))
                        lines.append(" " * indent + "}")
                else:
                    lines.append(" " * indent + f"{key}: {format_value(value)}")

            return lines

        return "\n".join(format_dict(config_data))

    def _check_triton_crd_exists(self) -> bool:
        """Check if the TritonServer CRD exists in the cluster."""
        try:
            api_client = client.ApiClient()
            api_instance = client.CustomObjectsApi(api_client)
            api_instance.get_api_resources_api("triton.nvidia.com", "v1alpha1")
            return True
        except ApiException:
            return False

    def _create_or_update_triton_deployment(self, model: ModelMetadata, params: DeploymentRequest) -> None:
        """Create or update a Kubernetes Deployment for Triton Inference Server."""
        # This is a simplified deployment creation
        # In a real implementation, this would include proper volume mounts,
        # resource requirements, etc.

        deployment_name = "triton-inference-server"

        try:
            # Check if deployment already exists
            existing_deployment = self.apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )

            # Update existing deployment
            existing_deployment.spec.template.spec.containers[0].env.append(
                client.V1EnvVar(
                    name=f"MODEL_{model.name.upper()}_VERSION",
                    value=params.version
                )
            )

            self.apps_api.replace_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=existing_deployment
            )

        except ApiException as e:
            if e.status == 404:
                # Deployment doesn't exist, create it
                container = client.V1Container(
                    name="triton-inference-server",
                    image="nvcr.io/nvidia/tritonserver:21.12-py3",
                    ports=[client.V1ContainerPort(container_port=8000)],
                    args=["tritonserver", "--model-repository=/models", "--strict-model-config=false"],
                    env=[
                        client.V1EnvVar(
                            name=f"MODEL_{model.name.upper()}_VERSION",
                            value=params.version
                        )
                    ],
                    resources=client.V1ResourceRequirements(
                        requests={"cpu": "1", "memory": "2Gi"} if not params.resources else params.resources,
                        limits={"cpu": "2", "memory": "4Gi"} if not params.resources else params.resources
                    ),
                    volume_mounts=[
                        client.V1VolumeMount(
                            name="model-storage",
                            mount_path="/models"
                        )
                    ]
                )

                template = client.V1PodTemplateSpec(
                    metadata=client.V1ObjectMeta(
                        labels={"app": "triton-inference-server"}
                    ),
                    spec=client.V1PodSpec(
                        containers=[container],
                        volumes=[
                            client.V1Volume(
                                name="model-storage",
                                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                                    claim_name="model-storage-pvc"
                                )
                            )
                        ]
                    )
                )

                spec = client.V1DeploymentSpec(
                    replicas=1,
                    selector=client.V1LabelSelector(
                        match_labels={"app": "triton-inference-server"}
                    ),
                    template=template
                )

                deployment = client.V1Deployment(
                    api_version="apps/v1",
                    kind="Deployment",
                    metadata=client.V1ObjectMeta(
                        name=deployment_name,
                        namespace=self.namespace
                    ),
                    spec=spec
                )

                self.apps_api.create_namespaced_deployment(
                    namespace=self.namespace,
                    body=deployment
                )

                # Create a service for the deployment
                service = client.V1Service(
                    api_version="v1",
                    kind="Service",
                    metadata=client.V1ObjectMeta(
                        name=deployment_name,
                        namespace=self.namespace
                    ),
                    spec=client.V1ServiceSpec(
                        selector={"app": "triton-inference-server"},
                        ports=[
                            client.V1ServicePort(
                                port=8000,
                                target_port=8000,
                                name="http"
                            ),
                            client.V1ServicePort(
                                port=8001,
                                target_port=8001,
                                name="grpc"
                            ),
                            client.V1ServicePort(
                                port=8002,
                                target_port=8002,
                                name="metrics"
                            )
                        ],
                        type="ClusterIP"
                    )
                )

                self.core_api.create_namespaced_service(
                    namespace=self.namespace,
                    body=service
                )
            else:
                raise

    def _create_or_update_triton_server_resource(self, model: ModelMetadata, params: DeploymentRequest) -> None:
        """Create or update a TritonServer custom resource."""
        resource_name = f"{model.name}-{params.version}"

        # Define the TritonServer custom resource
        triton_server = {
            "apiVersion": "triton.nvidia.com/v1alpha1",
            "kind": "TritonServer",
            "metadata": {
                "name": resource_name,
                "namespace": self.namespace
            },
            "spec": {
                "modelRepositoryPath": "/models",
                "models": [
                    {
                        "name": model.name,
                        "version": params.version
                    }
                ],
                "resources": {
                    "limits": {
                        "cpu": params.resources.get("cpu", "2"),
                        "memory": params.resources.get("memory", "4Gi"),
                        "nvidia.com/gpu": "1" if params.instance_type == "gpu" else "0"
                    },
                    "requests": {
                        "cpu": params.resources.get("cpu", "1"),
                        "memory": params.resources.get("memory", "2Gi"),
                        "nvidia.com/gpu": "1" if params.instance_type == "gpu" else "0"
                    }
                },
                "env": [var for var in params.environment_variables.items()],
                "replicas": params.instance_count,
                "storage": {
                    "pvc": {
                        "name": "model-storage-pvc",
                        "mountPath": "/models"
                    }
                }
            }
        }

        try:
            # Check if resource already exists
            self.custom_api.get_namespaced_custom_object(
                group="triton.nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="tritonservers",
                name=resource_name
            )

            # Update existing resource
            self.custom_api.replace_namespaced_custom_object(
                group="triton.nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="tritonservers",
                name=resource_name,
                body=triton_server
            )

        except ApiException as e:
            if e.status == 404:
                # Resource doesn't exist, create it
                self.custom_api.create_namespaced_custom_object(
                    group="triton.nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="tritonservers",
                    body=triton_server
                )
            else:
                raise
