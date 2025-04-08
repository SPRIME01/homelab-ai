import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface TritonServerOptions {
    namespace: string;
    version?: string;
    memoryLimit?: string;
    cpuLimit?: string;
    numGpus?: number;
    modelRepositoryPath?: string;
    minioEndpoint?: string;
    minioAccessKey?: pulumi.Input<string>;
    minioSecretKey?: pulumi.Input<string>;
    enableMetrics?: boolean;
    logLevel?: string;
    cacheSizeMB?: number;
    dynamicBatchingDelay?: number;
    concurrentModelExecutionCount?: number;
    storageClassName?: string;
    storageSize?: string;
}

export class TritonServer extends pulumi.ComponentResource {
    public readonly inferenceServer: k8s.apiextensions.CustomResource;
    public readonly service: k8s.core.v1.Service;
    public readonly configMap: k8s.core.v1.ConfigMap;
    public readonly secret?: k8s.core.v1.Secret;
    public readonly ingress?: k8s.networking.v1.Ingress;

    constructor(name: string, options: TritonServerOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:TritonServer", name, {}, opts);

        // Set default values if not provided
        const namespace = options.namespace;
        const version = options.version || "23.04-py3";
        const memoryLimit = options.memoryLimit || "40Gi"; // Jetson AGX Orin has 64GB, allocate most to Triton
        const cpuLimit = options.cpuLimit || "10"; // Jetson AGX Orin has 12-core ARM CPU
        const numGpus = options.numGpus || 1;
        const modelRepositoryPath = options.modelRepositoryPath || "/models";
        const enableMetrics = options.enableMetrics !== undefined ? options.enableMetrics : true;
        const logLevel = options.logLevel || "INFO";
        const cacheSizeMB = options.cacheSizeMB || 1024; // 1GB model cache
        const dynamicBatchingDelay = options.dynamicBatchingDelay || 100; // 100 microseconds
        const concurrentModelExecutionCount = options.concurrentModelExecutionCount || 4;
        const storageClassName = options.storageClassName || "local-path";
        const storageSize = options.storageSize || "100Gi";

        // Create ConfigMap for Triton configurations
        this.configMap = new k8s.core.v1.ConfigMap("triton-config", {
            metadata: {
                namespace: namespace,
                name: `${name}-config`,
                labels: {
                    app: name,
                    "app.kubernetes.io/name": "triton-inference-server",
                    "app.kubernetes.io/part-of": "homelab-ai",
                }
            },
            data: {
                "triton.json": JSON.stringify({
                    "log_verbose": true,
                    "log_info": true,
                    "log_warning": true,
                    "log_error": true,
                    "exit_on_error": true,
                    "log_format": "default",
                    "allow_metrics": enableMetrics,
                    "allow_gpu_metrics": enableMetrics,
                    "metrics_interval_ms": 2000,
                    "rate_limit": true,
                    "rate_limit_resource": "GPU",
                    "buffer_manager_thread_count": 0,
                    "model_control_mode": "explicit",
                    "strict_model_config": false,
                    "pinned_memory_pool_byte_size": 268435456, // 256MB
                    "response_cache_byte_size": cacheSizeMB * 1024 * 1024,
                    "min_supported_compute_capability": 7.2, // Jetson AGX Orin capability
                    "backend_directory": "/opt/tritonserver/backends",
                    "repository_directory": modelRepositoryPath,
                    "model_repository_directory": modelRepositoryPath
                }),
                "backend-config.json": JSON.stringify({
                    "tensorflow": {
                        "version": "2",
                        "backend_accelerator": ["tensorrt"],
                        "optimization": {
                            "execution_mode": "graph",
                            "inference_mode": "float16"
                        }
                    },
                    "tensorrt": {
                        "backend_config": {
                            "coalesce_requests": true
                        }
                    },
                    "onnxruntime": {
                        "backend_config": {
                            "default_target_device": "GPU",
                            "gpu_memory_limit": 8589934592, // 8GB per model
                            "optimization": {
                                "enable_transformer_optimization": true,
                                "graph_optimization_level": "all"
                            }
                        }
                    },
                    "python": {
                        "backend_config": {
                            "shm-region-prefix-name": "triton_python_backend_shm",
                            "shm-default-byte-size": 134217728 // 128MB
                        }
                    }
                })
            }
        }, { parent: this });

        // Create Secret for MinIO access if credentials are provided
        if (options.minioEndpoint && options.minioAccessKey && options.minioSecretKey) {
            this.secret = new k8s.core.v1.Secret("triton-minio-secret", {
                metadata: {
                    namespace: namespace,
                    name: `${name}-minio-credentials`,
                    labels: {
                        app: name,
                    }
                },
                type: "Opaque",
                stringData: {
                    AWS_ACCESS_KEY_ID: options.minioAccessKey,
                    AWS_SECRET_ACCESS_KEY: options.minioSecretKey,
                    AWS_ENDPOINT: options.minioEndpoint,
                    AWS_DEFAULT_REGION: "us-east-1",
                    AWS_REGION: "us-east-1",
                    S3_USE_HTTPS: "0",
                    S3_VERIFY_SSL: "0"
                }
            }, { parent: this });
        }

        // Create PVC for model storage
        const modelStoragePvc = new k8s.core.v1.PersistentVolumeClaim("triton-model-storage", {
            metadata: {
                namespace: namespace,
                name: `${name}-model-storage`,
                labels: {
                    app: name,
                }
            },
            spec: {
                accessModes: ["ReadWriteOnce"],
                resources: {
                    requests: {
                        storage: storageSize
                    }
                },
                storageClassName: storageClassName
            }
        }, { parent: this });

        // Create Triton Inference Server using the operator's CRD
        this.inferenceServer = new k8s.apiextensions.CustomResource("triton-inference-server", {
            apiVersion: "triton.nvidia.com/v1",
            kind: "InferenceServer",
            metadata: {
                namespace: namespace,
                name: name,
                labels: {
                    app: name,
                    "app.kubernetes.io/name": "triton-inference-server",
                    "app.kubernetes.io/part-of": "homelab-ai",
                },
                annotations: {
                    "prometheus.io/scrape": enableMetrics ? "true" : "false",
                    "prometheus.io/port": "8002",
                    "prometheus.io/path": "/metrics"
                }
            },
            spec: {
                // Use NVIDIA GPU Cloud Triton image
                image: `nvcr.io/nvidia/tritonserver:${version}`,
                modelRepositoryPath: modelRepositoryPath,
                replicas: 1,
                resources: {
                    limits: {
                        "nvidia.com/gpu": numGpus.toString(),
                        memory: memoryLimit,
                        cpu: cpuLimit
                    },
                    requests: {
                        "nvidia.com/gpu": numGpus.toString(),
                        memory: (parseInt(memoryLimit) / 2) + "Gi", // Request half of limit
                        cpu: (parseInt(cpuLimit) / 2).toString()  // Request half of limit
                    }
                },
                config: {
                    // Server configuration
                    serverConfig: {
                        strictModelConfig: false,
                        modelControlMode: "explicit",
                        minSupportedComputeCapability: 7.2, // Jetson AGX Orin
                        pinnedMemoryPoolByteSize: 268435456, // 256MB
                        responseCacheByteSize: cacheSizeMB * 1024 * 1024,
                        logLevel: logLevel,
                        metrics: {
                            enable: enableMetrics,
                            gpuMetrics: enableMetrics,
                            metricsInterval: 2000,
                        }
                    },
                    // Configure dynamic batching
                    dynamicBatching: {
                        preferred_batch_size: [1, 2, 4, 8],
                        max_queue_delay_microseconds: dynamicBatchingDelay,
                        preserve_ordering: true,
                        default_queue_policy: {
                            timeout_action: "DELAY",
                            default_timeout_microseconds: 10000000, // 10 seconds
                            allow_timeout_override: true,
                            max_queue_size: 100
                        }
                    },
                    // Configure instance groups for models
                    instanceGroups: [
                        {
                            name: "gpu-group",
                            kind: "KIND_GPU",
                            count: concurrentModelExecutionCount
                        },
                        {
                            name: "cpu-group",
                            kind: "KIND_CPU",
                            count: 2
                        }
                    ],
                    // Configure environment variables
                    env: [
                        { name: "TF_ENABLE_ONEDNN_OPTS", value: "1" },
                        { name: "TF_USE_CUDNN", value: "1" },
                        { name: "OMP_NUM_THREADS", value: "4" },
                        { name: "CUDA_DEVICE_ORDER", value: "PCI_BUS_ID" },
                        { name: "CUDA_VISIBLE_DEVICES", value: "0" }
                    ],
                    // Mount the config map
                    volumeMounts: [
                        {
                            name: "triton-config",
                            mountPath: "/opt/tritonserver/config"
                        },
                        {
                            name: "model-storage",
                            mountPath: modelRepositoryPath
                        }
                    ],
                    volumes: [
                        {
                            name: "triton-config",
                            configMap: {
                                name: this.configMap.metadata.name
                            }
                        },
                        {
                            name: "model-storage",
                            persistentVolumeClaim: {
                                claimName: modelStoragePvc.metadata.name
                            }
                        }
                    ]
                }
            }
        }, { parent: this });

        // Create Service for Triton Inference Server
        this.service = new k8s.core.v1.Service("triton-service", {
            metadata: {
                namespace: namespace,
                name: `${name}-service`,
                labels: {
                    app: name,
                    "app.kubernetes.io/name": "triton-inference-server",
                    "app.kubernetes.io/part-of": "homelab-ai",
                },
                annotations: enableMetrics ? {
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "8002",
                    "prometheus.io/path": "/metrics"
                } : {}
            },
            spec: {
                type: "ClusterIP",
                ports: [
                    { name: "http", port: 8000, targetPort: 8000 },
                    { name: "grpc", port: 8001, targetPort: 8001 },
                    { name: "metrics", port: 8002, targetPort: 8002 }
                ],
                selector: {
                    app: name,
                }
            }
        }, { parent: this });

        // Create Ingress for Triton Inference Server HTTP endpoints
        this.ingress = new k8s.networking.v1.Ingress("triton-ingress", {
            metadata: {
                namespace: namespace,
                name: `${name}-ingress`,
                labels: {
                    app: name,
                },
                annotations: {
                    "nginx.ingress.kubernetes.io/proxy-body-size": "0",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "3600",
                    "nginx.ingress.kubernetes.io/proxy-send-timeout": "3600",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            spec: {
                ingressClassName: "nginx",
                rules: [
                    {
                        host: `triton.homelab.local`,
                        http: {
                            paths: [
                                {
                                    path: "/",
                                    pathType: "Prefix",
                                    backend: {
                                        service: {
                                            name: this.service.metadata.name,
                                            port: { name: "http" }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ],
                tls: [
                    {
                        hosts: ["triton.homelab.local"],
                        secretName: "triton-tls-secret"
                    }
                ]
            }
        }, { parent: this });

        // Register outputs
        this.registerOutputs({
            inferenceServerName: name,
            serviceName: this.service.metadata.name,
            ingressHost: "triton.homelab.local",
            metricsEnabled: enableMetrics
        });
    }
}

export default TritonServer;
