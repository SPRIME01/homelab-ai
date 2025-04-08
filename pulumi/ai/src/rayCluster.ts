import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface RayClusterOptions {
    namespace: string;
    rayVersion?: string;
    headNodeCpuRequest?: string;
    headNodeMemoryRequest?: string;
    headNodeCpuLimit?: string;
    headNodeMemoryLimit?: string;
    headNodeGpuLimit?: number;
    workerNodeCpuRequest?: string;
    workerNodeMemoryRequest?: string;
    workerNodeCpuLimit?: string;
    workerNodeMemoryLimit?: string;
    workerNodeGpuLimit?: number;
    minWorkers?: number;
    maxWorkers?: number;
    storageClassName?: string;
    storageSize?: string;
    enableDashboard?: boolean;
    monitoringEnabled?: boolean;
    dashboardHost?: string;
    ingressClassName?: string;
    tlsSecretName?: string;
}

export class RayCluster extends pulumi.ComponentResource {
    public readonly cluster: k8s.apiextensions.CustomResource;
    public readonly headService: k8s.core.v1.Service;
    public readonly dashboardIngress?: k8s.networking.v1.Ingress;
    public readonly configMap: k8s.core.v1.ConfigMap;
    public readonly headPvc: k8s.core.v1.PersistentVolumeClaim;
    public readonly autoscalerService?: k8s.core.v1.Service;

    constructor(name: string, options: RayClusterOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:RayCluster", name, {}, opts);

        // Set default values if not provided
        const namespace = options.namespace;
        const rayVersion = options.rayVersion || "2.9.0";
        const headNodeCpuRequest = options.headNodeCpuRequest || "2";
        const headNodeMemoryRequest = options.headNodeMemoryRequest || "4Gi";
        const headNodeCpuLimit = options.headNodeCpuLimit || "4";
        const headNodeMemoryLimit = options.headNodeMemoryLimit || "8Gi";
        const headNodeGpuLimit = options.headNodeGpuLimit !== undefined ? options.headNodeGpuLimit : 1;
        const workerNodeCpuRequest = options.workerNodeCpuRequest || "1";
        const workerNodeMemoryRequest = options.workerNodeMemoryRequest || "2Gi";
        const workerNodeCpuLimit = options.workerNodeCpuLimit || "2";
        const workerNodeMemoryLimit = options.workerNodeMemoryLimit || "4Gi";
        const workerNodeGpuLimit = options.workerNodeGpuLimit !== undefined ? options.workerNodeGpuLimit : 0;
        const minWorkers = options.minWorkers || 1;
        const maxWorkers = options.maxWorkers || 3;
        const storageClassName = options.storageClassName || "local-path";
        const storageSize = options.storageSize || "10Gi";
        const enableDashboard = options.enableDashboard !== undefined ? options.enableDashboard : true;
        const monitoringEnabled = options.monitoringEnabled !== undefined ? options.monitoringEnabled : true;
        const dashboardHost = options.dashboardHost || "ray-dashboard.homelab.local";
        const ingressClassName = options.ingressClassName || "nginx";
        const tlsSecretName = options.tlsSecretName || "ray-tls-secret";

        // Create ConfigMap for Ray Cluster configuration
        this.configMap = new k8s.core.v1.ConfigMap("ray-cluster-config", {
            metadata: {
                namespace: namespace,
                name: `${name}-config`,
                labels: {
                    "app.kubernetes.io/name": "ray-cluster",
                    "app.kubernetes.io/part-of": "ray",
                    "app.kubernetes.io/instance": name,
                },
            },
            data: {
                "ray-start-params.yaml": `
runtime-env-setup-commands: []
runtime-env-config:
  setup_timeout_seconds: 600
resources:
  accelerator_type: "NVIDIA"
  accelerator_registry: {}
  allocation_strategy: "PRIORITY"
  priority_scheduling: true
  request_timeout_seconds: 600
  gpu_scheduling_strategy:
    strategy: "SPREAD"
    resources_per_task: 1
    strict: true
object-store-memory: "1073741824"  # 1GB
system-config:
  raylet-fair-sharing: true
  memory-monitor: true
  auto-gc: true
  object-spilling: true
  heap-limit: "8589934592"  # 8GB
  debug: false
`,
                "custom-metrics.yaml": `
metrics:
  prometheus:
    export_metrics: ${monitoringEnabled}
    namespace: ray_cluster_${name}
  prometheus_file_based_metrics:
    enabled: false
`,
            },
        }, { parent: this });

        // Create PVC for Ray head node
        this.headPvc = new k8s.core.v1.PersistentVolumeClaim("ray-head-pvc", {
            metadata: {
                namespace: namespace,
                name: `${name}-head-pvc`,
                labels: {
                    "app.kubernetes.io/name": "ray-cluster",
                    "app.kubernetes.io/part-of": "ray",
                    "app.kubernetes.io/instance": name,
                },
            },
            spec: {
                accessModes: ["ReadWriteOnce"],
                resources: {
                    requests: {
                        storage: storageSize,
                    },
                },
                storageClassName: storageClassName,
            },
        }, { parent: this });

        // Create Ray Cluster CR
        this.cluster = new k8s.apiextensions.CustomResource("ray-cluster", {
            apiVersion: "ray.io/v1",
            kind: "RayCluster",
            metadata: {
                namespace: namespace,
                name: name,
                labels: {
                    "app.kubernetes.io/name": "ray-cluster",
                    "app.kubernetes.io/part-of": "ray",
                    "app.kubernetes.io/instance": name,
                },
                annotations: monitoringEnabled ? {
                    "prometheus.io/scrape": "true",
                    "prometheus.io/port": "8080",
                } : {},
            },
            spec: {
                rayVersion: rayVersion,
                headGroupSpec: {
                    rayStartParams: {
                        "node-ip-address": "$MY_POD_IP",
                        "num-cpus": "0",  // All CPUs managed by Kubernetes
                        "block": "true",
                        "dashboard-host": "0.0.0.0",
                    },
                    template: {
                        metadata: {
                            labels: {
                                "ray.io/node-type": "head",
                                "app.kubernetes.io/name": "ray-cluster",
                                "app.kubernetes.io/component": "head",
                                "app.kubernetes.io/instance": name,
                            },
                            annotations: monitoringEnabled ? {
                                "prometheus.io/scrape": "true",
                                "prometheus.io/port": "8080",
                            } : {},
                        },
                        spec: {
                            nodeSelector: {
                                "nvidia.com/gpu-present": "true",  // Ensure head node runs on GPU-enabled node
                            },
                            serviceAccountName: "ray-operator",
                            volumes: [
                                {
                                    name: "ray-config",
                                    configMap: {
                                        name: this.configMap.metadata.name,
                                    },
                                },
                                {
                                    name: "ray-data",
                                    persistentVolumeClaim: {
                                        claimName: this.headPvc.metadata.name,
                                    },
                                },
                            ],
                            containers: [
                                {
                                    name: "ray-head",
                                    image: `rayproject/ray:${rayVersion}`,
                                    imagePullPolicy: "IfNotPresent",
                                    ports: [
                                        { name: "redis", containerPort: 6379 },
                                        { name: "ray-client", containerPort: 10001 },
                                        { name: "head", containerPort: 8265 },
                                        { name: "dashboard", containerPort: 8265 },
                                        { name: "metrics", containerPort: 8080 },
                                    ],
                                    env: [
                                        {
                                            name: "MY_POD_IP",
                                            valueFrom: {
                                                fieldRef: {
                                                    fieldPath: "status.podIP",
                                                },
                                            },
                                        },
                                        {
                                            name: "RAY_DISABLE_DOCKER_CPU_WARNING",
                                            value: "1",
                                        },
                                        {
                                            name: "RAY_BACKEND_LOG_LEVEL",
                                            value: "info",
                                        },
                                    ],
                                    resources: {
                                        limits: {
                                            cpu: headNodeCpuLimit,
                                            memory: headNodeMemoryLimit,
                                            ...(headNodeGpuLimit > 0 ? { "nvidia.com/gpu": headNodeGpuLimit.toString() } : {}),
                                        },
                                        requests: {
                                            cpu: headNodeCpuRequest,
                                            memory: headNodeMemoryRequest,
                                            ...(headNodeGpuLimit > 0 ? { "nvidia.com/gpu": headNodeGpuLimit.toString() } : {}),
                                        },
                                    },
                                    volumeMounts: [
                                        {
                                            name: "ray-config",
                                            mountPath: "/etc/ray/ray-start-params.yaml",
                                            subPath: "ray-start-params.yaml",
                                        },
                                        {
                                            name: "ray-config",
                                            mountPath: "/etc/ray/custom-metrics.yaml",
                                            subPath: "custom-metrics.yaml",
                                        },
                                        {
                                            name: "ray-data",
                                            mountPath: "/data",
                                        },
                                    ],
                                    lifecycle: {
                                        preStop: {
                                            exec: {
                                                command: ["/bin/sh", "-c", "ray stop"],
                                            },
                                        },
                                    },
                                    securityContext: {
                                        runAsUser: 1000,
                                        allowPrivilegeEscalation: false,
                                    },
                                    livenessProbe: {
                                        exec: {
                                            command: ["/bin/sh", "-c", "ray status"],
                                        },
                                        initialDelaySeconds: 30,
                                        periodSeconds: 20,
                                    },
                                    readinessProbe: {
                                        exec: {
                                            command: ["/bin/sh", "-c", "ray status"],
                                        },
                                        initialDelaySeconds: 10,
                                        periodSeconds: 10,
                                    },
                                },
                            ],
                        },
                    },
                    serviceType: "ClusterIP",
                    replicas: 1,
                },
                workerGroupSpecs: [
                    {
                        groupName: "small-workers",
                        replicas: minWorkers,
                        minReplicas: minWorkers,
                        maxReplicas: maxWorkers,
                        rayStartParams: {
                            "node-ip-address": "$MY_POD_IP",
                            "num-cpus": "0",  // All CPUs managed by Kubernetes
                            "block": "true",
                        },
                        template: {
                            metadata: {
                                labels: {
                                    "ray.io/node-type": "worker",
                                    "ray.io/group": "small-workers",
                                    "app.kubernetes.io/name": "ray-cluster",
                                    "app.kubernetes.io/component": "worker",
                                    "app.kubernetes.io/instance": name,
                                },
                                annotations: monitoringEnabled ? {
                                    "prometheus.io/scrape": "true",
                                    "prometheus.io/port": "8080",
                                } : {},
                            },
                            spec: {
                                serviceAccountName: "ray-operator",
                                volumes: [
                                    {
                                        name: "ray-config",
                                        configMap: {
                                            name: this.configMap.metadata.name,
                                        },
                                    },
                                ],
                                containers: [
                                    {
                                        name: "ray-worker",
                                        image: `rayproject/ray:${rayVersion}`,
                                        imagePullPolicy: "IfNotPresent",
                                        env: [
                                            {
                                                name: "MY_POD_IP",
                                                valueFrom: {
                                                    fieldRef: {
                                                        fieldPath: "status.podIP",
                                                    },
                                                },
                                            },
                                            {
                                                name: "RAY_DISABLE_DOCKER_CPU_WARNING",
                                                value: "1",
                                            },
                                            {
                                                name: "RAY_BACKEND_LOG_LEVEL",
                                                value: "info",
                                            },
                                        ],
                                        resources: {
                                            limits: {
                                                cpu: workerNodeCpuLimit,
                                                memory: workerNodeMemoryLimit,
                                                ...(workerNodeGpuLimit > 0 ? { "nvidia.com/gpu": workerNodeGpuLimit.toString() } : {}),
                                            },
                                            requests: {
                                                cpu: workerNodeCpuRequest,
                                                memory: workerNodeMemoryRequest,
                                                ...(workerNodeGpuLimit > 0 ? { "nvidia.com/gpu": workerNodeGpuLimit.toString() } : {}),
                                            },
                                        },
                                        volumeMounts: [
                                            {
                                                name: "ray-config",
                                                mountPath: "/etc/ray/ray-start-params.yaml",
                                                subPath: "ray-start-params.yaml",
                                            },
                                            {
                                                name: "ray-config",
                                                mountPath: "/etc/ray/custom-metrics.yaml",
                                                subPath: "custom-metrics.yaml",
                                            },
                                        ],
                                        lifecycle: {
                                            preStop: {
                                                exec: {
                                                    command: ["/bin/sh", "-c", "ray stop"],
                                                },
                                            },
                                        },
                                        securityContext: {
                                            runAsUser: 1000,
                                            allowPrivilegeEscalation: false,
                                        },
                                        livenessProbe: {
                                            exec: {
                                                command: ["/bin/sh", "-c", "ray status"],
                                            },
                                            initialDelaySeconds: 30,
                                            periodSeconds: 20,
                                        },
                                        readinessProbe: {
                                            exec: {
                                                command: ["/bin/sh", "-c", "ray status"],
                                            },
                                            initialDelaySeconds: 10,
                                            periodSeconds: 10,
                                        },
                                    },
                                ],
                            },
                        },
                    },
                ],
                enableInTreeAutoscaling: true,
            },
        }, { parent: this });

        // Create Ray head service
        this.headService = new k8s.core.v1.Service("ray-head-service", {
            metadata: {
                namespace: namespace,
                name: `${name}-head`,
                labels: {
                    "app.kubernetes.io/name": "ray-cluster",
                    "app.kubernetes.io/part-of": "ray",
                    "app.kubernetes.io/instance": name,
                    "app.kubernetes.io/component": "head",
                },
            },
            spec: {
                selector: {
                    "ray.io/node-type": "head",
                    "app.kubernetes.io/instance": name,
                },
                ports: [
                    { name: "client", port: 10001, targetPort: 10001 },
                    { name: "dashboard", port: 8265, targetPort: 8265 },
                    { name: "redis", port: 6379, targetPort: 6379 },
                    { name: "metrics", port: 8080, targetPort: 8080 },
                ],
                type: "ClusterIP",
            },
        }, { parent: this });

        // Create Ray dashboard ingress if enabled
        if (enableDashboard) {
            this.dashboardIngress = new k8s.networking.v1.Ingress("ray-dashboard-ingress", {
                metadata: {
                    namespace: namespace,
                    name: `${name}-dashboard`,
                    labels: {
                        "app.kubernetes.io/name": "ray-cluster",
                        "app.kubernetes.io/part-of": "ray",
                        "app.kubernetes.io/instance": name,
                        "app.kubernetes.io/component": "dashboard",
                    },
                    annotations: {
                        "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                        "nginx.ingress.kubernetes.io/proxy-body-size": "50m",
                        "nginx.ingress.kubernetes.io/proxy-connect-timeout": "300",
                        "nginx.ingress.kubernetes.io/proxy-read-timeout": "300",
                        "nginx.ingress.kubernetes.io/proxy-send-timeout": "300",
                        "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    },
                },
                spec: {
                    ingressClassName: ingressClassName,
                    tls: [
                        {
                            hosts: [dashboardHost],
                            secretName: tlsSecretName,
                        },
                    ],
                    rules: [
                        {
                            host: dashboardHost,
                            http: {
                                paths: [
                                    {
                                        path: "/",
                                        pathType: "Prefix",
                                        backend: {
                                            service: {
                                                name: this.headService.metadata.name,
                                                port: {
                                                    name: "dashboard",
                                                },
                                            },
                                        },
                                    },
                                ],
                            },
                        },
                    ],
                },
            }, { parent: this });
        }

        // Create Prometheus ServiceMonitor if monitoring is enabled
        if (monitoringEnabled) {
            new k8s.apiextensions.CustomResource("ray-servicemonitor", {
                apiVersion: "monitoring.coreos.com/v1",
                kind: "ServiceMonitor",
                metadata: {
                    namespace: namespace,
                    name: `${name}-monitor`,
                    labels: {
                        "app.kubernetes.io/name": "ray-cluster",
                        "app.kubernetes.io/part-of": "ray",
                        "app.kubernetes.io/instance": name,
                        "prometheus.io/scrape": "true",
                    },
                },
                spec: {
                    selector: {
                        matchLabels: {
                            "app.kubernetes.io/name": "ray-cluster",
                            "app.kubernetes.io/instance": name,
                        },
                    },
                    endpoints: [
                        {
                            port: "metrics",
                            interval: "15s",
                            path: "/metrics",
                        },
                    ],
                    namespaceSelector: {
                        matchNames: [namespace],
                    },
                },
            }, { parent: this });
        }

        // Create Ray autoscaler service
        this.autoscalerService = new k8s.core.v1.Service("ray-autoscaler-service", {
            metadata: {
                namespace: namespace,
                name: `${name}-autoscaler`,
                labels: {
                    "app.kubernetes.io/name": "ray-cluster",
                    "app.kubernetes.io/part-of": "ray",
                    "app.kubernetes.io/instance": name,
                    "app.kubernetes.io/component": "autoscaler",
                },
            },
            spec: {
                selector: {
                    "ray.io/node-type": "head",
                    "app.kubernetes.io/instance": name,
                },
                ports: [
                    { name: "autoscaler", port: 7600, targetPort: 7600 },
                ],
                type: "ClusterIP",
            },
        }, { parent: this });

        // Register outputs
        this.registerOutputs({
            clusterName: name,
            headServiceName: this.headService.metadata.name,
            dashboardUrl: enableDashboard ? `https://${dashboardHost}` : undefined,
            rayClientEndpoint: pulumi.interpolate`${this.headService.metadata.name}.${namespace}.svc.cluster.local:10001`,
        });
    }
}

export default RayCluster;
