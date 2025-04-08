import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface RayOperatorOptions {
    namespace: string;
    version?: string;
    memoryRequest?: string;
    cpuRequest?: string;
    memoryLimit?: string;
    cpuLimit?: string;
    monitoringEnabled?: boolean;
    logLevel?: string;
    createNamespace?: boolean;
}

export class RayOperator extends pulumi.ComponentResource {
    public readonly namespace: k8s.core.v1.Namespace;
    public readonly crd: k8s.apiextensions.CustomResource;
    public readonly serviceAccount: k8s.core.v1.ServiceAccount;
    public readonly role: k8s.rbac.v1.Role;
    public readonly roleBinding: k8s.rbac.v1.RoleBinding;
    public readonly clusterRole: k8s.rbac.v1.ClusterRole;
    public readonly clusterRoleBinding: k8s.rbac.v1.ClusterRoleBinding;
    public readonly deployment: k8s.apps.v1.Deployment;
    public readonly service: k8s.core.v1.Service;
    public readonly configMap: k8s.core.v1.ConfigMap;
    public readonly servicemonitor?: k8s.apiextensions.CustomResource;

    constructor(name: string, options: RayOperatorOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:RayOperator", name, {}, opts);

        // Set default values if not provided
        const namespace = options.namespace;
        const version = options.version || "2.9.0";
        const memoryRequest = options.memoryRequest || "128Mi";
        const cpuRequest = options.cpuRequest || "100m";
        const memoryLimit = options.memoryLimit || "256Mi";
        const cpuLimit = options.cpuLimit || "200m";
        const monitoringEnabled = options.monitoringEnabled !== undefined ? options.monitoringEnabled : true;
        const logLevel = options.logLevel || "info";
        const createNamespace = options.createNamespace !== undefined ? options.createNamespace : false;

        // Create namespace if requested
        if (createNamespace) {
            this.namespace = new k8s.core.v1.Namespace("ray-system", {
                metadata: {
                    name: namespace,
                    labels: {
                        "app.kubernetes.io/part-of": "ray",
                        "kubernetes.io/metadata.name": namespace,
                    },
                },
            }, { parent: this });
        }

        // Create the Ray CRDs
        this.crd = new k8s.apiextensions.CustomResource("ray-cluster-crd", {
            apiVersion: "apiextensions.k8s.io/v1",
            kind: "CustomResourceDefinition",
            metadata: {
                name: "rayclusters.ray.io",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            spec: {
                group: "ray.io",
                versions: [{
                    name: "v1",
                    served: true,
                    storage: true,
                    schema: {
                        openAPIV3Schema: {
                            type: "object",
                            properties: {
                                spec: {
                                    type: "object",
                                    properties: {
                                        rayVersion: { type: "string" },
                                        headGroupSpec: { type: "object" },
                                        workerGroupSpecs: {
                                            type: "array",
                                            items: { type: "object" },
                                        },
                                    },
                                },
                                status: {
                                    type: "object",
                                    properties: {
                                        state: { type: "string" },
                                        dashboard: { type: "string" },
                                        endpoints: { type: "object" },
                                    },
                                },
                            },
                        },
                    },
                    subresources: {
                        status: {},
                    },
                    additionalPrinterColumns: [
                        {
                            name: "Status",
                            type: "string",
                            jsonPath: ".status.state",
                        },
                        {
                            name: "Dashboard",
                            type: "string",
                            jsonPath: ".status.dashboard",
                        },
                    ],
                }],
                scope: "Namespaced",
                names: {
                    plural: "rayclusters",
                    singular: "raycluster",
                    kind: "RayCluster",
                    shortNames: ["rc"],
                },
            },
        }, { parent: this });

        // Create ServiceAccount for the Ray Operator
        this.serviceAccount = new k8s.core.v1.ServiceAccount("ray-operator-sa", {
            metadata: {
                namespace: namespace,
                name: "ray-operator",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
        }, { parent: this });

        // Create ClusterRole for the Ray Operator
        this.clusterRole = new k8s.rbac.v1.ClusterRole("ray-operator-cluster-role", {
            metadata: {
                name: "ray-operator-cluster-role",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            rules: [
                {
                    apiGroups: [""],
                    resources: ["pods", "pods/exec", "pods/log", "services", "endpoints", "persistentvolumeclaims", "events", "configmaps", "secrets"],
                    verbs: ["create", "patch", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["apps"],
                    resources: ["deployments", "statefulsets"],
                    verbs: ["create", "patch", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["ray.io"],
                    resources: ["rayclusters", "rayclusters/status", "rayclusters/finalizers"],
                    verbs: ["create", "patch", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["batch"],
                    resources: ["jobs"],
                    verbs: ["create", "patch", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["coordination.k8s.io"],
                    resources: ["leases"],
                    verbs: ["create", "get", "list", "update", "delete"],
                },
                {
                    apiGroups: ["policy"],
                    resources: ["poddisruptionbudgets"],
                    verbs: ["create", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["networking.k8s.io"],
                    resources: ["ingresses"],
                    verbs: ["create", "delete", "get", "list", "watch", "update"],
                },
                {
                    apiGroups: ["monitoring.coreos.com"],
                    resources: ["servicemonitors"],
                    verbs: ["create", "get"],
                },
            ],
        }, { parent: this });

        // Create ClusterRoleBinding for the Ray Operator
        this.clusterRoleBinding = new k8s.rbac.v1.ClusterRoleBinding("ray-operator-cluster-role-binding", {
            metadata: {
                name: "ray-operator-cluster-role-binding",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            roleRef: {
                apiGroup: "rbac.authorization.k8s.io",
                kind: "ClusterRole",
                name: this.clusterRole.metadata.name,
            },
            subjects: [{
                kind: "ServiceAccount",
                name: this.serviceAccount.metadata.name,
                namespace: namespace,
            }],
        }, { parent: this });

        // Create namespaced Role for the Ray Operator
        this.role = new k8s.rbac.v1.Role("ray-operator-role", {
            metadata: {
                namespace: namespace,
                name: "ray-operator-role",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            rules: [
                {
                    apiGroups: [""],
                    resources: ["configmaps"],
                    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"],
                },
                {
                    apiGroups: ["coordination.k8s.io"],
                    resources: ["leases"],
                    verbs: ["create", "get", "list", "update", "delete"],
                },
            ],
        }, { parent: this });

        // Create namespaced RoleBinding for the Ray Operator
        this.roleBinding = new k8s.rbac.v1.RoleBinding("ray-operator-role-binding", {
            metadata: {
                namespace: namespace,
                name: "ray-operator-role-binding",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            roleRef: {
                apiGroup: "rbac.authorization.k8s.io",
                kind: "Role",
                name: this.role.metadata.name,
            },
            subjects: [{
                kind: "ServiceAccount",
                name: this.serviceAccount.metadata.name,
                namespace: namespace,
            }],
        }, { parent: this });

        // Create ConfigMap for operator configuration
        this.configMap = new k8s.core.v1.ConfigMap("ray-operator-config", {
            metadata: {
                namespace: namespace,
                name: "ray-operator-config",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            data: {
                "config.yaml": `
log_level: ${logLevel}
monitoring_enabled: ${monitoringEnabled}
metrics_port: 8080
health_probe_port: 8081
leader_election_enabled: true
leader_election_namespace: ${namespace}
default_ray_version: ${version}
`,
            },
        }, { parent: this });

        // Create Deployment for the Ray Operator
        this.deployment = new k8s.apps.v1.Deployment("ray-operator", {
            metadata: {
                namespace: namespace,
                name: "ray-operator",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            spec: {
                replicas: 1,
                selector: {
                    matchLabels: {
                        "app.kubernetes.io/name": "ray-operator",
                    },
                },
                template: {
                    metadata: {
                        labels: {
                            "app.kubernetes.io/name": "ray-operator",
                        },
                        annotations: monitoringEnabled ? {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8080",
                        } : {},
                    },
                    spec: {
                        serviceAccountName: this.serviceAccount.metadata.name,
                        securityContext: {
                            runAsNonRoot: true,
                            runAsUser: 1000,
                            fsGroup: 2000,
                        },
                        containers: [{
                            name: "ray-operator",
                            image: `rayproject/ray:${version}`,
                            command: ["ray-operator"],
                            args: [
                                "--config=/etc/ray-operator/config.yaml",
                            ],
                            volumeMounts: [{
                                name: "config-volume",
                                mountPath: "/etc/ray-operator",
                            }],
                            securityContext: {
                                allowPrivilegeEscalation: false,
                                capabilities: {
                                    drop: ["ALL"],
                                },
                            },
                            resources: {
                                limits: {
                                    cpu: cpuLimit,
                                    memory: memoryLimit,
                                },
                                requests: {
                                    cpu: cpuRequest,
                                    memory: memoryRequest,
                                },
                            },
                            ports: [{
                                containerPort: 8080,
                                name: "metrics",
                            }, {
                                containerPort: 8081,
                                name: "health",
                            }],
                            livenessProbe: {
                                httpGet: {
                                    path: "/healthz",
                                    port: "health",
                                },
                                initialDelaySeconds: 15,
                                periodSeconds: 20,
                            },
                            readinessProbe: {
                                httpGet: {
                                    path: "/readyz",
                                    port: "health",
                                },
                                initialDelaySeconds: 5,
                                periodSeconds: 10,
                            },
                            env: [{
                                name: "WATCH_NAMESPACE",
                                valueFrom: {
                                    fieldRef: {
                                        fieldPath: "metadata.namespace",
                                    },
                                },
                            }, {
                                name: "POD_NAME",
                                valueFrom: {
                                    fieldRef: {
                                        fieldPath: "metadata.name",
                                    },
                                },
                            }, {
                                name: "OPERATOR_NAME",
                                value: "ray-operator",
                            }],
                        }],
                        volumes: [{
                            name: "config-volume",
                            configMap: {
                                name: this.configMap.metadata.name,
                            },
                        }],
                    },
                },
            },
        }, { parent: this });

        // Create Service for the Ray Operator
        this.service = new k8s.core.v1.Service("ray-operator-service", {
            metadata: {
                namespace: namespace,
                name: "ray-operator-metrics",
                labels: {
                    "app.kubernetes.io/name": "ray-operator",
                    "app.kubernetes.io/part-of": "ray",
                },
            },
            spec: {
                selector: {
                    "app.kubernetes.io/name": "ray-operator",
                },
                ports: [{
                    port: 8080,
                    targetPort: 8080,
                    name: "metrics",
                }, {
                    port: 8081,
                    targetPort: 8081,
                    name: "health",
                }],
            },
        }, { parent: this });

        // Create ServiceMonitor for Prometheus monitoring (if enabled)
        if (monitoringEnabled) {
            this.servicemonitor = new k8s.apiextensions.CustomResource("ray-operator-servicemonitor", {
                apiVersion: "monitoring.coreos.com/v1",
                kind: "ServiceMonitor",
                metadata: {
                    namespace: namespace,
                    name: "ray-operator",
                    labels: {
                        "app.kubernetes.io/name": "ray-operator",
                        "app.kubernetes.io/part-of": "ray",
                    },
                },
                spec: {
                    selector: {
                        matchLabels: {
                            "app.kubernetes.io/name": "ray-operator",
                        },
                    },
                    endpoints: [{
                        port: "metrics",
                        path: "/metrics",
                        interval: "15s",
                    }],
                    namespaceSelector: {
                        matchNames: [namespace],
                    },
                },
            }, { parent: this });
        }

        // Register outputs
        this.registerOutputs({
            operatorName: this.deployment.metadata.name,
            serviceName: this.service.metadata.name,
            namespace: namespace,
        });
    }
}

export default RayOperator;
