import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface TritonOperatorOptions {
    namespace: string;
    version?: string;
    memoryRequest?: string;
    cpuRequest?: string;
    memoryLimit?: string;
    cpuLimit?: string;
    monitoringEnabled?: boolean;
    logLevel?: string;
}

export class TritonOperator extends pulumi.ComponentResource {
    public readonly crd: k8s.apiextensions.v1.CustomResourceDefinition;
    public readonly serviceAccount: k8s.core.v1.ServiceAccount;
    public readonly role: k8s.rbac.v1.Role;
    public readonly roleBinding: k8s.rbac.v1.RoleBinding;
    public readonly clusterRole: k8s.rbac.v1.ClusterRole;
    public readonly clusterRoleBinding: k8s.rbac.v1.ClusterRoleBinding;
    public readonly deployment: k8s.apps.v1.Deployment;
    public readonly service: k8s.core.v1.Service;
    public readonly servicemonitor?: k8s.apiextensions.CustomResource;

    constructor(name: string, options: TritonOperatorOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:TritonOperator", name, {}, opts);

        // Set default values if not provided
        const namespace = options.namespace;
        const version = options.version || "23.04-py3";
        const memoryRequest = options.memoryRequest || "256Mi";
        const cpuRequest = options.cpuRequest || "100m";
        const memoryLimit = options.memoryLimit || "512Mi";
        const cpuLimit = options.cpuLimit || "500m";
        const monitoringEnabled = options.monitoringEnabled !== undefined ? options.monitoringEnabled : true;
        const logLevel = options.logLevel || "INFO";

        // Create ServiceAccount for Triton Operator
        this.serviceAccount = new k8s.core.v1.ServiceAccount("triton-operator-sa", {
            metadata: {
                name: "triton-operator",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                }
            }
        }, { parent: this });

        // Create ClusterRole for Triton Operator
        this.clusterRole = new k8s.rbac.v1.ClusterRole("triton-operator-cluster-role", {
            metadata: {
                name: "triton-operator-role",
                labels: {
                    app: "triton-operator",
                }
            },
            rules: [
                {
                    apiGroups: [""],
                    resources: ["pods", "services", "services/finalizers", "endpoints", "persistentvolumeclaims", "events", "configmaps", "secrets"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
                {
                    apiGroups: ["apps"],
                    resources: ["deployments", "statefulsets"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
                {
                    apiGroups: ["monitoring.coreos.com"],
                    resources: ["servicemonitors"],
                    verbs: ["create", "get"],
                },
                {
                    apiGroups: ["triton.nvidia.com"],
                    resources: ["inferenceservers", "inferenceservers/status", "inferenceservers/finalizers"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
                {
                    apiGroups: ["apiextensions.k8s.io"],
                    resources: ["customresourcedefinitions"],
                    verbs: ["create", "get"],
                },
                {
                    apiGroups: ["coordination.k8s.io"],
                    resources: ["leases"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
            ],
        }, { parent: this });

        // Create ClusterRoleBinding for Triton Operator
        this.clusterRoleBinding = new k8s.rbac.v1.ClusterRoleBinding("triton-operator-cluster-role-binding", {
            metadata: {
                name: "triton-operator-role-binding",
                labels: {
                    app: "triton-operator",
                }
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

        // Create Role for Triton Operator (namespace specific permissions)
        this.role = new k8s.rbac.v1.Role("triton-operator-role", {
            metadata: {
                name: "triton-operator-ns-role",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                }
            },
            rules: [
                {
                    apiGroups: [""],
                    resources: ["pods", "services", "services/finalizers", "endpoints", "persistentvolumeclaims", "events", "configmaps", "secrets"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
                {
                    apiGroups: ["apps"],
                    resources: ["deployments"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
                {
                    apiGroups: ["triton.nvidia.com"],
                    resources: ["inferenceservers", "inferenceservers/status", "inferenceservers/finalizers"],
                    verbs: ["create", "delete", "get", "list", "patch", "update", "watch"],
                },
            ],
        }, { parent: this });

        // Create RoleBinding for Triton Operator
        this.roleBinding = new k8s.rbac.v1.RoleBinding("triton-operator-role-binding", {
            metadata: {
                name: "triton-operator-ns-role-binding",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                }
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

        // Create InferenceServer CRD
        this.crd = new k8s.apiextensions.v1.CustomResourceDefinition("triton-inferenceserver-crd", {
            metadata: {
                name: "inferenceservers.triton.nvidia.com",
                labels: {
                    app: "triton-operator",
                }
            },
            spec: {
                group: "triton.nvidia.com",
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
                                        image: { type: "string" },
                                        modelRepositoryPath: { type: "string" },
                                        resources: { type: "object" },
                                        replicas: { type: "integer" },
                                        config: { type: "object" },
                                    },
                                },
                                status: {
                                    type: "object",
                                    properties: {
                                        conditions: {
                                            type: "array",
                                            items: {
                                                type: "object",
                                                properties: {
                                                    type: { type: "string" },
                                                    status: { type: "string" },
                                                    reason: { type: "string" },
                                                    message: { type: "string" },
                                                    lastTransitionTime: { type: "string" },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                    subresources: {
                        status: {},
                    },
                }],
                scope: "Namespaced",
                names: {
                    plural: "inferenceservers",
                    singular: "inferenceserver",
                    kind: "InferenceServer",
                    shortNames: ["triton", "trtn"],
                },
            },
        }, { parent: this });

        // Create ConfigMap for operator configuration
        const operatorConfig = new k8s.core.v1.ConfigMap("triton-operator-config", {
            metadata: {
                name: "triton-operator-config",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                }
            },
            data: {
                "config.yaml": `
log_level: ${logLevel}
monitoring_enabled: ${monitoringEnabled}
leader_election_enabled: true
metrics_port: 8080
health_probe_port: 8081
default_image: nvcr.io/nvidia/tritonserver:${version}
`,
            },
        }, { parent: this });

        // Create Deployment for Triton Operator
        this.deployment = new k8s.apps.v1.Deployment("triton-operator", {
            metadata: {
                name: "triton-operator",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                }
            },
            spec: {
                replicas: 1,
                selector: {
                    matchLabels: {
                        app: "triton-operator",
                    },
                },
                template: {
                    metadata: {
                        labels: {
                            app: "triton-operator",
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
                            runAsGroup: 1000,
                        },
                        containers: [{
                            name: "triton-operator",
                            image: "nvcr.io/nvidia/tritonserver-operator:v1.0",
                            imagePullPolicy: "IfNotPresent",
                            args: [
                                "--config=/etc/triton-operator/config.yaml",
                            ],
                            volumeMounts: [{
                                name: "config",
                                mountPath: "/etc/triton-operator",
                            }],
                            securityContext: {
                                allowPrivilegeEscalation: false,
                                capabilities: {
                                    drop: ["ALL"],
                                },
                            },
                            resources: {
                                requests: {
                                    cpu: cpuRequest,
                                    memory: memoryRequest,
                                },
                                limits: {
                                    cpu: cpuLimit,
                                    memory: memoryLimit,
                                },
                            },
                            ports: [
                                {
                                    name: "metrics",
                                    containerPort: 8080,
                                },
                                {
                                    name: "health-probe",
                                    containerPort: 8081,
                                },
                            ],
                            livenessProbe: {
                                httpGet: {
                                    path: "/healthz",
                                    port: "health-probe",
                                },
                                initialDelaySeconds: 15,
                                periodSeconds: 20,
                            },
                            readinessProbe: {
                                httpGet: {
                                    path: "/readyz",
                                    port: "health-probe",
                                },
                                initialDelaySeconds: 5,
                                periodSeconds: 10,
                            },
                        }],
                        volumes: [{
                            name: "config",
                            configMap: {
                                name: operatorConfig.metadata.name,
                            },
                        }],
                    },
                },
            },
        }, { parent: this });

        // Create Service for Triton Operator
        this.service = new k8s.core.v1.Service("triton-operator-svc", {
            metadata: {
                name: "triton-operator",
                namespace: namespace,
                labels: {
                    app: "triton-operator",
                },
            },
            spec: {
                selector: {
                    app: "triton-operator",
                },
                ports: [
                    {
                        name: "metrics",
                        port: 8080,
                        targetPort: 8080,
                    },
                    {
                        name: "health-probe",
                        port: 8081,
                        targetPort: 8081,
                    },
                ],
            },
        }, { parent: this });

        // Create ServiceMonitor for Prometheus monitoring (if monitoring is enabled)
        if (monitoringEnabled) {
            this.servicemonitor = new k8s.apiextensions.CustomResource("triton-operator-servicemonitor", {
                apiVersion: "monitoring.coreos.com/v1",
                kind: "ServiceMonitor",
                metadata: {
                    name: "triton-operator",
                    namespace: namespace,
                    labels: {
                        app: "triton-operator",
                        "prometheus.io/scrape": "true",
                    },
                },
                spec: {
                    selector: {
                        matchLabels: {
                            app: "triton-operator",
                        },
                    },
                    endpoints: [{
                        port: "metrics",
                        interval: "15s",
                        path: "/metrics",
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
            crdName: this.crd.metadata.name,
            serviceName: this.service.metadata.name,
        });
    }
}

export default TritonOperator;
