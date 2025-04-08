import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export class AINamespace extends pulumi.ComponentResource {
    public readonly namespace: k8s.core.v1.Namespace;
    public readonly resourceQuota: k8s.core.v1.ResourceQuota;
    public readonly limitRange: k8s.core.v1.LimitRange;
    public readonly networkPolicy: k8s.networking.v1.NetworkPolicy;

    constructor(name: string, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:Namespace", name, {}, opts);

        // Create the AI namespace
        this.namespace = new k8s.core.v1.Namespace("ai-namespace", {
            metadata: {
                name: "ai",
                labels: {
                    "app.kubernetes.io/name": "ai-inference",
                    "app.kubernetes.io/part-of": "homelab",
                    "homelab.io/gpu-enabled": "true",
                    "kubernetes.io/metadata.name": "ai"
                },
                annotations: {
                    "homelab.io/description": "Namespace for AI inference workloads",
                    "homelab.io/owner": "homelab-admin",
                    "prometheus.io/scrape": "true"
                }
            }
        }, { parent: this });

        // Create resource quota for the AI namespace
        // Allocating majority of Jetson AGX Orin resources (64GB RAM)
        this.resourceQuota = new k8s.core.v1.ResourceQuota("ai-resource-quota", {
            metadata: {
                namespace: this.namespace.metadata.name,
                name: "ai-quota"
            },
            spec: {
                hard: {
                    "requests.cpu": "12",             // 12 cores out of 16 cores available
                    "limits.cpu": "14",               // Maximum CPU limit
                    "requests.memory": "48Gi",        // 48GB out of 64GB RAM
                    "limits.memory": "56Gi",          // Maximum memory limit
                    "requests.nvidia.com/gpu": "1",   // 1 GPU
                    "limits.nvidia.com/gpu": "1",     // 1 GPU
                    "persistentvolumeclaims": "10",   // Max number of PVCs
                    "services": "20",                 // Max number of services
                    "pods": "50",                     // Max number of pods
                    "configmaps": "40",               // Max number of configmaps
                    "secrets": "40"                   // Max number of secrets
                }
            }
        }, { parent: this.namespace });

        // Create limit range for the AI namespace
        this.limitRange = new k8s.core.v1.LimitRange("ai-limit-range", {
            metadata: {
                namespace: this.namespace.metadata.name,
                name: "ai-limits"
            },
            spec: {
                limits: [{
                    type: "Container",
                    default: {
                        cpu: "500m",
                        memory: "1Gi"
                    },
                    defaultRequest: {
                        cpu: "250m",
                        memory: "512Mi"
                    },
                    max: {
                        cpu: "8",
                        memory: "24Gi"
                    },
                    min: {
                        cpu: "100m",
                        memory: "128Mi"
                    }
                }]
            }
        }, { parent: this.namespace });

        // Create network policy for the AI namespace
        this.networkPolicy = new k8s.networking.v1.NetworkPolicy("ai-network-policy", {
            metadata: {
                namespace: this.namespace.metadata.name,
                name: "ai-network-policy"
            },
            spec: {
                podSelector: {
                    matchLabels: {}  // Apply to all pods in the namespace
                },
                policyTypes: ["Ingress", "Egress"],
                ingress: [{
                    from: [
                        // Allow traffic from within the AI namespace
                        { podSelector: { matchLabels: {} } },
                        // Allow traffic from monitoring namespace
                        {
                            namespaceSelector: {
                                matchLabels: {
                                    "kubernetes.io/metadata.name": "monitoring"
                                }
                            }
                        },
                        // Allow traffic from home-automation namespace (for Home Assistant)
                        {
                            namespaceSelector: {
                                matchLabels: {
                                    "kubernetes.io/metadata.name": "home-automation"
                                }
                            }
                        }
                    ]
                }],
                egress: [{
                    // Allow all outbound traffic for model downloading and updates
                    to: [{}]
                }]
            }
        }, { parent: this.namespace });

        // Register all resources
        this.registerOutputs({
            namespaceName: this.namespace.metadata.name,
            resourceQuota: this.resourceQuota.metadata.name,
            limitRange: this.limitRange.metadata.name,
            networkPolicy: this.networkPolicy.metadata.name,
        });
    }
}

// Export the AINamespace class
export default AINamespace;
