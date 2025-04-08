import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export class GPUAccess extends pulumi.ComponentResource {
    constructor(name: string, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:gpu:Access", name, {}, opts);

        // NVIDIA device plugin namespace
        const namespace = new k8s.core.v1.Namespace("gpu-resources", {
            metadata: {
                name: "gpu-resources",
                labels: {
                    "pod-security.kubernetes.io/enforce": "privileged",
                    "nvidia.com/gpu-enabled": "true"
                }
            }
        }, { parent: this });

        // Node labels for GPU support
        const gpuNodeLabels = new k8s.apps.v1.DaemonSet("gpu-node-labels", {
            metadata: {
                namespace: namespace.metadata.name,
                name: "gpu-node-labels"
            },
            spec: {
                selector: {
                    matchLabels: {
                        name: "gpu-node-labels"
                    }
                },
                template: {
                    metadata: {
                        labels: {
                            name: "gpu-node-labels"
                        }
                    },
                    spec: {
                        containers: [{
                            name: "node-labels",
                            image: "k8s.gcr.io/pause:3.7",
                            resources: {
                                limits: {
                                    "nvidia.com/gpu": "1"
                                }
                            }
                        }],
                        nodeSelector: {
                            "nvidia.com/gpu-present": "true"
                        },
                        tolerations: [{
                            key: "nvidia.com/gpu",
                            operator: "Exists",
                            effect: "NoSchedule"
                        }]
                    }
                }
            }
        }, { parent: namespace });

        // NVIDIA device plugin
        const devicePlugin = new k8s.apps.v1.DaemonSet("nvidia-device-plugin", {
            metadata: {
                namespace: namespace.metadata.name,
                name: "nvidia-device-plugin-daemonset"
            },
            spec: {
                selector: {
                    matchLabels: {
                        name: "nvidia-device-plugin"
                    }
                },
                template: {
                    metadata: {
                        labels: {
                            name: "nvidia-device-plugin"
                        }
                    },
                    spec: {
                        containers: [{
                            name: "nvidia-device-plugin-ctr",
                            image: "nvcr.io/nvidia/k8s-device-plugin:v0.14.1",
                            securityContext: {
                                allowPrivilegeEscalation: false,
                                capabilities: {
                                    drop: ["ALL"]
                                }
                            },
                            volumeMounts: [{
                                name: "device-plugin",
                                mountPath: "/var/lib/kubelet/device-plugins"
                            }]
                        }],
                        volumes: [{
                            name: "device-plugin",
                            hostPath: {
                                path: "/var/lib/kubelet/device-plugins"
                            }
                        }],
                        tolerations: [{
                            key: "nvidia.com/gpu",
                            operator: "Exists",
                            effect: "NoSchedule"
                        }]
                    }
                }
            }
        }, { parent: namespace });

        // NVIDIA DCGM exporter for GPU metrics
        const dcgmExporter = new k8s.apps.v1.DaemonSet("dcgm-exporter", {
            metadata: {
                namespace: namespace.metadata.name,
                name: "dcgm-exporter"
            },
            spec: {
                selector: {
                    matchLabels: {
                        name: "dcgm-exporter"
                    }
                },
                template: {
                    metadata: {
                        labels: {
                            name: "dcgm-exporter"
                        },
                        annotations: {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9400"
                        }
                    },
                    spec: {
                        containers: [{
                            name: "dcgm-exporter",
                            image: "nvcr.io/nvidia/k8s-dcgm-exporter:3.1.7-3.1.5-ubuntu20.04",
                            ports: [{
                                name: "metrics",
                                containerPort: 9400
                            }],
                            securityContext: {
                                runAsNonRoot: true,
                                runAsUser: 65534
                            }
                        }],
                        tolerations: [{
                            key: "nvidia.com/gpu",
                            operator: "Exists",
                            effect: "NoSchedule"
                        }]
                    }
                }
            }
        }, { parent: namespace });

        // GPU feature discovery
        const featureDiscovery = new k8s.apps.v1.DaemonSet("gpu-feature-discovery", {
            metadata: {
                namespace: namespace.metadata.name,
                name: "gpu-feature-discovery"
            },
            spec: {
                selector: {
                    matchLabels: {
                        name: "gpu-feature-discovery"
                    }
                },
                template: {
                    metadata: {
                        labels: {
                            name: "gpu-feature-discovery"
                        }
                    },
                    spec: {
                        containers: [{
                            name: "gpu-feature-discovery",
                            image: "nvcr.io/nvidia/gpu-feature-discovery:v0.7.0",
                            env: [{
                                name: "NVIDIA_VISIBLE_DEVICES",
                                value: "all"
                            }],
                            securityContext: {
                                privileged: true
                            },
                            volumeMounts: [{
                                name: "host-sys",
                                mountPath: "/sys",
                                readOnly: true
                            }]
                        }],
                        volumes: [{
                            name: "host-sys",
                            hostPath: {
                                path: "/sys"
                            }
                        }],
                        tolerations: [{
                            key: "nvidia.com/gpu",
                            operator: "Exists",
                            effect: "NoSchedule"
                        }]
                    }
                }
            }
        }, { parent: namespace });

        this.registerOutputs({
            namespace: namespace.metadata.name,
            devicePluginName: devicePlugin.metadata.name,
            dcgmExporterName: dcgmExporter.metadata.name,
            featureDiscoveryName: featureDiscovery.metadata.name
        });
    }
}

export default GPUAccess;
