import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface MinioOptions {
    namespace: string;
    storageSize?: string;
    storageClassName?: string;
    memoryLimit?: string;
    cpuLimit?: string;
    memoryRequest?: string;
    cpuRequest?: string;
    buckets?: string[];
    ingressHost?: string;
    tlsSecretName?: string;
}

export class MinIODeployment extends pulumi.ComponentResource {
    public readonly operator: k8s.apiextensions.CustomResource;
    public readonly tenant: k8s.apiextensions.CustomResource;
    public readonly service: k8s.core.v1.Service;
    public readonly ingress: k8s.networking.v1.Ingress;
    public readonly serviceAccount: k8s.core.v1.ServiceAccount;

    constructor(name: string, options: MinioOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:ai:MinIODeployment", name, {}, opts);

        // Set default values if not provided
        const namespace = options.namespace;
        const storageSize = options.storageSize || "200Gi";
        const storageClassName = options.storageClassName || "high-performance";
        const memoryLimit = options.memoryLimit || "4Gi";
        const cpuLimit = options.cpuLimit || "2";
        const memoryRequest = options.memoryRequest || "2Gi";
        const cpuRequest = options.cpuRequest || "1";
        const buckets = options.buckets || ["models", "checkpoints", "datasets", "configs"];
        const ingressHost = options.ingressHost || "minio.homelab.local";
        const tlsSecretName = options.tlsSecretName || "minio-tls";

        // Create ServiceAccount for MinIO Operator
        const operatorServiceAccount = new k8s.core.v1.ServiceAccount("minio-operator-sa", {
            metadata: {
                name: "minio-operator",
                namespace: namespace
            }
        }, { parent: this });

        // Create ClusterRole for MinIO Operator
        const operatorClusterRole = new k8s.rbac.v1.ClusterRole("minio-operator-role", {
            metadata: {
                name: "minio-operator-role"
            },
            rules: [
                {
                    apiGroups: [""],
                    resources: ["namespaces", "secrets", "services", "endpoints", "pods"],
                    verbs: ["get", "watch", "list", "create", "update", "patch", "delete"]
                },
                {
                    apiGroups: ["apps"],
                    resources: ["statefulsets", "deployments"],
                    verbs: ["get", "watch", "list", "create", "update", "patch", "delete"]
                },
                {
                    apiGroups: ["batch"],
                    resources: ["jobs"],
                    verbs: ["get", "watch", "list", "create", "update", "patch", "delete"]
                },
                {
                    apiGroups: ["minio.min.io"],
                    resources: ["*"],
                    verbs: ["*"]
                },
                {
                    apiGroups: ["certificates.k8s.io"],
                    resources: ["certificatesigningrequests", "certificatesigningrequests/approval", "certificatesigningrequests/status"],
                    verbs: ["update", "create", "get", "delete", "watch"]
                }
            ]
        }, { parent: this });

        // Create ClusterRoleBinding for MinIO Operator
        const operatorClusterRoleBinding = new k8s.rbac.v1.ClusterRoleBinding("minio-operator-binding", {
            metadata: {
                name: "minio-operator-binding"
            },
            roleRef: {
                apiGroup: "rbac.authorization.k8s.io",
                kind: "ClusterRole",
                name: operatorClusterRole.metadata.name
            },
            subjects: [{
                kind: "ServiceAccount",
                name: operatorServiceAccount.metadata.name,
                namespace: namespace
            }]
        }, { parent: this });

        // Create Operator CustomResourceDefinitions
        const tenantCRD = new k8s.apiextensions.v1.CustomResourceDefinition("tenant-crd", {
            metadata: {
                name: "tenants.minio.min.io"
            },
            spec: {
                group: "minio.min.io",
                versions: [{
                    name: "v2",
                    served: true,
                    storage: true,
                    schema: {
                        openAPIV3Schema: {
                            type: "object",
                            properties: {
                                spec: {
                                    type: "object"
                                },
                                status: {
                                    type: "object"
                                }
                            }
                        }
                    }
                }],
                scope: "Namespaced",
                names: {
                    plural: "tenants",
                    singular: "tenant",
                    kind: "Tenant",
                    shortNames: ["t"]
                }
            }
        }, { parent: this });

        // Deploy MinIO Operator
        const operator = new k8s.apps.v1.Deployment("minio-operator", {
            metadata: {
                name: "minio-operator",
                namespace: namespace
            },
            spec: {
                selector: {
                    matchLabels: {
                        app: "minio-operator"
                    }
                },
                strategy: {
                    type: "Recreate"
                },
                template: {
                    metadata: {
                        labels: {
                            app: "minio-operator"
                        }
                    },
                    spec: {
                        serviceAccountName: operatorServiceAccount.metadata.name,
                        securityContext: {
                            runAsUser: 1000,
                            runAsGroup: 1000,
                            runAsNonRoot: true,
                            fsGroup: 1000
                        },
                        containers: [{
                            name: "minio-operator",
                            image: "minio/operator:v5.0.6",
                            imagePullPolicy: "IfNotPresent",
                            securityContext: {
                                allowPrivilegeEscalation: false,
                                capabilities: {
                                    drop: ["ALL"]
                                }
                            },
                            resources: {
                                requests: {
                                    memory: "64Mi",
                                    cpu: "100m"
                                },
                                limits: {
                                    memory: "128Mi",
                                    cpu: "200m"
                                }
                            }
                        }]
                    }
                }
            }
        }, { parent: this });

        // Create MinIO admin credentials secret
        const adminSecret = new k8s.core.v1.Secret("minio-admin-secret", {
            metadata: {
                name: "minio-admin-credentials",
                namespace: namespace
            },
            type: "Opaque",
            stringData: {
                MINIO_ROOT_USER: "minio-admin",
                MINIO_ROOT_PASSWORD: pulumi.output(new pulumi.random.RandomPassword("minio-admin-password", {
                    length: 20,
                    special: true
                }).result)
            }
        }, { parent: this });

        // Create MinIO user credentials for service accounts
        const apiSecret = new k8s.core.v1.Secret("minio-api-secret", {
            metadata: {
                name: "minio-api-credentials",
                namespace: namespace
            },
            type: "Opaque",
            stringData: {
                MINIO_ACCESS_KEY: "api-user",
                MINIO_SECRET_KEY: pulumi.output(new pulumi.random.RandomPassword("minio-api-password", {
                    length: 20,
                    special: true
                }).result)
            }
        }, { parent: this });

        // Create ServiceAccount for MinIO
        this.serviceAccount = new k8s.core.v1.ServiceAccount("minio-sa", {
            metadata: {
                name: "minio",
                namespace: namespace
            }
        }, { parent: this });

        // Create MinIO Tenant using Custom Resource
        this.tenant = new k8s.apiextensions.CustomResource("minio-tenant", {
            apiVersion: "minio.min.io/v2",
            kind: "Tenant",
            metadata: {
                name: "minio",
                namespace: namespace
            },
            spec: {
                serviceAccountName: this.serviceAccount.metadata.name,
                // MinIO tenant configuration
                image: "minio/minio:RELEASE.2023-05-04T21-44-30Z",
                imagePullPolicy: "IfNotPresent",
                credsSecret: {
                    name: adminSecret.metadata.name
                },
                pools: [{
                    name: "pool-1",
                    servers: 1,
                    volumesPerServer: 1,
                    volumeClaimTemplate: {
                        metadata: {
                            name: "data"
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
                    },
                    resources: {
                        requests: {
                            memory: memoryRequest,
                            cpu: cpuRequest
                        },
                        limits: {
                            memory: memoryLimit,
                            cpu: cpuLimit
                        }
                    },
                    securityContext: {
                        runAsUser: 1000,
                        runAsGroup: 1000,
                        runAsNonRoot: true,
                        fsGroup: 1000
                    }
                }],
                mountPath: "/data",
                requestAutoCert: true,
                exposeServices: {
                    minio: true,
                    console: true
                },
                podManagementPolicy: "Parallel"
            }
        }, {
            parent: this,
            dependsOn: [tenantCRD, operator]
        });

        // Create MinIO Service
        this.service = new k8s.core.v1.Service("minio-service", {
            metadata: {
                name: "minio",
                namespace: namespace,
                labels: {
                    app: "minio"
                }
            },
            spec: {
                selector: {
                    app: "minio"
                },
                ports: [
                    {
                        name: "api",
                        port: 9000,
                        targetPort: 9000
                    },
                    {
                        name: "console",
                        port: 9090,
                        targetPort: 9090
                    }
                ]
            }
        }, { parent: this });

        // Create MinIO Ingress
        this.ingress = new k8s.networking.v1.Ingress("minio-ingress", {
            metadata: {
                name: "minio-ingress",
                namespace: namespace,
                annotations: {
                    "nginx.ingress.kubernetes.io/ssl-redirect": "true",
                    "nginx.ingress.kubernetes.io/proxy-body-size": "500m",
                    "nginx.ingress.kubernetes.io/proxy-connect-timeout": "300",
                    "nginx.ingress.kubernetes.io/proxy-send-timeout": "300",
                    "nginx.ingress.kubernetes.io/proxy-read-timeout": "300",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod"
                }
            },
            spec: {
                ingressClassName: "nginx",
                tls: [{
                    hosts: [ingressHost],
                    secretName: tlsSecretName
                }],
                rules: [{
                    host: ingressHost,
                    http: {
                        paths: [
                            {
                                path: "/",
                                pathType: "Prefix",
                                backend: {
                                    service: {
                                        name: this.service.metadata.name,
                                        port: {
                                            name: "console"
                                        }
                                    }
                                }
                            },
                            {
                                path: "/api",
                                pathType: "Prefix",
                                backend: {
                                    service: {
                                        name: this.service.metadata.name,
                                        port: {
                                            name: "api"
                                        }
                                    }
                                }
                            }
                        ]
                    }
                }]
            }
        }, { parent: this });

        // Create buckets job
        buckets.forEach((bucketName, index) => {
            const bucketJob = new k8s.batch.v1.Job(`minio-create-bucket-${bucketName}`, {
                metadata: {
                    name: `minio-create-bucket-${bucketName}`,
                    namespace: namespace
                },
                spec: {
                    template: {
                        spec: {
                            containers: [{
                                name: "mc",
                                image: "minio/mc:RELEASE.2023-05-04T18-10-16Z",
                                command: ["/bin/sh", "-c"],
                                args: [
                                    `mc --insecure alias set myminio http://minio.${namespace}.svc.cluster.local:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD && ` +
                                    `mc --insecure mb --ignore-existing myminio/${bucketName} && ` +
                                    `mc --insecure policy set download myminio/${bucketName}`
                                ],
                                env: [
                                    {
                                        name: "MINIO_ROOT_USER",
                                        valueFrom: {
                                            secretKeyRef: {
                                                name: adminSecret.metadata.name,
                                                key: "MINIO_ROOT_USER"
                                            }
                                        }
                                    },
                                    {
                                        name: "MINIO_ROOT_PASSWORD",
                                        valueFrom: {
                                            secretKeyRef: {
                                                name: adminSecret.metadata.name,
                                                key: "MINIO_ROOT_PASSWORD"
                                            }
                                        }
                                    }
                                ],
                            }],
                            restartPolicy: "OnFailure"
                        }
                    },
                    backoffLimit: 5
                }
            }, {
                parent: this,
                dependsOn: [this.tenant]
            });
        });

        this.registerOutputs({
            tenantName: this.tenant.metadata.name,
            serviceName: this.service.metadata.name,
            ingressHost: ingressHost,
            adminSecretName: adminSecret.metadata.name,
            apiSecretName: apiSecret.metadata.name
        });
    }
}

export default MinIODeployment;
