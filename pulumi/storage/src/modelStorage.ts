import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

export interface ModelStorageOptions {
    namespace: string;
    storageSize?: string;
    storageClassName?: string;
    accessModes?: string[];
    allowVolumeExpansion?: boolean;
    labels?: { [key: string]: string };
    annotations?: { [key: string]: string };
}

export class ModelStorage extends pulumi.ComponentResource {
    public readonly persistentVolumeClaim: k8s.core.v1.PersistentVolumeClaim;

    constructor(name: string, options: ModelStorageOptions, opts?: pulumi.ComponentResourceOptions) {
        super("homelab:storage:ModelStorage", name, {}, opts);

        // Default values if not provided
        const storageSize = options.storageSize || "500Gi"; // Minimum 500GB for large models
        const storageClassName = options.storageClassName || "high-performance"; // Use high-performance storage class
        const accessModes = options.accessModes || ["ReadWriteMany"]; // Default to ReadWriteMany for multiple pods access
        const allowVolumeExpansion = options.allowVolumeExpansion === undefined ? true : options.allowVolumeExpansion;

        // Define default labels and merge with user-provided ones
        const defaultLabels = {
            "app.kubernetes.io/name": "ai-model-storage",
            "app.kubernetes.io/part-of": "homelab-ai",
            "homelab.io/storage-type": "model-repository"
        };
        const labels = { ...defaultLabels, ...(options.labels || {}) };

        // Define default annotations and merge with user-provided ones
        const defaultAnnotations = {
            "homelab.io/description": "Persistent storage for AI models",
            "homelab.io/owner": "homelab-admin",
            "backup.velero.io/backup-volumes": "model-storage",
            "volume.beta.kubernetes.io/storage-provisioner": "kubernetes.io/no-provisioner",
            "volume.kubernetes.io/selected-node": "jetson-agx-orin" // Target the Jetson node specifically
        };
        const annotations = { ...defaultAnnotations, ...(options.annotations || {}) };

        // Create the PersistentVolumeClaim for model storage
        this.persistentVolumeClaim = new k8s.core.v1.PersistentVolumeClaim("model-storage-pvc", {
            metadata: {
                name: `${name}-pvc`,
                namespace: options.namespace,
                labels: labels,
                annotations: annotations
            },
            spec: {
                accessModes: accessModes,
                resources: {
                    requests: {
                        storage: storageSize
                    }
                },
                storageClassName: storageClassName,
                volumeMode: "Filesystem",
            }
        }, { parent: this });

        // Create a ConfigMap for model storage mount options
        const storageMountOptions = new k8s.core.v1.ConfigMap("model-storage-options", {
            metadata: {
                namespace: options.namespace,
                name: `${name}-mount-options`,
                labels: {
                    "app.kubernetes.io/name": "ai-model-storage-config",
                    "app.kubernetes.io/part-of": "homelab-ai"
                }
            },
            data: {
                "storage-path": "/models",
                "fs-type": "ext4",
                "mount-options": "defaults,noatime,nodiratime",
                "backup-schedule": "0 1 * * *", // Daily backup at 1 AM
                "backup-retention": "7d"
            }
        }, { parent: this });

        // Register outputs
        this.registerOutputs({
            pvcName: this.persistentVolumeClaim.metadata.name,
            storageSize: storageSize,
            accessModes: accessModes,
            storageClassName: storageClassName
        });
    }
}

export default ModelStorage;
