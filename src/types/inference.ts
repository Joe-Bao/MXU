export type InferenceDevice =
  | { type: 'auto' }
  | { type: 'cpu' }
  | { type: 'directml'; instancePath: string | null; name: string };

export interface InferenceAdapter {
  index: number;
  name: string;
  instancePath: string | null;
  dedicatedMemory: number;
}

export interface InferenceDevices {
  adapters: InferenceAdapter[];
  active: Record<string, InferenceDevice | null>;
}

export function inferenceDeviceKey(device: InferenceDevice): string {
  return device.type === 'directml'
    ? `directml:${device.instancePath?.toLowerCase() || device.name}`
    : device.type;
}
