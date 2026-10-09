import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Cpu } from 'lucide-react';
import { useAppStore } from '@/stores/appStore';
import { maaService } from '@/services/maaService';
import { inferenceDeviceKey, type InferenceDevice, type InferenceDevices } from '@/types/inference';

export function InferenceSection() {
  const { t } = useTranslation();
  const device = useAppStore((state) => state.inferenceDevice);
  const setDevice = useAppStore((state) => state.setInferenceDevice);
  const instances = useAppStore((state) => state.instances);
  const [devices, setDevices] = useState<InferenceDevices>();
  const [loadFailed, setLoadFailed] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | undefined;
    const refresh = () =>
      maaService.getInferenceDevices().then(
        (result) => {
          if (!cancelled) {
            setDevices(result);
            setLoadFailed(false);
          }
        },
        () => {
          if (!cancelled) setLoadFailed(true);
        },
      );
    void refresh();
    void maaService
      .onCallback((message) => {
        if (
          message === 'Inference.Device.Applied' ||
          message === 'Resource.Loading.Succeeded' ||
          message === 'Resource.Loading.Failed'
        ) {
          void refresh();
        }
      })
      .then((dispose) => {
        if (cancelled) dispose();
        else unlisten = dispose;
      })
      .catch(() => {
        if (!cancelled) setLoadFailed(true);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const options: { device: InferenceDevice; label: string }[] = [
    { device: { type: 'auto' }, label: t('inference.auto') },
    { device: { type: 'cpu' }, label: 'CPU' },
    ...(devices?.adapters ?? []).map((adapter) => ({
      device: { type: 'directml' as const, instancePath: adapter.instancePath, name: adapter.name },
      label: `${adapter.name} (DirectML · #${adapter.index})`,
    })),
  ];
  const selectedKey = inferenceDeviceKey(device);
  const unavailable =
    !!devices && !options.some((option) => inferenceDeviceKey(option.device) === selectedKey);
  const pending = Object.values(devices?.active ?? {}).some(
    (active) => !active || inferenceDeviceKey(active) !== selectedKey,
  );

  return (
    <section id="section-inference" className="space-y-4 scroll-mt-4">
      <h2 className="text-sm font-semibold text-text-primary uppercase tracking-wider flex items-center gap-2">
        <Cpu className="w-4 h-4" />
        {t('inference.title')}
      </h2>
      <div className="bg-bg-secondary rounded-xl border border-border p-4 space-y-3">
        <label htmlFor="inference-device" className="block font-medium text-text-primary">
          {t('inference.device')}
        </label>
        <select
          id="inference-device"
          value={selectedKey}
          onChange={(event) => {
            const option = options.find(
              (item) => inferenceDeviceKey(item.device) === event.target.value,
            );
            if (option) setDevice(option.device);
          }}
          className="w-full rounded-lg border border-border bg-bg-primary px-3 py-2 text-text-primary focus:outline-none focus:border-accent"
        >
          {!options.some((option) => inferenceDeviceKey(option.device) === selectedKey) && (
            <option value={selectedKey} disabled>
              {device.type === 'directml' ? device.name : selectedKey}
            </option>
          )}
          {options.map((option) => (
            <option
              key={inferenceDeviceKey(option.device)}
              value={inferenceDeviceKey(option.device)}
            >
              {option.label}
            </option>
          ))}
        </select>
        <p className="text-xs text-text-muted">{t('inference.hint')}</p>
        {Object.entries(devices?.active ?? {}).map(([id, active]) => (
          <p key={id} className="text-xs text-text-muted">
            {t('inference.current', {
              instance: instances.find((instance) => instance.id === id)?.name ?? id,
              device: active
                ? (options.find(
                    (option) => inferenceDeviceKey(option.device) === inferenceDeviceKey(active),
                  )?.label ?? (active.type === 'directml' ? active.name : active.type))
                : t('inference.notReady'),
            })}
          </p>
        ))}
        {pending && (
          <p role="status" className="text-sm text-accent">
            {t('inference.pending')}
          </p>
        )}
        {unavailable && (
          <p role="alert" className="text-sm text-red-500">
            {t('inference.unavailable')}
          </p>
        )}
        {loadFailed && (
          <p role="alert" className="text-sm text-red-500">
            {t('inference.loadFailed')}
          </p>
        )}
      </div>
    </section>
  );
}
