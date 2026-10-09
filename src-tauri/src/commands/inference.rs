//! Inference selection and model cache lifecycle.

use maa_framework::resource::Resource;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};
use tauri::Manager;

use super::types::MaaState;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum InferenceDevice {
    #[default]
    Auto,
    Cpu,
    Directml {
        #[serde(rename = "instancePath")]
        instance_path: Option<String>,
        name: String,
    },
}

impl InferenceDevice {
    pub fn from_config(config: &serde_json::Value) -> Self {
        config
            .pointer("/settings/inferenceDevice")
            .cloned()
            .map_or_else(Self::default, |value| {
                serde_json::from_value(value).unwrap_or_else(|error| {
                    log::warn!(
                        "Invalid inference device setting, using automatic selection: {error}"
                    );
                    Self::default()
                })
            })
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InferenceAdapter {
    pub index: i32,
    pub name: String,
    pub instance_path: Option<String>,
    pub dedicated_memory: u64,
}

#[derive(Serialize)]
pub struct InferenceDevices {
    pub adapters: Vec<InferenceAdapter>,
    pub active: HashMap<String, Option<InferenceDevice>>,
}

#[tauri::command]
pub fn maa_get_inference_devices(
    state: tauri::State<'_, std::sync::Arc<super::types::MaaState>>,
) -> Result<InferenceDevices, String> {
    get_inference_devices(&state)
}

pub fn get_inference_devices(state: &super::types::MaaState) -> Result<InferenceDevices, String> {
    Ok(InferenceDevices {
        adapters: enumerate_adapters()?,
        active: state
            .instances
            .lock()
            .map_err(|e| e.to_string())?
            .iter()
            .filter(|(_, instance)| instance.resource.is_some())
            .map(|(id, instance)| (id.clone(), instance.inference_device.clone()))
            .collect(),
    })
}

pub fn saved_device(app: &tauri::AppHandle) -> Result<InferenceDevice, String> {
    let config = app.state::<Arc<super::app_config::AppConfigState>>();
    let value = config.config.lock().map_err(|e| e.to_string())?;
    Ok(InferenceDevice::from_config(&value))
}

pub fn resource_operation(
    state: &MaaState,
    instance_id: &str,
) -> Result<tokio::sync::OwnedMutexGuard<()>, String> {
    let operation = state
        .instances
        .lock()
        .map_err(|e| e.to_string())?
        .get(instance_id)
        .ok_or("Instance not found")?
        .resource_operation
        .clone();
    operation
        .try_lock_owned()
        .map_err(|_| "INFERENCE_DEVICE_BUSY".to_string())
}

/// Call while holding resource_operation, before the first batch of a new run.
/// Preserve the resource handle, custom registrations, sinks and controller binding.
pub fn prepare_inference(
    state: &MaaState,
    instance_id: &str,
    device: &InferenceDevice,
) -> Result<(), String> {
    let (resource, paths) = {
        let mut instances = state.instances.lock().map_err(|e| e.to_string())?;
        let instance = instances.get_mut(instance_id).ok_or("Instance not found")?;
        if instance
            .tasker
            .as_ref()
            .is_some_and(|tasker| tasker.running())
        {
            return Err("INFERENCE_DEVICE_BUSY".to_string());
        }
        instance.stop_in_progress = false;
        instance.stop_started_at = None;
        let resource = instance.resource.as_ref().ok_or("Resource not loaded")?;
        if instance.inference_device.as_ref() == Some(device) && resource.loaded() {
            return Ok(());
        }
        (resource.clone(), instance.resource_paths.clone())
    };
    // Resolve the hardware before clearing anything. A missing GPU leaves the old
    // resource intact, but prevents this run from silently using the wrong device.
    let index = device_index(device)?;
    if paths.is_empty() {
        return Err("INFERENCE_DEVICE_RELOAD_FAILED: no resource bundles".to_string());
    }
    resource
        .clear()
        .map_err(|e| format!("INFERENCE_DEVICE_RELOAD_FAILED: {e}"))?;
    {
        let mut instances = state.instances.lock().map_err(|e| e.to_string())?;
        instances
            .get_mut(instance_id)
            .ok_or("Instance not found")?
            .inference_device = None;
    }
    configure_resource_index(&resource, device, index)?;
    for path in paths {
        let job = resource
            .post_bundle(&path)
            .map_err(|e| format!("INFERENCE_DEVICE_RELOAD_FAILED: {path}: {e}"))?;
        if job.wait() != maa_framework::MaaStatus::SUCCEEDED {
            return Err(format!("INFERENCE_DEVICE_RELOAD_FAILED: {path}"));
        }
    }
    let mut instances = state.instances.lock().map_err(|e| e.to_string())?;
    instances
        .get_mut(instance_id)
        .ok_or("Instance not found")?
        .inference_device = Some(device.clone());
    log::info!("Inference device ready for instance {instance_id}: {device:?}");
    Ok(())
}

fn resolve_adapter<'a>(
    adapters: &'a [InferenceAdapter],
    instance_path: Option<&str>,
    name: &str,
) -> Result<&'a InferenceAdapter, String> {
    // A saved hardware identity must never fall back to a different card with the same name.
    let mut matches = adapters.iter().filter(|adapter| match instance_path {
        Some(path) => adapter
            .instance_path
            .as_deref()
            .is_some_and(|id| id.eq_ignore_ascii_case(path)),
        None => adapter.name == name,
    });
    match (matches.next(), matches.next()) {
        (Some(adapter), None) => Ok(adapter),
        _ => Err(format!("INFERENCE_DEVICE_UNAVAILABLE: {name}")),
    }
}

pub fn create_resource(device: &InferenceDevice) -> Result<Resource, String> {
    let resource = Resource::new().map_err(|e| e.to_string())?;
    configure_resource(&resource, device)?;
    Ok(resource)
}

fn configure_resource(resource: &Resource, device: &InferenceDevice) -> Result<(), String> {
    configure_resource_index(resource, device, device_index(device)?)
}

fn device_index(device: &InferenceDevice) -> Result<Option<i32>, String> {
    match device {
        InferenceDevice::Directml {
            instance_path,
            name,
        } => {
            let adapters = enumerate_adapters()?;
            Ok(Some(
                resolve_adapter(&adapters, instance_path.as_deref(), name)?.index,
            ))
        }
        _ => Ok(None),
    }
}

fn configure_resource_index(
    resource: &Resource,
    device: &InferenceDevice,
    index: Option<i32>,
) -> Result<(), String> {
    let result = match device {
        InferenceDevice::Auto => resource.use_auto_ep(),
        InferenceDevice::Cpu => resource.use_cpu(),
        InferenceDevice::Directml { name, .. } => {
            let index = index.ok_or("INFERENCE_DEVICE_UNAVAILABLE")?;
            log::info!("Inference device: DirectML index={}, name={}", index, name);
            resource.use_directml(index)
        }
    };
    result.map_err(|e| format!("INFERENCE_DEVICE_RELOAD_FAILED: {e}"))?;
    log::info!("Inference setting applied before loading models: {device:?}");
    Ok(())
}

#[cfg(not(windows))]
pub fn enumerate_adapters() -> Result<Vec<InferenceAdapter>, String> {
    Ok(Vec::new())
}

#[cfg(windows)]
pub fn enumerate_adapters() -> Result<Vec<InferenceAdapter>, String> {
    use windows::Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D12::{D3D12CreateDevice, ID3D12Device},
        Dxgi::{
            CreateDXGIFactory1, IDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
        },
    };

    let mut adapters = Vec::new();
    // DXGI/Direct3D are not wrapped by WinSafe. COM interfaces own their references.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().map_err(|e| e.to_string())?;
        for index in 0.. {
            let adapter = match factory.EnumAdapters1(index) {
                Ok(adapter) => adapter,
                Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(error) => return Err(error.to_string()),
            };
            let desc = adapter.GetDesc1().map_err(|e| e.to_string())?;
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            // Indirect display drivers can borrow a physical GPU's DXGI name and
            // D3D12 support (e.g. MuMu's virtual display). Check the adapter's own LUID.
            match is_software_or_indirect_adapter(desc.AdapterLuid) {
                Ok(true) => {
                    log::debug!("Skipping software or indirect display adapter: index={index}");
                    continue;
                }
                Ok(false) => {}
                Err(error) => {
                    log::warn!("Could not query adapter type for index={index}: {error}");
                }
            }
            let mut device: Option<ID3D12Device> = None;
            if D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).is_err() {
                continue;
            }
            let length = desc
                .Description
                .iter()
                .position(|&c| c == 0)
                .unwrap_or(desc.Description.len());
            adapters.push(InferenceAdapter {
                // Keep the original DXGI index, even when earlier adapters are filtered out.
                index: index as i32,
                name: String::from_utf16_lossy(&desc.Description[..length]),
                instance_path: adapter_instance_path(desc.AdapterLuid),
                dedicated_memory: desc.DedicatedVideoMemory as u64,
            });
        }
    }
    Ok(adapters)
}

#[cfg(windows)]
fn is_software_or_indirect_adapter(luid: windows::Win32::Foundation::LUID) -> Result<bool, String> {
    use windows::Wdk::Graphics::Direct3D::{
        D3DKMTCloseAdapter, D3DKMTOpenAdapterFromLuid, D3DKMTQueryAdapterInfo, D3DKMT_ADAPTERTYPE,
        D3DKMT_CLOSEADAPTER, D3DKMT_OPENADAPTERFROMLUID, D3DKMT_QUERYADAPTERINFO,
        KMTQAITYPE_ADAPTERTYPE,
    };

    let mut adapter = D3DKMT_OPENADAPTERFROMLUID {
        AdapterLuid: luid,
        ..Default::default()
    };
    // These GDI kernel-thunk APIs are not wrapped by WinSafe. The handle is closed
    // immediately after the query, including when the query fails.
    unsafe {
        let status = D3DKMTOpenAdapterFromLuid(&mut adapter);
        if status.0 < 0 {
            return Err(format!("D3DKMTOpenAdapterFromLuid: {:#x}", status.0));
        }
        let mut adapter_type = D3DKMT_ADAPTERTYPE::default();
        let mut query = D3DKMT_QUERYADAPTERINFO {
            hAdapter: adapter.hAdapter,
            Type: KMTQAITYPE_ADAPTERTYPE,
            pPrivateDriverData: std::ptr::from_mut(&mut adapter_type).cast(),
            PrivateDriverDataSize: std::mem::size_of_val(&adapter_type) as u32,
        };
        let status = D3DKMTQueryAdapterInfo(&mut query);
        let close_status = D3DKMTCloseAdapter(&D3DKMT_CLOSEADAPTER {
            hAdapter: adapter.hAdapter,
        });
        if close_status.0 < 0 {
            log::warn!("D3DKMTCloseAdapter: {:#x}", close_status.0);
        }
        if status.0 < 0 {
            return Err(format!("D3DKMTQueryAdapterInfo: {:#x}", status.0));
        }
        // D3DKMT_ADAPTERTYPE bitfields from d3dkmthk.h; windows exposes their raw value.
        // https://learn.microsoft.com/windows-hardware/drivers/ddi/d3dkmthk/ns-d3dkmthk-_d3dkmt_adaptertype
        const SOFTWARE_DEVICE: u32 = 1 << 2;
        const INDIRECT_DISPLAY_DEVICE: u32 = 1 << 6;
        Ok(adapter_type.Anonymous.Value & (SOFTWARE_DEVICE | INDIRECT_DISPLAY_DEVICE) != 0)
    }
}

#[cfg(windows)]
fn adapter_instance_path(luid: windows::Win32::Foundation::LUID) -> Option<String> {
    use windows::{
        core::PCWSTR,
        Win32::Devices::{
            DeviceAndDriverInstallation::{
                CM_Get_Device_Interface_PropertyW, CR_BUFFER_SMALL, CR_SUCCESS,
            },
            Display::{
                DisplayConfigGetDeviceInfo, DISPLAYCONFIG_ADAPTER_NAME,
                DISPLAYCONFIG_DEVICE_INFO_GET_ADAPTER_NAME,
            },
            Properties::{DEVPKEY_Device_InstanceId, DEVPROPTYPE, DEVPROP_TYPE_STRING},
        },
    };
    let mut info = DISPLAYCONFIG_ADAPTER_NAME::default();
    info.header.r#type = DISPLAYCONFIG_DEVICE_INFO_GET_ADAPTER_NAME;
    info.header.size = std::mem::size_of_val(&info) as u32;
    info.header.adapterId = luid;
    unsafe {
        if DisplayConfigGetDeviceInfo(&mut info.header) != 0 {
            return None;
        }
        let mut property_type = DEVPROPTYPE::default();
        let mut size = 0;
        let path = PCWSTR(info.adapterDevicePath.as_ptr());
        if CM_Get_Device_Interface_PropertyW(
            path,
            &DEVPKEY_Device_InstanceId,
            &mut property_type,
            None,
            &mut size,
            0,
        ) != CR_BUFFER_SMALL
        {
            return None;
        }
        let mut buffer = vec![0u16; (size as usize).div_ceil(2)];
        if CM_Get_Device_Interface_PropertyW(
            path,
            &DEVPKEY_Device_InstanceId,
            &mut property_type,
            Some(buffer.as_mut_ptr().cast()),
            &mut size,
            0,
        ) != CR_SUCCESS
            || property_type != DEVPROP_TYPE_STRING
        {
            return None;
        }
        let length = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        (length > 0).then(|| String::from_utf16_lossy(&buffer[..length]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Supply an isolated runtime, OCR resource bundle and a PNG containing "MXU 12345".
    /// Run with --ignored --nocapture; inspect maafw.log for the actual execution provider.
    /// MXU_TEST_CONFIG accepts one config or an array for repeated in-process switches.
    /// Set MXU_TEST_AGENT=1 to exercise RPC, and MXU_TEST_AGENT_TCP=1 for TCP mode.
    #[cfg(windows)]
    #[test]
    #[ignore = "requires MaaFramework DLLs, OCR models and local GPU hardware"]
    fn hardware_ocr_smoke() {
        use maa_framework::{buffer::MaaImageBuffer, tasker::Tasker, toolkit::Toolkit, MaaStatus};
        let runtime = std::env::var("MXU_TEST_RUNTIME").expect("MXU_TEST_RUNTIME");
        let bundle = std::env::var("MXU_TEST_RESOURCE").expect("MXU_TEST_RESOURCE");
        let image_path = std::env::var("MXU_TEST_IMAGE").expect("MXU_TEST_IMAGE");
        let log_dir = std::env::var("MXU_TEST_LOG_DIR").expect("MXU_TEST_LOG_DIR");
        winsafe::SetDllDirectory(Some(&runtime)).unwrap();
        maa_framework::load_library(&std::path::Path::new(&runtime).join("MaaFramework.dll"))
            .unwrap();
        Toolkit::init_option(&log_dir, "{}").unwrap();
        let mut image = MaaImageBuffer::new().unwrap();
        image
            .set_encoded(&std::fs::read(image_path).unwrap())
            .unwrap();
        let adapters = enumerate_adapters().unwrap();
        println!("Adapters: {}", serde_json::to_string(&adapters).unwrap());
        let config_path = std::env::var("MXU_TEST_CONFIG").expect("MXU_TEST_CONFIG");
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
        let configs = config.as_array().cloned().unwrap_or_else(|| vec![config]);
        use super::super::types::InstanceRuntime;
        use maa_framework::{agent_client::AgentClient, custom::FnRecognition};
        use std::os::windows::process::CommandExt;
        let state = Arc::new(MaaState::default());
        let resource = Resource::new().unwrap();
        let tasker = Tasker::new().unwrap();
        tasker.bind_resource(&resource).unwrap();
        let other = create_resource(&InferenceDevice::Cpu).unwrap();
        assert_eq!(
            other.post_bundle(&bundle).unwrap().wait(),
            MaaStatus::SUCCEEDED
        );
        let other_tasker = Tasker::new().unwrap();
        other_tasker.bind_resource(&other).unwrap();
        let mut other_instance = InstanceRuntime::default();
        other_instance.resource = Some(other);
        other_instance.tasker = Some(other_tasker.clone());
        other_instance.inference_device = Some(InferenceDevice::Cpu);
        other_instance.resource_paths = vec![bundle.clone()];
        state
            .instances
            .lock()
            .unwrap()
            .insert("other".into(), other_instance);
        let mut instance = InstanceRuntime::default();
        instance.resource = Some(resource.clone());
        instance.tasker = Some(tasker.clone());
        instance.resource_paths = vec![bundle.clone()];
        state
            .instances
            .lock()
            .unwrap()
            .insert("test".into(), instance);
        // Opt in to Agent RPC; otherwise exercise a local custom registration.
        let use_agent = std::env::var_os("MXU_TEST_AGENT").is_some();
        if use_agent {
            let mut client = if std::env::var_os("MXU_TEST_AGENT_TCP").is_some() {
                AgentClient::create_tcp(0).unwrap()
            } else {
                AgentClient::new(None).unwrap()
            };
            client.bind(resource.clone()).unwrap();
            let child = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["hardware_agent_server", "--ignored", "--nocapture"])
                .env("MXU_TEST_AGENT_ID", client.identifier().unwrap())
                .current_dir(&log_dir)
                .creation_flags(0x08000000)
                .spawn()
                .unwrap();
            state
                .instances
                .lock()
                .unwrap()
                .get_mut("test")
                .unwrap()
                .agent_children
                .push(child);
            client.connect().unwrap();
            client.register_resource_sink(resource.clone()).unwrap();
            state
                .instances
                .lock()
                .unwrap()
                .get_mut("test")
                .unwrap()
                .agent_clients
                .push(client);
        } else {
            resource
                .register_custom_recognition("AgentOcr", ocr_recognizer())
                .unwrap();
            println!("Agent RPC not requested; testing local custom recognition");
        }
        let weak = Arc::downgrade(&state);
        resource
            .add_sink(move |_, _| {
                // Would deadlock if reloading held the global instances mutex.
                if let Some(state) = weak.upgrade() {
                    drop(state.instances.lock().unwrap());
                }
            })
            .unwrap();
        for config in configs {
            let device = InferenceDevice::from_config(&config);
            println!("OCR START: {device:?}");
            let _operation = resource_operation(&state, "test").unwrap();
            prepare_inference(&state, "test", &device).unwrap();
            let job = tasker
                .post_recognition("OCR", r#"{"expected":"MXU 12345"}"#, &image)
                .unwrap();
            assert_eq!(job.wait(), MaaStatus::SUCCEEDED);
            // post_recognition returns a task ID; the binding's job.get() treats it as a reco ID.
            let task = tasker
                .get_task_detail(job.id)
                .unwrap()
                .expect("task detail");
            let node = tasker
                .get_node_detail(task.node_id_list[0])
                .unwrap()
                .expect("node detail");
            let detail = tasker
                .get_recognition_detail(node.reco_id)
                .unwrap()
                .expect("OCR result");
            println!("OCR RESULT: {device:?}: {}", detail.detail);
            assert!(detail.hit, "OCR failed for {device:?}");
            let job = tasker
                .post_recognition("Custom", r#"{"custom_recognition":"AgentOcr"}"#, &image)
                .unwrap();
            assert_eq!(job.wait(), MaaStatus::SUCCEEDED);
            let task = tasker.get_task_detail(job.id).unwrap().unwrap();
            let node = tasker
                .get_node_detail(task.node_id_list[0])
                .unwrap()
                .unwrap();
            assert!(
                tasker
                    .get_recognition_detail(node.reco_id)
                    .unwrap()
                    .unwrap()
                    .hit
            );
            println!("CUSTOM OCR PASSED (agent={use_agent}): {device:?}");
            let job = other_tasker
                .post_recognition("OCR", r#"{"expected":"MXU 12345"}"#, &image)
                .unwrap();
            assert_eq!(job.wait(), MaaStatus::SUCCEEDED);
            assert_eq!(
                state.instances.lock().unwrap()["other"].inference_device,
                Some(InferenceDevice::Cpu)
            );
        }
        let _operation = resource_operation(&state, "test").unwrap();
        // Missing hardware must not mutate an already usable resource.
        let before = state.instances.lock().unwrap()["test"]
            .inference_device
            .clone();
        let missing = InferenceDevice::Directml {
            instance_path: Some("missing-test-device".into()),
            name: "Missing GPU".into(),
        };
        assert!(prepare_inference(&state, "test", &missing)
            .unwrap_err()
            .contains("UNAVAILABLE"));
        assert_eq!(
            state.instances.lock().unwrap()["test"].inference_device,
            before
        );
        // A real running custom recognition prevents any cache/device changes.
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = std::sync::Mutex::new(release_rx);
        resource
            .register_custom_recognition(
                "Hold",
                Box::new(FnRecognition::new(move |_, args| {
                    entered_tx.send(()).unwrap();
                    release_rx
                        .lock()
                        .unwrap()
                        .recv_timeout(std::time::Duration::from_secs(10))
                        .unwrap();
                    Some((*args.roi, "{}".into()))
                })),
            )
            .unwrap();
        let job = tasker
            .post_recognition("Custom", r#"{"custom_recognition":"Hold"}"#, &image)
            .unwrap();
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap();
        assert!(prepare_inference(&state, "test", &InferenceDevice::Auto)
            .unwrap_err()
            .contains("BUSY"));
        assert_eq!(
            state.instances.lock().unwrap()["test"].inference_device,
            before
        );
        release_tx.send(()).unwrap();
        assert_eq!(job.wait(), MaaStatus::SUCCEEDED);
        // Reload failures cannot be recorded as applied. A corrected path can recover.
        {
            let mut instances = state.instances.lock().unwrap();
            let instance = instances.get_mut("test").unwrap();
            instance
                .resource_paths
                .push(format!("{bundle}/missing-test-bundle"));
            instance.inference_device = None;
        }
        assert!(prepare_inference(&state, "test", &InferenceDevice::Cpu)
            .unwrap_err()
            .contains("RELOAD_FAILED"));
        assert!(state.instances.lock().unwrap()["test"]
            .inference_device
            .is_none());
        state
            .instances
            .lock()
            .unwrap()
            .get_mut("test")
            .unwrap()
            .resource_paths
            .pop();
        prepare_inference(&state, "test", &InferenceDevice::Cpu).unwrap();
        println!("LIFECYCLE PASSED: missing device, running guard, multi-instance isolation, failed reload and recovery");
    }

    #[cfg(windows)]
    fn ocr_recognizer() -> Box<dyn maa_framework::custom::CustomRecognition> {
        Box::new(maa_framework::custom::FnRecognition::new(
            |context, args| {
                let expected =
                    std::env::var("MXU_TEST_EXPECTED").unwrap_or_else(|_| "MXU 12345".to_string());
                let detail = context
                    .run_recognition_direct(
                        "OCR",
                        &serde_json::json!({"expected": expected}).to_string(),
                        args.image,
                    )
                    .ok()??;
                detail
                    .hit
                    .then_some((detail.box_rect, detail.detail.to_string()))
            },
        ))
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "subprocess fixture for hardware_ocr_smoke"]
    fn hardware_agent_server() {
        use maa_framework::agent_server::AgentServer;
        let runtime = std::env::var("MXU_TEST_RUNTIME").unwrap();
        winsafe::SetDllDirectory(Some(&runtime)).unwrap();
        maa_framework::load_library(&std::path::Path::new(&runtime).join("MaaAgentServer.dll"))
            .unwrap();
        AgentServer::register_custom_recognition("AgentOcr", ocr_recognizer()).unwrap();
        AgentServer::add_resource_sink(|_, _| {}).unwrap();
        AgentServer::start_up(&std::env::var("MXU_TEST_AGENT_ID").unwrap()).unwrap();
        AgentServer::join();
        AgentServer::shut_down();
    }

    #[test]
    fn resource_operations_are_exclusive_per_instance() {
        let state = MaaState::default();
        state
            .instances
            .lock()
            .unwrap()
            .insert("a".into(), Default::default());
        state
            .instances
            .lock()
            .unwrap()
            .insert("b".into(), Default::default());
        let guard = resource_operation(&state, "a").unwrap();
        assert!(resource_operation(&state, "a").is_err());
        assert!(resource_operation(&state, "b").is_ok());
        drop(guard);
        assert!(resource_operation(&state, "a").is_ok());
    }

    fn adapter(index: i32, path: &str, name: &str, memory: u64) -> InferenceAdapter {
        InferenceAdapter {
            index,
            instance_path: Some(path.into()),
            name: name.into(),
            dedicated_memory: memory,
        }
    }

    #[test]
    fn resolves_identity_after_reordering_without_preferring_larger_memory() {
        let adapters = [
            adapter(0, "integrated", "Integrated", 8 << 30),
            adapter(3, "discrete", "Discrete", 6 << 30),
        ];
        assert_eq!(
            resolve_adapter(&adapters, Some("DISCRETE"), "Old name")
                .unwrap()
                .index,
            3
        );
    }

    #[test]
    fn missing_identity_does_not_select_same_name_replacement() {
        let adapters = [adapter(0, "replacement", "GPU", 0)];
        assert!(resolve_adapter(&adapters, Some("missing"), "GPU").is_err());
        assert_eq!(resolve_adapter(&adapters, None, "GPU").unwrap().index, 0);
    }

    #[test]
    fn ambiguous_names_are_rejected() {
        let adapters = [adapter(0, "one", "GPU", 0), adapter(1, "two", "GPU", 0)];
        assert!(resolve_adapter(&adapters, None, "GPU").is_err());
    }

    #[test]
    fn legacy_config_defaults_to_auto_and_device_roundtrips() {
        assert_eq!(
            InferenceDevice::from_config(&serde_json::json!({"settings": {}})),
            InferenceDevice::Auto
        );
        let config = serde_json::json!({"settings": {"inferenceDevice": {"type": "directml", "instancePath": "PCI\\GPU", "name": "GPU"}}});
        let device = InferenceDevice::from_config(&config);
        assert_eq!(
            serde_json::to_value(device).unwrap(),
            config["settings"]["inferenceDevice"]
        );
    }
}
