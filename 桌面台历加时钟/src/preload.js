const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('desktopClock', {
  minimize: () => ipcRenderer.invoke('window:minimize'),
  close: () => ipcRenderer.invoke('window:close'),
  setOpacity: (opacity) => ipcRenderer.invoke('window:set-opacity', opacity),
  setAlwaysOnTop: (enabled) => ipcRenderer.invoke('window:set-always-on-top', enabled),
  setOpenAtLogin: (enabled) => ipcRenderer.invoke('app:set-open-at-login', enabled),
  getLoginSettings: () => ipcRenderer.invoke('app:get-login-settings'),
  syncTime: () => ipcRenderer.invoke('time:sync'),
  onAlwaysOnTopChanged: (callback) => {
    ipcRenderer.on('settings:alwaysOnTop', (_event, enabled) => callback(Boolean(enabled)));
  },
});
