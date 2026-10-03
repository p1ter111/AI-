const { app, BrowserWindow, Menu, Tray, ipcMain, nativeImage, screen } = require('electron');
const https = require('node:https');
const path = require('node:path');

let mainWindow;
let tray;
let isQuitting = false;
let alwaysOnTop = true;

const DEFAULT_BOUNDS = {
  width: 360,
  height: 620,
};

const iconSvg = `
<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">
  <rect width="64" height="64" rx="14" fill="#20242a"/>
  <rect x="12" y="15" width="40" height="37" rx="7" fill="#f4f2ea"/>
  <path d="M12 24h40" stroke="#d3503f" stroke-width="5"/>
  <path d="M22 11v9M42 11v9" stroke="#f4f2ea" stroke-width="5" stroke-linecap="round"/>
  <circle cx="32" cy="38" r="9" fill="#20242a"/>
  <path d="M32 32v7l5 3" stroke="#f4f2ea" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

function createIcon() {
  return nativeImage.createFromDataURL(`data:image/svg+xml;charset=utf-8,${encodeURIComponent(iconSvg)}`);
}

function getInitialBounds() {
  const display = screen.getPrimaryDisplay().workArea;
  return {
    ...DEFAULT_BOUNDS,
    x: Math.max(display.x + display.width - DEFAULT_BOUNDS.width - 28, display.x),
    y: Math.max(display.y + 32, display.y),
  };
}

function createWindow() {
  mainWindow = new BrowserWindow({
    ...getInitialBounds(),
    minWidth: 320,
    minHeight: 560,
    maxWidth: 460,
    maxHeight: 780,
    frame: false,
    transparent: true,
    resizable: true,
    hasShadow: true,
    alwaysOnTop: true,
    show: false,
    skipTaskbar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'renderer', 'index.html'));

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
  });

  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault();
      mainWindow.hide();
    }
  });
}

function createTray() {
  if (tray) {
    tray.destroy();
  }

  tray = new Tray(createIcon());
  tray.setToolTip('桌面台历时钟');
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: '显示/隐藏', click: toggleWindow },
    { type: 'separator' },
    { label: '悬浮置顶', type: 'checkbox', checked: alwaysOnTop, click: (item) => setAlwaysOnTop(item.checked) },
    { label: '开机自启', type: 'checkbox', checked: app.getLoginItemSettings().openAtLogin, click: (item) => setOpenAtLogin(item.checked) },
    { type: 'separator' },
    { label: '退出', click: () => { isQuitting = true; app.quit(); } },
  ]));
  tray.on('click', toggleWindow);
}

function toggleWindow() {
  if (!mainWindow) return;
  if (mainWindow.isVisible()) {
    mainWindow.hide();
  } else {
    mainWindow.show();
    mainWindow.focus();
  }
}

function setAlwaysOnTop(enabled) {
  if (!mainWindow) return;
  alwaysOnTop = Boolean(enabled);
  mainWindow.setAlwaysOnTop(alwaysOnTop, alwaysOnTop ? 'screen-saver' : 'normal');
  mainWindow.setVisibleOnAllWorkspaces(alwaysOnTop, { visibleOnFullScreen: false });
  mainWindow.webContents.send('settings:alwaysOnTop', alwaysOnTop);
  createTray();
}

function setOpenAtLogin(enabled) {
  app.setLoginItemSettings({ openAtLogin: enabled });
}

const TIME_SOURCES = [
  'https://www.baidu.com',
  'https://www.qq.com',
  'https://www.microsoft.com',
  'https://worldtimeapi.org/api/timezone/Etc/UTC',
];

function requestServerTime(url) {
  return new Promise((resolve, reject) => {
    const startedAt = Date.now();
    const request = https.request(url, { method: 'HEAD', timeout: 6000 }, (response) => {
      const endedAt = Date.now();
      const dateHeader = response.headers.date;
      response.resume();

      if (!dateHeader) {
        reject(new Error(`No Date header from ${url}`));
        return;
      }

      const serverTime = Date.parse(dateHeader);
      if (!Number.isFinite(serverTime)) {
        reject(new Error(`Invalid Date header from ${url}`));
        return;
      }

      resolve({
        source: new URL(url).hostname,
        offsetMs: serverTime - Math.round((startedAt + endedAt) / 2),
        serverTime,
        checkedAt: endedAt,
      });
    });

    request.on('timeout', () => {
      request.destroy(new Error(`Timeout while checking ${url}`));
    });
    request.on('error', reject);
    request.end();
  });
}

async function syncNetworkTime() {
  const errors = [];

  for (const source of TIME_SOURCES) {
    try {
      return await requestServerTime(source);
    } catch (error) {
      errors.push(error.message);
    }
  }

  throw new Error(errors.join('; '));
}

app.whenReady().then(() => {
  app.setAppUserModelId('com.desktop.calendarclock');
  createWindow();
  createTray();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
    else toggleWindow();
  });
});

app.on('before-quit', () => {
  isQuitting = true;
});

app.on('window-all-closed', (event) => {
  event.preventDefault();
});

ipcMain.handle('window:minimize', () => mainWindow?.hide());
ipcMain.handle('window:close', () => mainWindow?.hide());
ipcMain.handle('window:set-opacity', (_event, opacity) => {
  const value = Number(opacity);
  if (!Number.isFinite(value)) return;
  mainWindow?.setOpacity(Math.min(1, Math.max(0.55, value)));
});
ipcMain.handle('window:set-always-on-top', (_event, enabled) => {
  setAlwaysOnTop(Boolean(enabled));
});
ipcMain.handle('app:set-open-at-login', (_event, enabled) => {
  setOpenAtLogin(Boolean(enabled));
});
ipcMain.handle('app:get-login-settings', () => app.getLoginItemSettings());
ipcMain.handle('time:sync', () => syncNetworkTime());
