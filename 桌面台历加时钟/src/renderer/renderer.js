const api = window.desktopClock;

const TIME_ZONES = [
  { value: 'Asia/Shanghai', label: '中国 北京 UTC+8' },
  { value: 'Asia/Tokyo', label: '日本 东京 UTC+9' },
  { value: 'Asia/Seoul', label: '韩国 首尔 UTC+9' },
  { value: 'Asia/Singapore', label: '新加坡 UTC+8' },
  { value: 'Asia/Dubai', label: '迪拜 UTC+4' },
  { value: 'Europe/London', label: '英国 伦敦' },
  { value: 'Europe/Paris', label: '法国 巴黎' },
  { value: 'America/New_York', label: '美国 纽约' },
  { value: 'America/Los_Angeles', label: '美国 洛杉矶' },
  { value: 'Australia/Sydney', label: '澳大利亚 悉尼' },
  { value: 'UTC', label: '世界标准时间 UTC' },
];

const state = {
  visibleYear: 0,
  visibleMonth: 0,
  timeZone: localStorage.getItem('timeZone') || Intl.DateTimeFormat().resolvedOptions().timeZone || 'Asia/Shanghai',
  timeOffsetMs: Number(localStorage.getItem('timeOffsetMs') || '0'),
  lastSyncAt: Number(localStorage.getItem('lastSyncAt') || '0'),
  timeSource: localStorage.getItem('timeSource') || '',
  syncing: false,
};

const els = {
  ampm: document.getElementById('ampm'),
  time: document.getElementById('time'),
  dateLine: document.getElementById('dateLine'),
  syncLine: document.getElementById('syncLine'),
  monthTitle: document.getElementById('monthTitle'),
  todayTip: document.getElementById('todayTip'),
  calendarGrid: document.getElementById('calendarGrid'),
  prevMonth: document.getElementById('prevMonth'),
  nextMonth: document.getElementById('nextMonth'),
  minimizeBtn: document.getElementById('minimizeBtn'),
  closeBtn: document.getElementById('closeBtn'),
  timeZone: document.getElementById('timeZone'),
  syncTimeBtn: document.getElementById('syncTimeBtn'),
  alwaysOnTop: document.getElementById('alwaysOnTop'),
  openAtLogin: document.getElementById('openAtLogin'),
  opacity: document.getElementById('opacity'),
};

const pad = (value) => String(value).padStart(2, '0');

function accurateNow() {
  return new Date(Date.now() + state.timeOffsetMs);
}

function getZonedParts(date = accurateNow()) {
  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: state.timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hourCycle: 'h23',
  }).formatToParts(date);

  return Object.fromEntries(parts.filter((part) => part.type !== 'literal').map((part) => [part.type, Number(part.value)]));
}

function formatChineseDate(date) {
  return new Intl.DateTimeFormat('zh-CN', {
    timeZone: state.timeZone,
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long',
  }).format(date);
}

function setCurrentVisibleMonth() {
  const parts = getZonedParts();
  state.visibleYear = parts.year;
  state.visibleMonth = parts.month - 1;
}

function renderTime() {
  const now = accurateNow();
  const parts = getZonedParts(now);

  els.ampm.textContent = parts.hour < 12 ? '上午' : '下午';
  els.time.textContent = `${pad(parts.hour)}:${pad(parts.minute)}:${pad(parts.second)}`;
  els.dateLine.textContent = formatChineseDate(now);
  renderSyncStatus();
}

function renderSyncStatus(message) {
  if (message) {
    els.syncLine.textContent = message;
    return;
  }

  if (state.syncing) {
    els.syncLine.textContent = '正在校准网络时间...';
    return;
  }

  if (!state.lastSyncAt) {
    els.syncLine.textContent = '未校时，暂用本机时间';
    return;
  }

  const minutes = Math.max(0, Math.floor((Date.now() - state.lastSyncAt) / 60000));
  const offsetSeconds = Math.round(state.timeOffsetMs / 1000);
  const offsetText = offsetSeconds === 0 ? '无偏差' : `已修正 ${offsetSeconds > 0 ? '+' : ''}${offsetSeconds} 秒`;
  els.syncLine.textContent = `${state.timeSource} 校时，${minutes} 分钟前，${offsetText}`;
}

function getWeekday(year, month, day) {
  const date = new Date(Date.UTC(year, month, day, 12, 0, 0));
  const weekday = new Intl.DateTimeFormat('en-US', {
    timeZone: state.timeZone,
    weekday: 'short',
  }).format(date);
  return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].indexOf(weekday);
}

function getDayNumber(year, month, day) {
  const date = new Date(Date.UTC(year, month, day, 12, 0, 0));
  const parts = new Intl.DateTimeFormat('zh-CN', {
    timeZone: state.timeZone,
    day: 'numeric',
  }).formatToParts(date);
  return Number(parts.find((part) => part.type === 'day')?.value || day);
}

function sameDateParts(dateParts, year, month, day) {
  return dateParts.year === year && dateParts.month === month + 1 && dateParts.day === day;
}

function renderCalendar() {
  const todayParts = getZonedParts();
  const mondayFirstOffset = (getWeekday(state.visibleYear, state.visibleMonth, 1) + 6) % 7;
  const gridStart = new Date(Date.UTC(state.visibleYear, state.visibleMonth, 1 - mondayFirstOffset, 12, 0, 0));

  els.monthTitle.textContent = `${state.visibleYear}年${state.visibleMonth + 1}月`;
  els.todayTip.textContent = sameDateParts(todayParts, state.visibleYear, state.visibleMonth, todayParts.day) ? '本月' : '点击月份两侧切换';
  els.calendarGrid.innerHTML = '';

  for (let index = 0; index < 42; index += 1) {
    const date = new Date(gridStart);
    date.setUTCDate(gridStart.getUTCDate() + index);

    const year = date.getUTCFullYear();
    const month = date.getUTCMonth();
    const dayValue = getDayNumber(year, month, date.getUTCDate());
    const weekday = getWeekday(year, month, date.getUTCDate());

    const day = document.createElement('div');
    day.className = 'day';
    day.textContent = dayValue;

    if (month !== state.visibleMonth) day.classList.add('outside');
    if (weekday === 0 || weekday === 6) day.classList.add('weekend');
    if (sameDateParts(todayParts, year, month, date.getUTCDate())) day.classList.add('today');

    els.calendarGrid.appendChild(day);
  }
}

function populateTimeZones() {
  const hasCurrentZone = TIME_ZONES.some((zone) => zone.value === state.timeZone);
  const zones = hasCurrentZone
    ? TIME_ZONES
    : [{ value: state.timeZone, label: `当前系统时区 ${state.timeZone}` }, ...TIME_ZONES];

  els.timeZone.innerHTML = '';
  for (const zone of zones) {
    const option = document.createElement('option');
    option.value = zone.value;
    option.textContent = zone.label;
    els.timeZone.appendChild(option);
  }
  els.timeZone.value = state.timeZone;
}

async function syncNetworkTime({ manual = false } = {}) {
  state.syncing = true;
  els.syncTimeBtn.disabled = true;
  renderSyncStatus(manual ? '正在手动校时...' : undefined);

  try {
    const result = await api.syncTime();
    state.timeOffsetMs = result.offsetMs;
    state.lastSyncAt = Date.now();
    state.timeSource = result.source;
    localStorage.setItem('timeOffsetMs', String(state.timeOffsetMs));
    localStorage.setItem('lastSyncAt', String(state.lastSyncAt));
    localStorage.setItem('timeSource', state.timeSource);
  } catch (error) {
    renderSyncStatus(`网络校时失败，暂用本机时间`);
  } finally {
    state.syncing = false;
    els.syncTimeBtn.disabled = false;
    renderTime();
    renderCalendar();
  }
}

function bindEvents() {
  els.prevMonth.addEventListener('click', () => {
    state.visibleMonth -= 1;
    if (state.visibleMonth < 0) {
      state.visibleMonth = 11;
      state.visibleYear -= 1;
    }
    renderCalendar();
  });

  els.nextMonth.addEventListener('click', () => {
    state.visibleMonth += 1;
    if (state.visibleMonth > 11) {
      state.visibleMonth = 0;
      state.visibleYear += 1;
    }
    renderCalendar();
  });

  els.timeZone.addEventListener('change', (event) => {
    state.timeZone = event.target.value;
    localStorage.setItem('timeZone', state.timeZone);
    setCurrentVisibleMonth();
    renderTime();
    renderCalendar();
  });

  els.syncTimeBtn.addEventListener('click', () => syncNetworkTime({ manual: true }));

  els.minimizeBtn.addEventListener('click', () => api.minimize());
  els.closeBtn.addEventListener('click', () => api.close());

  els.opacity.addEventListener('input', (event) => {
    const opacity = Number(event.target.value);
    localStorage.setItem('opacity', String(opacity));
    api.setOpacity(opacity);
  });

  els.alwaysOnTop.addEventListener('change', (event) => {
    const enabled = event.target.checked;
    localStorage.setItem('alwaysOnTop', String(enabled));
    api.setAlwaysOnTop(enabled);
  });

  els.openAtLogin.addEventListener('change', (event) => {
    api.setOpenAtLogin(event.target.checked);
  });

  api.onAlwaysOnTopChanged((enabled) => {
    els.alwaysOnTop.checked = enabled;
    localStorage.setItem('alwaysOnTop', String(enabled));
  });
}

async function restoreSettings() {
  const opacity = Number(localStorage.getItem('opacity') || '0.94');
  els.opacity.value = String(opacity);
  api.setOpacity(opacity);

  const alwaysOnTop = localStorage.getItem('alwaysOnTop') !== 'false';
  els.alwaysOnTop.checked = alwaysOnTop;
  api.setAlwaysOnTop(alwaysOnTop);

  const loginSettings = await api.getLoginSettings();
  els.openAtLogin.checked = Boolean(loginSettings.openAtLogin);
}

populateTimeZones();
setCurrentVisibleMonth();
bindEvents();
restoreSettings();
renderTime();
renderCalendar();
syncNetworkTime();
setInterval(renderTime, 1000);
setInterval(() => syncNetworkTime(), 30 * 60 * 1000);
