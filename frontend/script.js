// 全局变量
let wavesurfer;
let wordElements = []; // 存储 { start, end, el, text }
let currentTaskId = null;
let pollingInterval = null;

document.addEventListener('DOMContentLoaded', () => {
    initWavesurfer();
    setupUpload();
});

// 1. 初始化波形图
function initWavesurfer() {
    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#BDC1C6',
        progressColor: '#4285F4',
        cursorColor: '#4285F4',
        barWidth: 2,
        barRadius: 3,
        height: 80,
    });

    // 播放按钮
    const playBtn = document.getElementById('play-btn');
    playBtn.addEventListener('click', () => {
        wavesurfer.playPause();
        const icon = playBtn.querySelector('.material-icons');
        icon.textContent = wavesurfer.isPlaying() ? 'pause' : 'play_arrow';
    });

    wavesurfer.on('play', () => playBtn.querySelector('.material-icons').textContent = 'pause');
    wavesurfer.on('pause', () => playBtn.querySelector('.material-icons').textContent = 'play_arrow');

    // 核心：时间更新时高亮文字
    wavesurfer.on('timeupdate', (currentTime) => {
        highlightWords(currentTime);
        document.getElementById('current-time').textContent = formatTime(currentTime);
    });

    wavesurfer.on('ready', () => {
        document.getElementById('total-time').textContent = formatTime(wavesurfer.getDuration());
    });
}

// 2. 上传逻辑
function setupUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadLoading = document.getElementById('upload-loading');

    // 绑定点击事件
    dropZone.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        // UI 状态更新
        uploadLoading.classList.remove('hidden');
        
        // 立即加载本地音频用于播放 (无需等待服务器)
        const localAudioUrl = URL.createObjectURL(file);
        wavesurfer.load(localAudioUrl);

        // 发送给后端
        const formData = new FormData();
        formData.append('file', file); // 确保字段名为 'file'

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.status === 'ok') {
                currentTaskId = data.task_id;
                
                // A. 立即渲染 ASR 结果 (Stage 1)
                renderASR(data.asr);
                
                // 显示播放界面
                document.getElementById('input-card').classList.add('hidden');
                document.getElementById('stage').classList.remove('hidden');

                // B. 开始轮询分析结果 (Stage 2 & 3)
                startPolling(currentTaskId);
            } else {
                alert('上传失败: ' + data.error);
            }
        } catch (err) {
            console.error(err);
            alert('网络错误或后端启动失败');
        } finally {
            uploadLoading.classList.add('hidden');
        }
    });
}

// 渲染 ASR 文本 (不仅是显示，还建立了时间索引)
function renderASR(asrData) {
    const container = document.getElementById('transcript-box');
    container.innerHTML = '';
    wordElements = [];

    // 根据你的 json 结构: text, segments -> words
    if (asrData.segments) {
        asrData.segments.forEach(seg => {
            const items = seg.words || [{ word: seg.text, start: seg.start, end: seg.end }];
            
            items.forEach(w => {
                const span = document.createElement('span');
                span.textContent = w.word;
                span.className = 'word';
                
                // 点击跳跃播放
                span.onclick = () => wavesurfer.setTime(w.start);

                container.appendChild(span);

                wordElements.push({
                    start: w.start,
                    end: w.end,
                    el: span,
                    text: w.word.trim()
                });
            });
        });
    }
}

// 轮询后端状态
function startPolling(taskId) {
    pollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`/status/${taskId}`);
            const status = await res.json();

            if (status.done) {
                clearInterval(pollingInterval);
                fetchResults(taskId);
            } else if (status.error) {
                clearInterval(pollingInterval);
                document.getElementById('status-badge').textContent = "分析失败";
                document.getElementById('status-badge').className = "badge error";
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 1000); // 每秒查一次
}

// 获取最终结果并应用特效
async function fetchResults(taskId) {
    const res = await fetch(`/result/${taskId}`);
    const data = await res.json();

    // 1. 提取关键词列表
    const keywordSet = new Set(data.keywords.map(k => k.keyword));

    // 2. 应用到现有的 DOM 上
    applyKeywords(keywordSet);

    // 3. 显示词云
    const wcImg = document.getElementById('wordcloud-img');
    // 加个时间戳防止缓存
    wcImg.src = `${data.wordcloud}?t=${new Date().getTime()}`;
    wcImg.onload = () => {
        wcImg.classList.remove('hidden');
        document.getElementById('wc-loading').classList.add('hidden');
        document.getElementById('wc-placeholder').classList.add('hidden');
    };

    // 4. 更新状态徽章
    const badge = document.getElementById('status-badge');
    badge.textContent = "分析完成";
    badge.className = "badge done";
}

// 把关键词 CSS 类贴上去
function applyKeywords(keywordSet) {
    wordElements.forEach(item => {
        if (keywordSet.has(item.text)) {
            item.el.classList.add('keyword-hit');
        }
    });
}

// 高亮逻辑 (每一帧调用)
function highlightWords(time) {
    wordElements.forEach(item => {
        if (time >= item.start && time <= item.end) {
            item.el.classList.add('active');
            item.el.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'nearest' });
        } else {
            // 移除高亮，读过的文字恢复暗色
            item.el.classList.remove('active');
        }
    });
}

// 辅助：时间格式化
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}