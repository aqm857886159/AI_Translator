document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const downloadLink = document.getElementById('downloadLink');
    const uploadButton = document.getElementById('uploadButton');
    const resultContainer = document.getElementById('resultContainer');
    const progressSection = document.querySelector('.progress-section');
    let selectedFile = null;
    let isUploading = false;

    // 阻止表单的点击事件冒泡
    uploadForm.addEventListener('click', function(e) {
        e.stopPropagation();
    });

    // 处理文件选择
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            handleFileSelect(file);
        }
    });

    // 处理拖放
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('highlight');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('highlight');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    }

    function handleFileSelect(file) {
        // 检查文件类型
        const allowedTypes = ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown'];
        if (!allowedTypes.includes(file.type)) {
            alert('不支持的文件类型。请上传 PDF、Word、TXT 或 Markdown 文件。');
            fileInput.value = '';
            return;
        }
        
        // 检查文件大小（限制为 10MB）
        if (file.size > 10 * 1024 * 1024) {
            alert('文件大小超过限制。请上传小于 10MB 的文件。');
            fileInput.value = '';
            return;
        }

        selectedFile = file;
        const fileName = document.getElementById('file-name');
        if (fileName) {
            fileName.textContent = file.name;
        }
        uploadButton.disabled = false;
    }

    // 点击上传区域触发文件选择
    uploadArea.addEventListener('click', (e) => {
        if (e.target === uploadArea || e.target.closest('.upload-content')) {
            fileInput.click();
        }
    });

    // 表单提交处理
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        if (!selectedFile || isUploading) {
            return;
        }
        
        // 设置上传状态
        isUploading = true;
        
        // 创建 FormData 对象
        const formData = new FormData();
        formData.append('file', selectedFile);
        
        try {
            // 禁用上传按钮和文件输入
            uploadButton.disabled = true;
            fileInput.disabled = true;
            
            // 显示进度条
            progressSection.style.display = 'block';
            progressBar.value = 0;
            progressText.textContent = '准备上传...';
            
            // 发送文件
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // 处理服务器发送的事件
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const {value, done} = await reader.read();
                if (done) break;
                
                const text = decoder.decode(value);
                const lines = text.split('\n');
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.type === 'progress') {
                            // 更新进度条
                            const progress = Math.round(data.progress * 100);
                            progressBar.value = progress;
                            progressText.textContent = `翻译进度: ${progress}%`;
                        } else if (data.type === 'completed') {
                            // 显示下载链接
                            resultContainer.style.display = 'block';
                            downloadLink.href = data.download_url;
                            downloadLink.textContent = '下载翻译结果';
                            progressText.textContent = '翻译完成！';
                        } else if (data.type === 'error') {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        console.error('解析响应数据时出错:', e);
                        alert('处理响应数据时出错: ' + e.message);
                    }
                }
            }
        } catch (error) {
            console.error('上传过程中发生错误:', error);
            alert('上传失败: ' + error.message);
            progressText.textContent = '上传失败';
        } finally {
            // 恢复按钮状态
            uploadButton.disabled = false;
            fileInput.disabled = false;
            // 清除选择的文件
            selectedFile = null;
            fileInput.value = '';
            // 重置上传状态
            isUploading = false;
        }
    });
}); 