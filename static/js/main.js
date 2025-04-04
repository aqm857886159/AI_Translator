document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const progressContainer = document.getElementById('progressContainer');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const resultContainer = document.getElementById('resultContainer');
    const downloadLink = document.getElementById('downloadLink');

    // 拖拽上传
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#007AFF';
        uploadArea.style.background = 'rgba(0, 122, 255, 0.05)';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#D2D2D7';
        uploadArea.style.background = 'white';
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#D2D2D7';
        uploadArea.style.background = 'white';
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // 点击上传
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    function handleFile(file) {
        // 检查文件类型
        const allowedTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown'];
        if (!allowedTypes.includes(file.type)) {
            alert('不支持的文件类型！请上传PDF、Word、TXT或Markdown文件。');
            return;
        }

        // 显示进度条
        progressContainer.style.display = 'block';
        resultContainer.style.display = 'none';

        // 创建FormData对象
        const formData = new FormData();
        formData.append('file', file);

        // 发送文件到服务器
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // 更新进度条
            progressBar.style.width = '100%';
            progressText.textContent = '翻译完成！';
            
            // 显示下载链接
            resultContainer.style.display = 'block';
            downloadLink.href = data.download_url;
        })
        .catch(error => {
            console.error('Error:', error);
            progressText.textContent = `错误: ${error.message}`;
            progressBar.style.background = '#FF3B30';
        });
    }
}); 