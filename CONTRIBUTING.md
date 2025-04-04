# 贡献指南

感谢您对AI Translator项目的关注！我们欢迎任何形式的贡献，包括但不限于：

- 报告问题
- 提交功能建议
- 改进文档
- 提交代码修复
- 添加新功能

## 如何贡献

### 1. 报告问题

如果您发现了一个bug或者有改进建议，请：

1. 在提交issue之前，请先搜索是否已经存在相关issue
2. 使用issue模板，提供详细的问题描述
3. 如果是bug，请提供复现步骤
4. 如果是功能建议，请说明使用场景

### 2. 提交代码

如果您想提交代码改进，请：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的改动 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个Pull Request

### 3. 代码规范

- 使用Python代码规范（PEP 8）
- 添加适当的注释
- 确保代码经过测试
- 保持代码简洁清晰

### 4. 文档改进

- 确保文档清晰易懂
- 添加必要的示例
- 更新README.md如果需要
- 添加适当的注释

## 开发环境设置

1. 克隆仓库
```bash
git clone https://github.com/aqm857886159/AI_Translator.git
cd AI_Translator
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 设置环境变量
创建 `.env` 文件并添加必要的环境变量

## 提交规范

提交信息应该清晰明了，建议使用以下格式：

```
<类型>: <描述>

[可选的详细描述]

[可选的关闭issue引用]
```

类型可以是：
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动

## 联系方式

如果您有任何问题，请通过以下方式联系我们：

- 提交issue
- 发送邮件到[您的邮箱]

感谢您的贡献！ 