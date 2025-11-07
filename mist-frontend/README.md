-  Vue3 + Element Plus（按需导入）+ Pinia + Vue Router + Axios， Vue Router 路由管理

## 环境准备

1. **安装 Node.js**

   - 访问 [Node.js 官网](https://nodejs.org/) 下载并安装 LTS 版本（目前是 18.x）

   - 验证安装成功：

     ```
     node -v
     npm -v
     ```

2. **安装依赖**

```
npm install

# 安装 Vue Router
npm install vue-router@4

# 安装 pinia 状态管理
npm install pinia

# 安装 Axios（用于 API 调用）
npm install axios

npm install @vitejs/plugin-vue

npm install vite @vitejs/plugin-vue vue@^3.2.0 vue-router@^4.0.0 vuex@^4.0.0

如果按需导入插件或 Element Plus 未安装（首次搭建或换机），请补充安装：
npm i element-plus @element-plus/icons-vue unplugin-auto-import unplugin-vue-components
```

## 启动项目

```
# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

### 状态管理与接口封装

- 状态管理：`src/stores/task.js`，提供文件上传、处理提交与轮询查询、错误处理与重置。
- 接口封装：`src/utils/api.js`
  - `uploadImage(file)` → `POST /api/upload`
  - `startProcess({ mode, dehaze_strength, text_prompt })` → `POST /api/process`
  - `getTask(taskId)` → `GET /api/task/{task_id}`
  - `compareTasks(payload)` → `POST /api/compare`

### 基础模式使用流程

- 上传图像（拖拽/点击）→ 调整“去雾强度” → 点击“开始处理”。
- 前端提交 `/api/process` 后轮询 `/api/task/{task_id}`，完成后展示“去雾结果”与“目标检测结果”（含列表）。
