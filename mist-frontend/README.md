- 项目名称：清视智监-无人机图像去雾目标检测系统（前端）
- 架构说明：前后端分离；前端采用 Vue3 + Element Plus（按需导入）+ Pinia + Vue Router + Axios，使用 Vite 构建与开发。

采用典型的 Vue.js 项目结构，使用了 Vue Router 进行路由管理和 Pinia 作为状态管理工具。

## 环境准备

1. **安装 Node.js**

   - 访问 [Node.js 官网](https://nodejs.org/) 下载并安装 LTS 版本（目前是 18.x）

   - 验证安装成功：

     ```
     node -v
     npm -v
     ```

2. **安装代码编辑器**

   - 推荐使用 VS Code，并安装以下插件：
     - Volar（Vue 官方推荐插件）
     - Vue Language Features (Volar)
     - ESLint
     - Prettier

3. **安装依赖**

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

## 开发流程

1. **先构建基础页面**
   - 从 `BaseView.vue` 开始
   - 实现基本的路由跳转功能
2. **逐步实现各功能模块**
   - 按照剩余模块顺序开发
   - 每个模块对应相应的 store 模块和组件
3. **集成 API 调用**
   - 在 `utils/api.js` 中封装 axios 实例
   - 在 store 的 actions 中调用 API
4. **添加样式和组件**
   - 使用 `assets/styles/` 中的 CSS 文件
   - 开发可复用组件放在 `components/` 目录下

## 启动项目

```
# 启动开发服务器
npm run dev

# 构建生产版本
npm run build
```

### Element Plus 按需导入配置

- 已在 `vite.config.js` 集成按需导入插件与解析器：
  ```js
  import AutoImport from 'unplugin-auto-import/vite'
  import Components from 'unplugin-vue-components/vite'
  import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

  plugins: [
    vue(),
    AutoImport({ resolvers: [ElementPlusResolver({ importStyle: 'css' })] }),
    Components({ resolvers: [ElementPlusResolver({ importStyle: 'css' })] })
  ]
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
