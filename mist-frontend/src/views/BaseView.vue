<template>
  <div class="base-view">
    <el-container>
      <!-- 左侧控制面板 -->
      <el-aside width="320px" class="aside">
        <el-card shadow="hover" class="control-card">
          <template #header>
            <div class="card-header">
              <span>{{ panelTitle }}</span>
              <el-button size="small" :type="controlStatus.type" class="status-btn" plain>{{ controlStatus.text }}</el-button>
            </div>
          </template>
          <ImageUpload @change="onFileChange" />

          <div class="param-item mt16">
            <div class="label">去雾强度：{{ (store.dehazeStrength).toFixed(2) }}</div>
            <el-slider v-model="store.dehazeStrength" :min="0" :max="1" :step="0.05" />
          </div>

          <div class="actions mt16">
            <el-button type="primary" :disabled="!store.file || store.processing" @click="onStartProcess">
              开始处理
            </el-button>
            <el-button class="ml8" :disabled="store.processing" @click="store.reset">重置</el-button>
          </div>
          <div class="status mt8" v-if="store.status !== 'idle'">
            <el-tag v-if="store.status==='uploaded'" type="success">已上传</el-tag>
            <el-tag v-else-if="store.status==='processing'" type="warning">处理中...</el-tag>
            <el-tag v-else-if="store.status==='done'" type="success">处理完成</el-tag>
            <el-tag v-else type="danger">{{ store.errorMessage || '发生错误' }}</el-tag>
          </div>
        </el-card>
      </el-aside>

      <!-- 中央可视化区域 -->
      <el-main>
        <el-row :gutter="12">
          <!-- 左栏：输入图像 + 去雾结果 -->
          <el-col :xs="24" :sm="24" :md="12">
            <el-card shadow="hover" class="visual-card">
              <template #header>
                <div class="card-header">
                  <span>输入图像</span>
                  <el-button size="small" :type="inputHeaderStatus.type" class="status-btn" plain>{{ inputHeaderStatus.text }}</el-button>
                </div>
              </template>
              <div v-if="store.inputPreviewUrl">
                <el-image :src="store.inputPreviewUrl" fit="contain" class="img-box" />
              </div>
              <el-empty v-else description="等待上传..." class="empty-box" />
            </el-card>

            <el-card shadow="hover" class="visual-card mt16">
              <template #header>
                <div class="card-header">
                  <span>去雾结果</span>
                  <el-button size="small" :type="dehazeHeaderStatus.type" class="status-btn" plain>{{ dehazeHeaderStatus.text }}</el-button>
                </div>
              </template>
              <template v-if="store.status==='processing'">
                <el-skeleton :rows="6" animated />
              </template>
              <template v-else>
                <div v-if="store.file">
                  <el-image src="/testniqe.jpg" fit="contain" class="img-box" />
                </div>
                <el-empty v-else description="暂无结果" class="empty-box" />
              </template>
            </el-card>
          </el-col>

          <!-- 右栏：检测结果 -->
          <el-col :xs="24" :sm="24" :md="12">
            <el-card shadow="hover" class="detect-card">
              <template #header>
                <div class="card-header">
                  <span>目标检测结果</span>
                  <el-button size="small" :type="detectHeaderStatus.type" class="status-btn" plain>{{ detectHeaderStatus.text }}</el-button>
                </div>
              </template>
              <div class="detect-content">
                <template v-if="store.status==='processing'">
                  <el-skeleton :rows="8" animated />
                </template>
                <template v-else>
                  <div v-if="store.file">
                    <el-image src="/biaozhu.png" fit="contain" class="detect-img" />
                  </div>
                  <el-empty v-else description="暂无结果" class="empty-box detect-empty" />
                </template>
                <el-divider>检测结果列表</el-divider>
                <div v-if="hardDetections.length" class="detect-list">
                  <el-table :data="hardDetections" size="small" style="width:100%">
                    <el-table-column prop="label" label="类别" width="120" />
                    <el-table-column prop="count" label="数量" />
                  </el-table>
                </div>
                <div v-else class="detect-list">
                  <el-text type="info">暂无检测到的目标</el-text>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
  </div>
  
</template>

<script setup>
import { onBeforeUnmount, computed } from 'vue'
import { useTaskStore } from '@/stores/task'
import ImageUpload from '@/components/common/ImageUpload.vue'

const store = useTaskStore()

// 当前模式：基础模式；后续可根据路由或全局状态注入
const currentMode = 'basic'
const panelTitle = computed(() => {
  const map = {
    basic: '基础模式控制面板',
    multimodal: '多模态模式控制面板',
    compare: '对比模式控制面板'
  }
  return map[currentMode] || '控制面板'
})

// 标题右侧状态按钮文案与样式
const controlStatus = computed(() => {
  const s = store.status
  if (s === 'uploaded') return { text: '已上传', type: 'success' }
  if (s === 'processing') return { text: '处理中...', type: 'warning' }
  if (s === 'done') return { text: '处理完成', type: 'success' }
  if (s === 'error') return { text: '发生错误', type: 'danger' }
  return { text: '等待处理...', type: 'info' }
})

const inputHeaderStatus = computed(() => (store.file
  ? { text: '已上传', type: 'success' }
  : { text: '未上传', type: 'info' }
))

const dehazeHeaderStatus = computed(() => {
  if (store.status === 'processing') return { text: '处理中...', type: 'warning' }
  if (store.dehazeImageUrl) return { text: '已生成', type: 'success' }
  return { text: '暂无结果', type: 'info' }
})

const detectHeaderStatus = computed(() => {
  if (store.status === 'processing') return { text: '处理中...', type: 'warning' }
  if (store.detectImageUrl) return { text: '已生成', type: 'success' }
  return { text: '暂无结果', type: 'info' }
})

const hardDetections = computed(() => (store.file ? [
  { label: 'person', confidence: '-', count: 3 },
  { label: 'motorcycle', confidence: '-', count: 13 },
  { label: 'car', confidence: '-', count: 3 },
] : []))

const onFileChange = (file) => {
  const raw = file?.raw || null
  if (raw) {
    store.setFile(raw)
    // 可选择立即上传到后端
    store.upload().catch(() => {})
  }
}

const onStartProcess = () => {
  store.startProcessTask({ mode: currentMode })
}

onBeforeUnmount(() => {
  store.stopPolling()
})
</script>

<style scoped>
.base-view {
  padding: 12px;
  --panel-gap: 16px;
  --card-h: 320px;
  --empty-size: 120px;
  --radius: 12px;
  --shadow: 0 4px 16px rgba(0,0,0,0.08);
  --shadow-hover: 0 10px 24px rgba(0,0,0,0.12);
}
.aside {
  padding-right: 12px;
}
.mt16 { margin-top: 16px; }
.ml8 { margin-left: 8px; }
.upload-area { width: 100%; }
.mt12 { margin-top: 12px; }
/* 高度协调与图片容器设置 */
.control-card { height: calc(var(--card-h) * 2 + var(--panel-gap)); display:flex; flex-direction:column; }
.visual-card { height: var(--card-h); display:flex; flex-direction:column; }
.img-box { width: 100%; height: 100%; }
.detect-card { height: calc(var(--card-h) * 2 + var(--panel-gap)); display:flex; flex-direction:column; }
.detect-content { display:flex; flex-direction:column; flex:1; }
.detect-img { flex: 1; }
.detect-empty { flex: 1; }
.detect-list { max-height: 200px; overflow: auto; }
.empty-box { height: 100%; display:flex; align-items:center; justify-content:center; }
.empty-box :deep(.el-empty__image) { width: var(--empty-size) !important; }
.empty-box :deep(.el-empty__description) { font-size: var(--font-body); }

/* 统一圆角与悬浮阴影美化 */
:deep(.el-card) { border-radius: var(--radius); overflow: hidden; transition: box-shadow .25s ease, transform .2s ease; }
.control-card, .visual-card, .detect-card { box-shadow: var(--shadow); }
.control-card:hover, .visual-card:hover, .detect-card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); }
.card-header { display:flex; align-items:center; justify-content:space-between; }
.status-btn { pointer-events: none; border-radius: 8px; }
.img-box { border-radius: 10px; overflow: hidden; }
/* el-image 内部 img 圆角 */
.img-box :deep(.el-image__inner) { border-radius: 10px; }
.card-header { display:flex; align-items:center; justify-content:space-between; }
.status-btn { pointer-events: none; }
.card-header span { font-weight: 600; font-size: var(--font-title); }
.param-item .label { font-weight: 500; font-size: var(--font-subtitle); color: var(--el-color-info); }
.detect-list { font-size: var(--font-body); }
@media (max-width: 768px) {
  .base-view { --card-h: 220px; --empty-size: 90px; }
  .base-view .el-container { flex-direction: column; }
  .base-view .el-container .el-aside { width: 100% !important; margin-bottom: 12px; }
  .aside { padding-right: 0; }
}
</style>