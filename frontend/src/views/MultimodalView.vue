<template>
  <div class="multimodal-view">
    <el-container>
      <!-- 左侧控制面板 -->
      <el-aside width="320px" class="aside">
        <el-card shadow="hover" class="control-card">
          <template #header>
            <div class="card-header">
              <span>{{ panelTitle }}</span>
              <div class="header-actions">
                <el-button size="small" :type="controlStatus.type" class="status-btn" plain>{{ controlStatus.text }}</el-button>
              </div>
            </div>
          </template>
          <ImageUpload @change="onFileChange" />

          <!-- 文本引导区域 -->
          <div class="text-guidance-section mt16">
            <div class="section-title">
              <span class="indicator"></span>
              文本引导
            </div>
            <el-input
              v-model="store.textPrompt"
              type="textarea"
              :rows="2"
              placeholder="请使用英文描述你期望得到的图像"
              class="guidance-input"
            />
            <div class="prompt-pool mt12">
              <span class="pool-label">常用场景：</span>
              <div class="tags-wrapper">
                <el-tag
                  v-for="item in promptPool"
                  :key="item.en"
                  class="prompt-tag"
                  type="info"
                  effect="light"
                  @click="applyPrompt(item)"
                >
                  <div class="tag-content">
                    <div class="tag-en">{{ item.en }}</div>
                    <div class="tag-zh">{{ item.zh }}</div>
                  </div>
                </el-tag>
              </div>
            </div>
            <div class="guidance-hint">
              提示：输入特定关键词可增强检测效果
            </div>
          </div>

          <!-- 模型信息说明 -->
          <div class="model-info mt16">
            <div class="section-title">算法配置</div>
            <div class="info-item">
              <span class="label">去雾算法：</span>
              <el-tag size="small" effect="plain" type="success">CoA (CLIP-Oriented)</el-tag>
            </div>
            <div class="info-item mt-mini">
              <span class="label">目标检测：</span>
              <el-tag size="small" effect="plain" type="warning">Feature Fusion YOLO</el-tag>
            </div>
          </div>

          <div class="actions mt16">
            <el-space :size="8">
              <el-button type="primary" :icon="Upload" :disabled="!store.file || store.processing" :loading="store.processing" @click="onStartProcess">开始处理</el-button>
              <el-button :icon="Refresh" :disabled="store.processing" @click="store.reset">重置</el-button>
            </el-space>
          </div>
          <div class="status mt16" v-if="store.status !== 'idle'">
            <el-tag v-if="store.status==='uploaded'" type="success">已上传</el-tag>
            <el-tag v-else-if="store.status==='processing'" type="warning">处理中...</el-tag>
            <el-tag v-else-if="store.status==='done'" type="success">处理完成</el-tag>
            <el-alert v-else type="error" :closable="false" show-icon title="发生错误" :description="store.errorMessage || '处理失败'" />
          </div>
        </el-card>
      </el-aside>

      <!-- 中央可视化区域 -->
      <el-main>
        <el-row :gutter="12">
          <!-- 左栏：输入图像 + 基于文本的检测结果 -->
          <el-col :xs="24" :sm="24" :md="12">
            <el-card shadow="hover" class="visual-card">
              <template #header>
                <div class="card-header">
                  <span>输入图像</span>
                  <el-button size="small" :type="inputHeaderStatus.type" class="status-btn" plain>{{ inputHeaderStatus.text }}</el-button>
                </div>
              </template>
              <div v-if="store.inputPreviewUrl">
                <el-image :src="store.inputPreviewUrl" :preview-src-list="[store.inputPreviewUrl]" lazy fit="contain" class="img-box" />
              </div>
              <el-empty v-else description="等待上传..." class="empty-box" />
            </el-card>

            <!-- 标题变化：去雾结果 -> 基于文本引导的检测结果 -->
            <el-card shadow="hover" class="visual-card mt16 linked-result-card">
              <template #header>
                <div class="card-header">
                  <div class="linked-title">
                    <span class="indicator"></span>
                    <span>基于文本引导的去雾结果</span>
                  </div>
                  <el-button size="small" :type="dehazeHeaderStatus.type" class="status-btn" plain>{{ dehazeHeaderStatus.text }}</el-button>
                </div>
              </template>
              <template v-if="store.status==='processing'">
                <el-skeleton :rows="6" animated />
              </template>
              <template v-else>
                <div v-if="store.dehazeImageUrl">
                  <el-image :src="store.dehazeImageUrl" :preview-src-list="[store.dehazeImageUrl]" lazy fit="contain" class="img-box" />
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
                  <div v-if="store.detectImageUrl">
                    <el-image :src="store.detectImageUrl" :preview-src-list="[store.detectImageUrl]" lazy fit="contain" class="detect-img" />
                  </div>
                  <el-empty v-else description="暂无结果" class="empty-box detect-empty" />
                </template>
                <div class="detect-scroll-wrapper scroll-container">
                  <el-divider>检测类别统计</el-divider>
                  <div v-if="store.detectionSummary.length" class="detect-list">
                    <el-table :data="store.detectionSummary" size="small" style="width:100%" max-height="150">
                      <el-table-column prop="label" label="类别" width="160" />
                      <el-table-column prop="count" label="数量" />
                    </el-table>
                  </div>
                  <div v-else class="detect-list">
                    <el-text type="info">暂无统计数据</el-text>
                  </div>
                  <el-divider>检测结果列表</el-divider>
                  <div v-if="store.detections.length" class="detect-list">
                    <el-table :data="store.detections" size="small" style="width:100%" max-height="150">
                      <el-table-column prop="label" label="类别" width="160" />
                      <el-table-column prop="confidence" label="置信度" />
                    </el-table>
                  </div>
                  <div v-else class="detect-list">
                    <el-text type="info">暂无检测到的目标</el-text>
                  </div>
                </div>
              </div>
            </el-card>
          </el-col>
        </el-row>
      </el-main>
    </el-container>
    
    <HistoryPanel 
      ref="historyPanelRef" 
      mode="multimodal" 
      @select="onHistorySelect" 
    />
  </div>
  
</template>

<script setup>
import { onBeforeUnmount, computed, ref } from 'vue'
import { Upload, Refresh, Clock } from '@element-plus/icons-vue'
import { useTaskStore } from '@/stores/task'
import ImageUpload from '@/components/common/ImageUpload.vue'
import HistoryPanel from '@/components/common/HistoryPanel.vue'

const store = useTaskStore()
const historyPanelRef = ref(null)

const openHistory = () => {
  historyPanelRef.value?.open()
}

const onHistorySelect = (item) => {
  store.loadHistory(item)
  if (item.params && item.params.text_prompt) {
      store.textPrompt = item.params.text_prompt
  }
}

defineExpose({ openHistory })

// 当前模式：多模态模式
const currentMode = 'fusion'
const panelTitle = '多模态控制面板'

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


const promptPool = [
  { zh: '清晰锐利', en: 'Clear, sharp photo without haze' },
  { zh: '明亮干净', en: 'bright and clean' },
  { zh: '无雾视野', en: 'Free from haze or fog' },
  { zh: '高分辨率', en: 'High resolution and sharp details' },
  { zh: '高对比度，清晰的细节', en: 'High contrast, clear details' },
  { zh: '水晶般清晰', en: 'Crystal clear image' },
  { zh: '色彩鲜艳', en: 'vivid colors' }
]

const applyPrompt = (item) => {
  store.textPrompt = item.en
}

const onFileChange = (file) => {
  const raw = file?.raw || null
  if (raw) {
    store.setFile(raw)
    // 可选择立即上传到后端
    store.upload().catch(() => {})
  }
}

const onStartProcess = () => {
  store.startProcessTask({ mode: currentMode, textPrompt: store.textPrompt })
}
</script>

<style scoped>
.multimodal-view {
  padding: 12px;
  --panel-gap: 16px;
  --card-h: 320px;
  --empty-size: 120px;
  --radius: 12px;
  --shadow: 0 4px 16px rgba(0,0,0,0.08);
  --shadow-hover: 0 10px 24px rgba(0,0,0,0.12);
  --accent-color: #722ed1; /* 紫色系代表多模态融合 */
}
.aside {
  padding-right: 12px;
  position: sticky;
  top: 12px;
  align-self: flex-start;
}
.mt16 { margin-top: 16px; }
.upload-area { width: 100%; }
.mt12 { margin-top: 12px; }
.mt-mini { margin-top: 8px; }
.section-title { font-weight: 600; margin-bottom: 8px; font-size: var(--font-subtitle); }
.info-item { display: flex; align-items: center; justify-content: space-between; font-size: 13px; color: var(--el-text-color-regular); }

.prompt-pool { display: flex; flex-direction: column; gap: 8px; }
.pool-label { font-size: 12px; color: var(--el-text-color-secondary); }
.tags-wrapper { display: flex; flex-wrap: wrap; gap: 8px; }
.prompt-tag { cursor: pointer; transition: all 0.2s; height: auto; padding: 4px 8px; }
.prompt-tag:hover { color: var(--accent-color); border-color: var(--accent-color); background-color: rgba(114, 46, 209, 0.1); }
.tag-content { display: flex; flex-direction: column; align-items: center; line-height: 1.2; }
.tag-en { font-size: 12px; font-weight: 500; }
.tag-zh { font-size: 10px; opacity: 0.8; }

/* 视觉关联：指示条 */
.indicator {
  display: inline-block;
  width: 4px;
  height: 16px;
  background-color: var(--accent-color);
  margin-right: 8px;
  border-radius: 2px;
  vertical-align: middle;
}
.text-guidance-section .section-title {
  display: flex;
  align-items: center;
}
.linked-title {
  display: flex;
  align-items: center;
}
.guidance-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
/* 输入框聚焦时的样式关联 */
.guidance-input :deep(.el-textarea__inner:focus) {
  border-color: var(--accent-color);
  box-shadow: 0 0 0 1px var(--accent-color) inset;
}
/* 关联的结果卡片标题颜色 */
/* .linked-result-card .card-header span {
  color: var(--accent-color);
} */

/* 高度协调与图片容器设置 */
.control-card { min-height: calc(var(--card-h) * 2 + var(--panel-gap)); display:flex; flex-direction:column; }
.visual-card { height: var(--card-h); display:flex; flex-direction:column; }
.img-box { width: 100%; height: 100%; }
.detect-card { height: calc(var(--card-h) * 2 + var(--panel-gap)); display:flex; flex-direction:column; }
.detect-content { display:flex; flex-direction:column; flex:1; overflow: hidden; }
.detect-scroll-wrapper { flex: 1; overflow-y: auto; padding-right: 4px; }
.detect-img { flex: 0 0 auto; height: 180px; width: 100%; display: block; }
.detect-empty { flex: 0 0 auto; height: 180px; }
.detect-list { max-height: none; overflow: visible; }
.empty-box { height: 100%; display:flex; align-items:center; justify-content:center; }
.empty-box :deep(.el-empty__image) { width: var(--empty-size) !important; }
.empty-box :deep(.el-empty__description) { font-size: var(--font-body); }

/* 统一圆角与悬浮阴影美化 */
:deep(.el-card) { border-radius: var(--radius); overflow: hidden; transition: box-shadow .25s ease, transform .2s ease; }
.control-card, .visual-card, .detect-card { box-shadow: var(--shadow); }
.control-card:hover, .visual-card:hover, .detect-card:hover { box-shadow: var(--shadow-hover); transform: translateY(-2px); }
.card-header { display:flex; align-items:center; justify-content:space-between; }
.header-actions { display:flex; align-items:center; gap: 8px; }
.status-btn { pointer-events: none; border-radius: 8px; }
.img-box { border-radius: 10px; overflow: hidden; }
/* el-image 内部 img 圆角 */
.img-box :deep(.el-image__inner) { border-radius: 10px; }
.card-header span { font-weight: 600; font-size: var(--font-title); }
.param-item .label { font-weight: 500; font-size: var(--font-subtitle); color: var(--el-color-info); }
.detect-list { font-size: var(--font-body); }

/* 自定义滚动条样式 */
.scroll-container :deep(.el-table__body-wrapper::-webkit-scrollbar) {
  width: 6px;
  height: 6px;
}
.scroll-container :deep(.el-table__body-wrapper::-webkit-scrollbar-thumb) {
  background-color: #dcdfe6;
  border-radius: 3px;
}
.scroll-container :deep(.el-table__body-wrapper::-webkit-scrollbar-track) {
  background-color: #f5f7fa;
}
.scroll-container :deep(.el-table__body-wrapper:hover::-webkit-scrollbar-thumb) {
  background-color: #c0c4cc;
}

@media (max-width: 768px) {
  .multimodal-view { --card-h: 220px; --empty-size: 90px; }
  .multimodal-view .el-container { flex-direction: column; }
  .multimodal-view .el-container .el-aside { width: 100% !important; margin-bottom: 12px; }
  .aside { padding-right: 0; position: static; }
}
</style>
