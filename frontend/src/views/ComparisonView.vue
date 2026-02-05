<template>
  <div class="comparison-view">
    <el-container>
      <!-- 左侧控制面板 -->
      <el-aside width="320px" class="aside">
        <el-card shadow="hover" class="control-card">
          <template #header>
            <div class="card-header">
              <span>对比模式控制面板</span>
            </div>
          </template>
          
          <div class="section-title mt16">
            <span class="indicator"></span>
            1. 图像上传
          </div>
          <ImageUpload @change="onFileChange" />

          <div class="section-title mt16">
            <span class="indicator"></span>
            2. 文本引导
          </div>
          <el-input
            v-model="textPrompt"
            type="textarea"
            :rows="3"
            placeholder="例如：'检测车辆'，'寻找远处的船'..."
            class="guidance-input"
          />
          <div class="guidance-hint">
            提示：仅针对方案二（多模态引导）生效
          </div>

          <div class="actions mt16">
            <el-button 
              type="primary" 
              class="w-100" 
              :loading="processing" 
              :disabled="!store.file"
              @click="handleCompare"
            >
              开始对比分析
            </el-button>
          </div>
        </el-card>
      </el-aside>

      <!-- 中央可视化区域 -->
      <el-main class="main-area">
        <div class="comparison-container">
          
          <!-- Scheme 1: Basic -->
          <el-card shadow="hover" class="scheme-card mb-4">
            <template #header>
              <div class="scheme-header">
                <span class="badge basic">方案一</span>
                <span class="title">基础去雾检测 (Original -> Dehaze -> Detect)</span>
              </div>
            </template>
            <div class="pipeline-row">
              <div class="image-box-wrapper">
                <span class="img-label">原图</span>
                <div class="image-box">
                  <el-image :src="store.inputPreviewUrl" fit="contain" class="pipeline-img">
                    <template #error><div class="placeholder">等待上传</div></template>
                  </el-image>
                </div>
              </div>
              <div class="arrow">→</div>
              <div class="image-box-wrapper">
                <span class="img-label">去雾结果</span>
                <div class="image-box">
                  <el-image :src="basicResult.dehazed" fit="contain" class="pipeline-img" :preview-src-list="[basicResult.dehazed]">
                    <template #error><div class="placeholder">待处理</div></template>
                  </el-image>
                </div>
              </div>
              <div class="arrow">→</div>
              <div class="image-box-wrapper">
                <span class="img-label">检测结果</span>
                <div class="image-box">
                  <el-image :src="basicResult.detected" fit="contain" class="pipeline-img" :preview-src-list="[basicResult.detected]">
                    <template #error><div class="placeholder">待处理</div></template>
                  </el-image>
                </div>
              </div>
              
              <!-- Key Indicators -->
              <div class="metrics-panel">
                <div class="metric-item">
                  <span class="label">目标总数</span>
                  <span class="value">{{ basicResult.count || 0 }}</span>
                </div>
                <div class="metric-item">
                  <span class="label">平均置信度</span>
                  <span class="value">{{ basicResult.avgConf || '0%' }}</span>
                </div>
              </div>
            </div>
          </el-card>

          <!-- Scheme 2: Multimodal -->
          <el-card shadow="hover" class="scheme-card mb-4">
            <template #header>
              <div class="scheme-header">
                <span class="badge multimodal">方案二</span>
                <span class="title">多模态引导检测 (Original -> Text -> Dehaze -> Detect)</span>
              </div>
            </template>
            <div class="pipeline-row">
              <div class="image-box-wrapper">
                <span class="img-label">原图</span>
                <div class="image-box">
                  <el-image :src="store.inputPreviewUrl" fit="contain" class="pipeline-img">
                     <template #error><div class="placeholder">等待上传</div></template>
                  </el-image>
                </div>
              </div>
              <div class="arrow with-text">
                <span>+ 文本引导</span>
                <div class="text-hint" v-if="textPrompt">"{{ textPrompt }}"</div>
                <div class="arrow-icon">→</div>
              </div>
              <div class="image-box-wrapper">
                <span class="img-label">引导去雾</span>
                <div class="image-box">
                  <el-image :src="multimodalResult.dehazed" fit="contain" class="pipeline-img" :preview-src-list="[multimodalResult.dehazed]">
                    <template #error><div class="placeholder">待处理</div></template>
                  </el-image>
                </div>
              </div>
              <div class="arrow">→</div>
              <div class="image-box-wrapper">
                <span class="img-label">联合检测</span>
                <div class="image-box">
                  <el-image :src="multimodalResult.detected" fit="contain" class="pipeline-img" :preview-src-list="[multimodalResult.detected]">
                    <template #error><div class="placeholder">待处理</div></template>
                  </el-image>
                </div>
              </div>

              <!-- Key Indicators -->
              <div class="metrics-panel">
                <div class="metric-item">
                  <span class="label">目标总数</span>
                  <span class="value highlight">{{ multimodalResult.count || 0 }}</span>
                </div>
                <div class="metric-item">
                  <span class="label">平均置信度</span>
                  <span class="value highlight">{{ multimodalResult.avgConf || '0%' }}</span>
                </div>
              </div>
            </div>
          </el-card>

          <!-- Quantitative Comparison Card -->
          <el-card shadow="hover" class="scheme-card mb-4">
            <template #header>
              <div class="scheme-header">
                <span class="badge comparison">对比总结</span>
                <span class="title">量化指标对比</span>
              </div>
            </template>
            <el-table :data="comparisonData" border style="width: 100%">
              <el-table-column prop="metric" label="指标" width="180" />
              <el-table-column label="方案一 (基础去雾)" align="center">
                <template #default="scope">
                  <span :class="{ 'winner': scope.row.basic > scope.row.multimodal }">{{ scope.row.basic }}</span>
                </template>
              </el-table-column>
              <el-table-column label="方案二 (多模态引导)" align="center">
                <template #default="scope">
                  <span :class="{ 'winner': scope.row.multimodal > scope.row.basic }">{{ scope.row.multimodal }}</span>
                </template>
              </el-table-column>
              <el-table-column label="提升幅度" align="center">
                <template #default="scope">
                  <el-tag :type="scope.row.improvement > 0 ? 'success' : 'info'">
                    {{ scope.row.improvement > 0 ? '+' : '' }}{{ scope.row.improvement }}%
                  </el-tag>
                </template>
              </el-table-column>
            </el-table>
          </el-card>

          <!-- Export Button Area -->
          <div class="export-section">
             <el-button type="primary" size="large" class="export-btn" @click="exportReport">
              <el-icon><Document /></el-icon>
              生成对比报告 (PDF)
            </el-button>
          </div>

        </div>
      </el-main>
    </el-container>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useTaskStore } from '@/stores/task'
import ImageUpload from '@/components/common/ImageUpload.vue'
import { Document } from '@element-plus/icons-vue'
import { startProcess, getDetections } from '@/utils/api'
import { ElMessage } from 'element-plus'

const store = useTaskStore()
const textPrompt = ref('')
const processing = ref(false)
const comparisonDone = ref(false)

const basicResult = reactive({
  dehazed: '',
  detected: '',
  count: 0,
  avgConf: '0%'
})

const multimodalResult = reactive({
  dehazed: '',
  detected: '',
  count: 0,
  avgConf: '0%'
})

const comparisonData = computed(() => {
  const basicCount = basicResult.count
  const multiCount = multimodalResult.count
  const basicConf = parseFloat(basicResult.avgConf) || 0
  const multiConf = parseFloat(multimodalResult.avgConf) || 0

  const countImp = basicCount === 0 ? (multiCount > 0 ? 100 : 0) : ((multiCount - basicCount) / basicCount * 100).toFixed(1)
  const confImp = basicConf === 0 ? (multiConf > 0 ? 100 : 0) : ((multiConf - basicConf) / basicConf * 100).toFixed(1)

  return [
    { metric: '目标检测数量', basic: basicCount, multimodal: multiCount, improvement: countImp },
    { metric: '平均置信度 (%)', basic: basicConf, multimodal: multiConf, improvement: confImp }
  ]
})

const onFileChange = (file) => {
  if (file?.raw) {
    store.setFile(file.raw)
    // Reset results
    resetResults()
  }
}

const resetResults = () => {
  comparisonDone.value = false
  Object.assign(basicResult, { dehazed: '', detected: '', count: 0, avgConf: '0%' })
  Object.assign(multimodalResult, { dehazed: '', detected: '', count: 0, avgConf: '0%' })
}

const parseDetections = async (timestamp, basename) => {
  try {
    const resp = await getDetections(timestamp, basename)
    const list = resp?.data?.list || []
    const count = list.length
    const avg = count > 0 
      ? (list.reduce((acc, cur) => acc + (parseFloat(cur.confidence) || 0), 0) / count * 100).toFixed(1) + '%'
      : '0%'
    return { count, avgConf: avg }
  } catch (e) {
    return { count: 0, avgConf: '0%' }
  }
}

const handleCompare = async () => {
  if (!store.file) return
  processing.value = true
  comparisonDone.value = false
  
  try {
    // 1. Upload and get Basic Result (assuming upload returns basic processing)
    // Using store.upload() which calls api.uploadImage
    const uploadData = await store.upload()
    
    // Fill Basic Results
    basicResult.dehazed = uploadData?.dehazed || ''
    basicResult.detected = uploadData?.detected || ''
    
    // Parse metrics for Basic
    if (uploadData?.timestamp && uploadData?.original) {
        const m = uploadData.original.match(/1_original_(.+)\.jpg$/)
        const basename = m ? m[1] : ''
        if (basename) {
            const metrics = await parseDetections(uploadData.timestamp, basename)
            basicResult.count = metrics.count
            basicResult.avgConf = metrics.avgConf
        }
    }

    // 2. Call Multimodal Process
    // We need to ensure the backend knows which image we are talking about.
    // Since startProcess doesn't take an ID, we assume it uses the last uploaded file in the session.
    // This is a risky assumption but consistent with the current architecture observed.
    const multiResp = await startProcess({
        mode: 'multimodal',
        text_prompt: textPrompt.value,
        dehaze_strength: store.dehazeStrength
    })
    
    const multiData = multiResp.data
    multimodalResult.dehazed = multiData?.dehazed || ''
    multimodalResult.detected = multiData?.detected || ''

    // Parse metrics for Multimodal
    // Assuming the backend returns similar structure
    if (multiData?.timestamp && multiData?.original) {
        const m = multiData.original.match(/1_original_(.+)\.jpg$/)
        const basename = m ? m[1] : ''
         if (basename) {
            const metrics = await parseDetections(multiData.timestamp, basename)
            multimodalResult.count = metrics.count
            multimodalResult.avgConf = metrics.avgConf
        }
    }

    comparisonDone.value = true
    ElMessage.success('对比分析完成')

  } catch (err) {
    console.error(err)
    ElMessage.error('处理失败: ' + (err.message || '未知错误'))
  } finally {
    processing.value = false
  }
}

const exportReport = () => {
  window.print()
}
</script>

<style scoped>
.comparison-view {
  padding: 12px;
  --panel-gap: 16px;
  --radius: 12px;
  --shadow: 0 4px 16px rgba(0,0,0,0.08);
  --shadow-hover: 0 10px 24px rgba(0,0,0,0.12);
  --accent-color: #722ed1;
}

.aside {
  padding-right: 12px;
  position: sticky;
  top: 12px;
  align-self: flex-start;
}
.control-card { 
  border: none; 
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  transition: box-shadow .25s ease, transform .2s ease;
}
.control-card:hover {
  box-shadow: var(--shadow-hover);
}

.card-header { font-weight: bold; font-size: 16px; }
.mt16 { margin-top: 16px; }
.mb-4 { margin-bottom: 16px; }
.w-100 { width: 100%; }

/* Reuse indicator style */
.indicator {
  display: inline-block;
  width: 4px;
  height: 16px;
  background-color: var(--accent-color);
  margin-right: 8px;
  border-radius: 2px;
  vertical-align: middle;
}
.section-title {
  font-weight: 600;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
}
.guidance-hint {
  font-size: 12px;
  color: #909399;
  margin-top: 4px;
}
.guidance-input :deep(.el-textarea__inner:focus) {
  border-color: var(--accent-color);
  box-shadow: 0 0 0 1px var(--accent-color) inset;
}

.main-area {
  padding: 0 0 20px 12px;
  overflow-y: auto;
}
.comparison-container {
  max-width: 1200px;
  margin: 0 auto;
}
.scheme-card {
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  transition: box-shadow .25s ease, transform .2s ease;
}
.scheme-card:hover {
  box-shadow: var(--shadow-hover);
}

.scheme-header {
  display: flex;
  align-items: center;
}
.badge {
  padding: 4px 8px;
  border-radius: 4px;
  color: #fff;
  font-size: 12px;
  margin-right: 10px;
}
.badge.basic { background-color: #909399; }
.badge.multimodal { background-color: var(--accent-color); }
.badge.comparison { background-color: #67C23A; }
.title { font-weight: bold; font-size: var(--font-subtitle); }

.pipeline-row {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 16px;
  overflow-x: auto;
  padding-bottom: 10px;
}
.image-box-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 200px;
  flex-shrink: 0;
}
.img-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}
.image-box {
  width: 200px;
  height: 150px;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid #ebeef5;
}
.pipeline-img {
  width: 100%;
  height: 100%;
  background: #f5f7fa;
}
.placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  font-size: 12px;
}
.arrow {
  font-size: 20px;
  color: #C0C4CC;
  font-weight: bold;
}
.arrow.with-text {
  display: flex;
  flex-direction: column;
  align-items: center;
  font-size: 12px;
  color: var(--accent-color);
}
.text-hint {
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 10px;
  color: #666;
}
.arrow-icon {
  font-size: 20px;
  color: #C0C4CC;
}

.metrics-panel {
  margin-left: auto;
  background: #f9fafc;
  padding: 12px;
  border-radius: 8px;
  min-width: 180px;
  border: 1px solid #ebeef5;
}
.metric-item {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-size: 13px;
}
.metric-item:last-child { margin-bottom: 0; }
.value { font-weight: bold; }
.value.highlight { color: var(--accent-color); }

.export-section {
  display: flex;
  justify-content: center;
  margin-top: 32px;
  margin-bottom: 32px;
}
.export-btn {
  width: 240px;
  font-weight: bold;
  box-shadow: 0 4px 12px rgba(64, 158, 255, 0.3);
}

.winner { color: #67C23A; font-weight: bold; }

@media print {
  .aside-panel, .aside { display: none; }
  .el-header { display: none; }
  .main-area { padding: 0; width: 100%; }
  .scheme-card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }
  .export-section { display: none; }
}
</style>
