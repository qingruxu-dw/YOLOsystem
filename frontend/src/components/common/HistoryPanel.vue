<template>
  <el-drawer
    v-model="visible"
    :title="title"
    direction="rtl"
    :size="drawerWidth"
    @open="fetchHistory"
    class="history-drawer"
  >
    <div class="drawer-container">
      <!-- 拖拽手柄 -->
      <div class="resize-handle" @mousedown="startResize"></div>
      
      <!-- 内容区域 -->
      <div v-loading="loading" class="history-list">
        <div v-if="history.length === 0" class="empty-text">
          暂无历史记录
        </div>
        
        <div 
          v-for="item in history" 
          :key="item.id || item.filename" 
          class="history-item"
          @click="selectItem(item)"
        >
          <div class="history-thumb">
            <el-image 
              v-if="mode !== 'comparison'"
              :src="item.detected || item.original" 
              fit="cover" 
              class="thumb-img"
            >
              <template #error>
                <div class="image-slot">
                  <el-icon><Picture /></el-icon>
                </div>
              </template>
            </el-image>
            <div v-else class="comparison-icon">
              <el-icon :size="24"><Files /></el-icon>
            </div>
          </div>
          <div class="history-info">
            <div class="filename" :title="item.original_filename || item.filename">
              {{ item.original_filename || item.filename }}
            </div>
            <div class="time">{{ formatTime(item.time) }}</div>
          </div>
        </div>
      </div>
    </div>
  </el-drawer>
</template>

<script setup>
import { ref, defineProps, defineExpose, defineEmits, onUnmounted } from 'vue'
import { Picture, Files } from '@element-plus/icons-vue'
import api from '@/utils/api'

const props = defineProps({
  mode: {
    type: String,
    required: true
  },
  title: {
    type: String,
    default: '处理历史'
  }
})

const emit = defineEmits(['select'])

const visible = ref(false)
const loading = ref(false)
const history = ref([])
const drawerWidth = ref(450)

// 拖拽相关逻辑
const isResizing = ref(false)
let startX = 0
let startWidth = 0

const startResize = (e) => {
  isResizing.value = true
  startX = e.clientX
  startWidth = drawerWidth.value
  
  document.addEventListener('mousemove', doResize)
  document.addEventListener('mouseup', stopResize)
  // 防止选中文本
  document.body.style.userSelect = 'none'
}

const doResize = (e) => {
  if (!isResizing.value) return
  const deltaX = startX - e.clientX // 向左移动是正数（增加宽度），因为是 RTL
  const newWidth = startWidth + deltaX
  // 限制最小和最大宽度
  if (newWidth >= 300 && newWidth <= 800) {
    drawerWidth.value = newWidth
  }
}

const stopResize = () => {
  isResizing.value = false
  document.removeEventListener('mousemove', doResize)
  document.removeEventListener('mouseup', stopResize)
  document.body.style.userSelect = ''
}

const open = () => {
  visible.value = true
}

const fetchHistory = async () => {
  loading.value = true
  try {
    const resp = await api.get('/history', {
      params: { mode: props.mode }
    })
    if (resp.data.success) {
      history.value = resp.data.history
    }
  } catch (error) {
    console.error('Failed to fetch history:', error)
  } finally {
    loading.value = false
  }
}

const selectItem = (item) => {
  emit('select', item)
  visible.value = false
}

const formatTime = (timeStr) => {
  if (!timeStr) return ''
  try {
      // Backend returns UTC usually or local string. 
      // If it's a full ISO string or similar, convert to locale
      const date = new Date(timeStr)
      return date.toLocaleString()
  } catch (e) {
      return timeStr
  }
}

defineExpose({ open })
</script>

<style scoped>
/* 覆盖 el-drawer 默认 padding，以便让 handle 贴边 */
:deep(.history-drawer .el-drawer__body) {
  padding: 0;
  overflow: visible; /* 允许 handle 显示（如果在外部） */
}

.drawer-container {
  position: relative;
  height: 100%;
  width: 100%;
  padding: 20px; /* 恢复内部 padding */
  box-sizing: border-box;
}

.resize-handle {
  position: absolute;
  left: 0;
  top: 0;
  bottom: 0;
  width: 6px;
  cursor: col-resize;
  background: transparent;
  transition: background 0.2s;
  z-index: 100;
}
.resize-handle:hover,
.resize-handle:active {
  background: rgba(64, 158, 255, 0.3); /* 悬停时显示淡淡的蓝色 */
}

.history-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
  overflow-y: auto;
}
.history-item {
  display: flex;
  gap: 12px;
  padding: 10px;
  border-radius: 8px;
  border: 1px solid #ebeef5;
  cursor: pointer;
  transition: all 0.2s;
}
.history-item:hover {
  background-color: #f5f7fa;
  border-color: #dcdfe6;
}
.history-thumb {
  width: 80px;
  height: 60px;
  border-radius: 4px;
  overflow: hidden;
  flex-shrink: 0;
  background-color: #f0f2f5;
  display: flex;
  align-items: center;
  justify-content: center;
}
.comparison-icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: #909399;
}
.thumb-img {
  width: 100%;
  height: 100%;
}
.history-info {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.filename {
  font-weight: 500;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  margin-bottom: 4px;
}
.time {
  font-size: 12px;
  color: #909399;
}
.empty-text {
  text-align: center;
  color: #909399;
  padding: 20px;
}
.image-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  font-size: 20px;
  color: #909399;
}
</style>