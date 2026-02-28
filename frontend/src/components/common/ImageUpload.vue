<template>
  <el-upload
    class="upload-area"
    drag
    :auto-upload="false"
    :limit="limit"
    :on-change="handleChange"
    :accept="accept"
  >
    <div class="upload-content">
      <el-icon class="el-icon--upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        拖拽或<em>点击上传</em>
      </div>
    </div>
  </el-upload>
</template>

<script setup>
import { defineProps, defineEmits } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'

const props = defineProps({
  accept: { type: String, default: 'image/*' },
  limit: { type: Number, default: 1 }
})

const emit = defineEmits(['change'])

const handleChange = (file) => {
  emit('change', file)
}
</script>

<style scoped>
.upload-area { width: 100%; }
:deep(.el-upload-dragger) {
  padding: 10px !important;
  height: auto !important;
  display: flex;
  justify-content: center;
  align-items: center;
}
.upload-content {
  display: flex;
  align-items: center;
  gap: 8px;
}
.el-icon--upload {
  font-size: 20px;
  color: var(--el-text-color-secondary);
  margin-bottom: 0;
}
.el-upload__text {
  font-size: 13px;
  line-height: 1;
}
</style>