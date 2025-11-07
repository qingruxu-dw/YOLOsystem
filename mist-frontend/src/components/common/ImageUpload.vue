<template>
  <el-upload
    class="upload-area"
    drag
    :auto-upload="false"
    :limit="limit"
    :on-change="handleChange"
    :accept="accept"
  >
    <i class="el-icon-upload"></i>
    <div class="el-upload__text">将文件拖到此处，或<em>点击上传</em></div>
    <template #tip>
      <div class="el-upload__tip">支持 JPG / PNG 等常见图片格式</div>
    </template>
  </el-upload>
</template>

<script setup>
import { defineProps, defineEmits } from 'vue'

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
</style>