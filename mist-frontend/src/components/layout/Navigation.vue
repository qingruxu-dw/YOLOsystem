<template>
  <header class="navigation">
    <div class="brand">清视智监-无人机图像去雾目标检测系统</div>
    <div class="mode">
      <span class="label">模式：</span>
      <el-select v-model="currentMode" size="small" @change="onModeChange" style="width:160px">
        <el-option label="基础模式" value="basic" />
        <el-option label="多模态模式" value="multimodal" disabled />
        <el-option label="对比模式" value="compare" disabled />
      </el-select>
    </div>
  </header>
  
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const currentMode = ref('basic')

const onModeChange = (val) => {
  if (val === 'basic') router.push('/baseHome')
}
</script>

<style scoped>
.navigation {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background-color: #f6f9ff;
  border-bottom: 1px solid rgba(30, 128, 255, 0.15);
  margin-bottom: 12px;
}
.brand {
  font-weight: 700;
  color: var(--el-color-primary);
  letter-spacing: 0.5px;
}
.mode { display: flex; align-items: center; }
.label { margin-right: 8px; color: var(--el-color-info); font-weight: 500; font-size: var(--font-subtitle); }
</style>
