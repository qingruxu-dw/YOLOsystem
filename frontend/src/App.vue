<template>
  <div id="app">
    <Navigation @open-history="onOpenHistory" />
    <router-view v-slot="{ Component }">
      <component :is="Component" ref="viewRef" />
    </router-view>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import Navigation from "./components/layout/Navigation.vue";

const viewRef = ref(null)

const onOpenHistory = () => {
  // viewRef is the component instance
  // When using <script setup>, methods are not exposed by default
  // But we use defineExpose in BaseView/MultimodalView/ComparisonView
  
  if (viewRef.value) {
      // Direct access if exposed
      if (typeof viewRef.value.openHistory === 'function') {
          viewRef.value.openHistory()
          return
      }
      // Access via internal instance if needed (sometimes helps in dev mode)
      if (viewRef.value.$ && viewRef.value.$.exposed && viewRef.value.$.exposed.openHistory) {
           viewRef.value.$.exposed.openHistory()
      }
  }
}
</script>

<style>
:root {
  /* 主题色：科技蓝 */
  --el-color-primary: #1E80FF;
  /* 辅助色：成功/处理中/错误 */
  --el-color-success: #16A34A; /* 绿色 */
  --el-color-warning: #F59E0B; /* 橙色，处理中/警告 */
  --el-color-danger: #EF4444;  /* 红色，错误 */
  --el-color-info: #5B6C85;    /* 次级信息 */

  /* 字体层级 */
  --font-title: 18px;
  --font-subtitle: 15px;
  --font-body: 14px;
}

#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 8px;
}
html, body { font-size: var(--font-body); }
</style>
