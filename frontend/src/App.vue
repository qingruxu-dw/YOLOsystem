<template>
  <div id="app">
    <Navigation />
    <router-view />
  </div>
</template>

<script>
import Navigation from "./components/layout/Navigation.vue";

export default {
  name: "App",
  components: {
    Navigation,
  },
};
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
