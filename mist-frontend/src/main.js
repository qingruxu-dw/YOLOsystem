import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { createPinia } from 'pinia'

const app = createApp(App)
const pinia = createPinia()// 创建 Pinia 实例

app.use(pinia) // 注册 Pinia 插件
app.use(router)
app.mount('#app')