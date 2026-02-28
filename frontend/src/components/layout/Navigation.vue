<template>
  <header class="navigation">
    <div class="brand-container">
      <img src="@/assets/images/tubiao.png" alt="Logo" class="logo" />
      <div class="brand">雾里看花</div>
      <div class="divider"></div>
      <div class="subtitle">基于多模态去雾的无人机航拍图像增强与目标检测系统</div>
    </div>
    <div class="right">
      <div class="history-btn-wrapper" v-if="showModeSwitch">
         <el-button type="primary" plain :icon="Clock" @click="emitOpenHistory">
           历史记录
         </el-button>
      </div>
      <div class="mode" v-if="showModeSwitch">
        <el-radio-group v-model="currentMode" size="small" @change="onModeChange" class="mode-switch">
          <el-radio-button label="basic">基础模式</el-radio-button>
          <el-radio-button label="multimodal">多模态模式</el-radio-button>
          <el-radio-button label="compare">对比模式</el-radio-button>
        </el-radio-group>
      </div>
      <div class="auth">
        <template v-if="auth.isAuthed">
          <el-dropdown trigger="click">
            <span class="el-dropdown-link user-profile">
              <el-avatar :size="32" :icon="UserFilled" class="user-avatar" />
              <span class="username">{{ auth.user?.username || '用户' }}</span>
              <el-icon class="el-icon--right"><arrow-down /></el-icon>
            </span>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item @click="onLogout">退出登录</el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </template>
        <template v-else>
          <el-button size="small" plain @click="toLogin">登录</el-button>
          <el-button size="small" type="primary" @click="toRegister">注册</el-button>
        </template>
      </div>
    </div>
  </header>
  
</template>

<script setup>
import { computed, ref, watch, defineEmits } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { Clock, UserFilled, ArrowDown } from '@element-plus/icons-vue'

const router = useRouter()
const route = useRoute()
const currentMode = ref('basic')
const auth = useAuthStore()
const emit = defineEmits(['open-history'])

const showModeSwitch = computed(() => !['/login', '/register'].includes(route.path))

watch(() => route.path, (path) => {
  if (path.includes('multimodal')) {
    currentMode.value = 'multimodal'
  } else if (path.includes('compare')) {
    currentMode.value = 'compare'
  } else {
    currentMode.value = 'basic'
  }
}, { immediate: true })

const onModeChange = (val) => {
  if (val === 'basic') router.push('/baseHome')
  if (val === 'multimodal') router.push('/multimodal')
  if (val === 'compare') router.push('/compare')
}

const emitOpenHistory = () => {
    emit('open-history')
}

const toLogin = () => {
  router.push({ path: '/login', query: { redirect: route.fullPath } })
}

const toRegister = () => {
  router.push({ path: '/register', query: { redirect: route.fullPath } })
}

const onLogout = () => {
  auth.logout()
  router.replace('/login')
}
</script>

<style scoped>
.navigation {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  height: 64px;
  background-color: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  position: sticky;
  top: 0;
  z-index: 100;
  border-bottom: 1px solid #f0f0f0;
}

.brand-container {
  display: flex;
  align-items: center;
  gap: 12px;
}

.right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  height: 62px;
  width: auto;
  object-fit: contain;
}

.brand {
  font-weight: 700;
  font-size: 22px;
  color: #1a1a1a;
  letter-spacing: 0.5px;
}

.divider {
  width: 1px;
  height: 16px;
  background-color: #dcdfe6;
}

.subtitle {
  font-size: 14px;
  color: #606266;
  font-weight: 400;
}

.history-btn-wrapper {
  margin-right: 12px;
}

.mode {
  display: flex;
  align-items: center;
}

.auth {
  display: flex;
  align-items: center;
  gap: 10px;
}

.auth :deep(.el-dropdown .el-button) {
  font-size: 16px;
}

.el-dropdown-link {
  cursor: pointer;
  display: flex;
  align-items: center;
  color: var(--el-color-primary);
}

.user-profile {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  border-radius: 4px;
  transition: background-color 0.2s;
}

.user-profile:hover {
  background-color: #f5f7fa;
}

.username {
  font-size: 14px;
  font-weight: 500;
  color: #606266;
  max-width: 100px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.user-avatar {
  background-color: var(--el-color-primary-light-9);
  color: var(--el-color-primary);
}

.mode-switch :deep(.el-radio-button__inner) {
  padding: 8px 16px;
  font-weight: 500;
  font-size: 14px;
}

/* 响应式适配 */
@media (max-width: 768px) {
  .subtitle, .divider {
    display: none;
  }
  .navigation {
    padding: 0 12px;
  }
  .brand {
    font-size: 16px;
  }
}
</style>
