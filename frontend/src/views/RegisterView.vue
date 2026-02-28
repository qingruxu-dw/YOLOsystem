<template>
  <AuthSplitLayout>
    <el-card class="auth-card" shadow="hover">
      <template #header>
        <div class="auth-title">用户注册</div>
      </template>

      <el-form ref="formRef" :model="form" :rules="rules" label-position="top">
        <el-form-item label="用户名" prop="username">
          <el-input v-model="form.username" placeholder="请输入用户名" autocomplete="username" />
        </el-form-item>
        <el-form-item label="密码" prop="password">
          <el-input v-model="form.password" type="password" placeholder="请输入密码" autocomplete="new-password" show-password />
        </el-form-item>
        <el-form-item label="确认密码" prop="confirmPassword">
          <el-input v-model="form.confirmPassword" type="password" placeholder="请再次输入密码" autocomplete="new-password" show-password />
        </el-form-item>
        <el-alert v-if="auth.errorMessage" :title="auth.errorMessage" type="error" :closable="false" show-icon class="mb12" />
        <el-button type="primary" :loading="auth.loading" style="width: 100%" @click="onSubmit">注册</el-button>
      </el-form>

      <div class="auth-footer">
        <el-text type="info">已有账号？</el-text>
        <el-link type="primary" @click="toLogin">去登录</el-link>
      </div>
    </el-card>
  </AuthSplitLayout>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import AuthSplitLayout from '@/components/auth/AuthSplitLayout.vue'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()

const formRef = ref()
const form = reactive({
  username: '',
  password: '',
  confirmPassword: ''
})

const confirmValidator = (rule, value, callback) => {
  if (!value) return callback(new Error('请确认密码'))
  if (value !== form.password) return callback(new Error('两次密码输入不一致'))
  callback()
}

const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 32, message: '用户名长度需在 3-32 之间', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 128, message: '密码长度需在 6-128 之间', trigger: 'blur' }
  ],
  confirmPassword: [
    { validator: confirmValidator, trigger: 'blur' }
  ]
}

const onSubmit = async () => {
  const ok = await formRef.value?.validate?.().catch(() => false)
  if (!ok) return
  const user = await auth.register(form.username.trim(), form.password)
  if (!user) return
  const redirect = typeof route.query.redirect === 'string' ? route.query.redirect : '/baseHome'
  router.replace(redirect)
}

const toLogin = () => {
  router.push({ path: '/login', query: { redirect: route.query.redirect } })
}
</script>

<style scoped>
.auth-card {
  width: 420px;
  max-width: 95vw;
  text-align: left;
}

.auth-title {
  font-size: 18px;
  font-weight: 700;
  color: #1a1a1a;
}

.auth-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 14px;
}

.mb12 {
  margin-bottom: 12px;
}
</style>
