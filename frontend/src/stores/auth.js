import { defineStore } from 'pinia'
import { authLogin, authRegister, authMe } from '@/utils/api'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: (() => {
      try { return localStorage.getItem('auth_token') || '' } catch (e) { return '' }
    })(),
    user: (() => {
      try { return JSON.parse(localStorage.getItem('auth_user') || 'null') } catch (e) { return null }
    })(),
    loading: false,
    errorMessage: ''
  }),
  getters: {
    isAuthed: (state) => Boolean(state.token)
  },
  actions: {
    setSession({ token, user }) {
      this.token = token || ''
      this.user = user || null
      try { localStorage.setItem('auth_token', this.token) } catch (e) {}
      try { localStorage.setItem('auth_user', JSON.stringify(this.user)) } catch (e) {}
    },
    clearSession() {
      this.token = ''
      this.user = null
      try { localStorage.removeItem('auth_token') } catch (e) {}
      try { localStorage.removeItem('auth_user') } catch (e) {}
    },
    async fetchMe() {
      if (!this.token) return null
      this.loading = true
      this.errorMessage = ''
      try {
        const { data } = await authMe()
        if (data?.success) {
          this.setSession({ token: this.token, user: data.user })
          return data.user
        }
        this.clearSession()
        return null
      } catch (err) {
        this.clearSession()
        this.errorMessage = err?.response?.data?.error || err.message || '获取用户信息失败'
        return null
      } finally {
        this.loading = false
      }
    },
    async login(username, password) {
      this.loading = true
      this.errorMessage = ''
      try {
        const { data } = await authLogin(username, password)
        if (!data?.success) {
          this.errorMessage = data?.error || '登录失败'
          return null
        }
        this.setSession({ token: data.token, user: data.user })
        return data.user
      } catch (err) {
        this.errorMessage = err?.response?.data?.error || err.message || '登录失败'
        return null
      } finally {
        this.loading = false
      }
    },
    async register(username, password) {
      this.loading = true
      this.errorMessage = ''
      try {
        const { data } = await authRegister(username, password)
        if (!data?.success) {
          this.errorMessage = data?.error || '注册失败'
          return null
        }
        this.setSession({ token: data.token, user: data.user })
        return data.user
      } catch (err) {
        this.errorMessage = err?.response?.data?.error || err.message || '注册失败'
        return null
      } finally {
        this.loading = false
      }
    },
    logout() {
      this.clearSession()
    }
  }
})

