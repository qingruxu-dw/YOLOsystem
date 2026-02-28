import axios from 'axios'

// Axios 实例，基地址指向后端 API 网关
const api = axios.create({
  baseURL: '/api',
  timeout: 420000  // 延长至 7 分钟 (CoA 去雾算法耗时较长)
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers = config.headers || {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

api.interceptors.response.use(
  (resp) => resp,
  (err) => {
    if (err?.response?.status === 401) {
      try { localStorage.removeItem('auth_token') } catch (e) {}
      try { localStorage.removeItem('auth_user') } catch (e) {}
    }
    return Promise.reject(err)
  }
)

// 文件上传接口：/api/upload
export function uploadImage(file, mode = 'normal', textPrompt = '', dehazeStrength = 0.5) {
  const form = new FormData()
  form.append('image', file)
  form.append('mode', mode)
  if (textPrompt) {
    form.append('text_prompt', textPrompt)
  }
  if (dehazeStrength !== undefined && dehazeStrength !== null) {
    form.append('dehaze_strength', dehazeStrength)
  }
  return api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

// 读取检测结果TXT解析后的结构化数据：/api/detections/{timestamp}/{basename}
export function getDetections(timestamp, basename) {
  return api.get(`/detections/${encodeURIComponent(timestamp)}/${encodeURIComponent(basename)}`)
}

export function authRegister(username, password) {
  return api.post('/auth/register', { username, password })
}

export function authLogin(username, password) {
  return api.post('/auth/login', { username, password })
}

export function authMe() {
  return api.get('/auth/me')
}

export default api
