import axios from 'axios'

// Axios 实例，基地址指向后端 API 网关
const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// 文件上传接口：/api/upload
export function uploadImage(file, mode = 'normal', textPrompt = '') {
  const form = new FormData()
  form.append('image', file)
  form.append('mode', mode)
  if (textPrompt) {
    form.append('text_prompt', textPrompt)
  }
  return api.post('/upload', form, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

// 图像处理任务接口：/api/process（支持基础与多模态，通过 mode 与 text_prompt 区分）
export function startProcess({ mode = 'basic', dehaze_strength = 0.5, text_prompt = '' }) {
  return api.post('/process', {
    mode,
    params: { dehaze_strength },
    text_prompt
  })
}

// 结果查询接口：/api/task/{task_id}，轮询状态与结果
export function getTask(taskId) {
  return api.get(`/task/${taskId}`)
}

// 对比实验接口：/api/compare
export function compareTasks(payload) {
  return api.post('/compare', payload)
}

// 读取检测结果TXT解析后的结构化数据：/api/detections/{timestamp}/{basename}
export function getDetections(timestamp, basename) {
  return api.get(`/detections/${encodeURIComponent(timestamp)}/${encodeURIComponent(basename)}`)
}

export default api
